"""Collect supervised depth trajectories and distill a privileged teacher.

Round zero collects expert or teacher trajectories on the nominal training
stream. Later rounds visit the frozen student's states and label them with
the same teacher. Fit on the union of all rounds. Each example stores the
actual three delayed camera frames (pooled), proprioception and the partner
truth at capture time. Only the offline loss can access that truth.
"""

import argparse
import hashlib
import fcntl
import os
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _source(rollout, checkpoint):
    if rollout == "expert":
        return {"policy": rollout}
    if checkpoint is None:
        raise ValueError(f"{rollout} requires --checkpoint")
    paths = ([checkpoint] if rollout in ("student", "ik") else
             [checkpoint.with_suffix(".zip"), checkpoint.with_suffix(".norm.pkl")])
    return {"policy": rollout, "sha256": [_digest(p) for p in paths]}


def _init_worker(args):
    import torch
    from dap.envs.dap_env import DapEnv
    torch.set_num_threads(1)
    global _env, _bank_specs, _rollout, _policy, _directory
    _env = DapEnv(obs_mode="pixels", scenario="nominal", bank=args.bank)
    _bank_specs = None if _env.bank is None else _env.bank.specs
    _rollout, _directory = args.rollout, args.out
    if _rollout == "expert":
        from dap.teacher import ServoExpert
        _policy = ServoExpert()
    elif _rollout == "ik":
        from dap.pixel_policy import PixelPolicy
        _policy = PixelPolicy(args.checkpoint)
    elif _rollout == "teacher":
        from dap.teacher import load_teacher
        _policy = load_teacher(args.checkpoint)
    else:
        from dap.student import load_student
        _policy = load_student(args.checkpoint)


def _collect_episode(seed):
    from dap.student import pack_depth
    from dap.scenario import ScenarioError
    if _bank_specs is None:
        try:
            obs, info = _env.reset(options={"trial_seed": seed, "split": "train"})
        except ScenarioError:
            return (seed, 0, "scenario_error")
    else:
        # Visit every entry once before cycling. The episode seed also fixes
        # the sensor RNG, so repeated visits can see different noisy images.
        _env.bank.specs = [_bank_specs[seed % len(_bank_specs)]]
        obs, info = _env.reset(seed=seed)
    if hasattr(_policy, "reset"):
        _policy.reset()
    frames, privileged = [], []
    while not _env.event and _env.k < _env.k_end:
        if np.any(obs["depth"][-1]):
            frames.append(pack_depth(obs["depth"]))
            privileged.append(info["privileged"])
        if _rollout in ("student", "ik"):
            action = _policy(obs)
        else:
            action = _policy({"state": info["privileged"]})
        obs, _, terminated, truncated, info = _env.step(action)
        if terminated or truncated:
            break
    dest = _directory / f"{seed:08d}.npz"
    tmp = dest.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, depth=np.asarray(frames, np.int16).reshape(-1, 3, 60, 106),
                        privileged=np.asarray(privileged, np.float32).reshape(-1, 43),
                        scenario_id=str(info["scenario_id"]), success=info["success"])
    tmp.replace(dest)
    return seed, len(frames), info["event"] or "timeout"


def collect(args):
    from dap.bank import provenance
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    spec = {"format": 1, "split": "train", "preset": "nominal",
            "bank_sha256": _digest(args.bank) if args.bank else None,
            "rollout": _source(args.rollout, args.checkpoint)}
    # Slurm array tasks may create the same corpus simultaneously.
    with open(args.out / ".manifest.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest_path.is_file():
            stored = json.loads(manifest_path.read_text())
            if stored["spec"] != spec:
                raise ValueError(f"training data in {args.out} has different collection settings")
        else:
            tmp = manifest_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"spec": spec, "provenance": provenance()}, indent=2))
            tmp.replace(manifest_path)
    seeds = [s for s in range(args.first, args.first + args.count)
             if not (args.out / f"{s:08d}.npz").exists()]
    if not seeds:
        return
    with ProcessPoolExecutor(min(args.workers, len(seeds)), initializer=_init_worker,
                             initargs=(args,)) as pool:
        n = 0
        for i, (seed, steps, outcome) in enumerate(pool.map(_collect_episode, seeds), 1):
            n += steps
            if outcome == "scenario_error":
                print(f"training seed {seed}: scenario sampling failed", flush=True)
            if i % 16 == 0 or i == len(seeds):
                print(f"{i}/{len(seeds)} training trajectories, {n} labeled states", flush=True)


def export_actor(args):
    """Extract the deterministic SB3 actor so fitting can also run outside Drake."""
    import torch
    from dap.student import DepthStudent
    from dap.teacher import load_teacher
    teacher = load_teacher(args.checkpoint)
    student = DepthStudent()
    actor = list(teacher.policy.mlp_extractor.policy_net) + [teacher.policy.action_net]
    student.actor.load_state_dict(torch.nn.Sequential(*actor).state_dict())
    stats = {"actor": student.actor.state_dict(),
             "mean": torch.tensor(teacher.mean, dtype=torch.float32),
             "scale": torch.tensor(np.sqrt(teacher.var + 1e-8), dtype=torch.float32),
             "source": _source("teacher", args.checkpoint)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, args.out)


def _dataset(dirs):
    import torch
    paths, metadata = [], []
    for directory in dirs:
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["spec"]["split"] != "train":
            raise ValueError(f"{directory} is not training data")
        parts = sorted(directory.glob("[0-9]" * 8 + ".npz"))
        if not parts:
            raise ValueError(f"no episodes in {directory}")
        paths.extend(parts)
        checksum = hashlib.sha256()
        for p in parts:
            checksum.update(f"{p.name} {_digest(p)}\n".encode())
        metadata.append({"directory": str(directory), "spec": manifest["spec"],
                         "sha256_of_named_episode_hashes": checksum.hexdigest(),
                         "episode_seeds": [int(p.stem) for p in parts]})
    # Allocate once after reading small state arrays, so fitting does not
    # require two full copies of the images in RAM.
    states = []
    for p in paths:
        with np.load(p, allow_pickle=False) as row:
            states.append(row["privileged"])
    privileged = torch.from_numpy(np.concatenate(states))
    depth = torch.empty((len(privileged), 3, 60, 106), dtype=torch.int16)
    start = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as row:
            x = torch.from_numpy(row["depth"])
            depth[start:start + len(x)] = x
            start += len(x)
    return depth, privileged, metadata


def fit(args):
    import torch
    from torch.nn import functional as F
    from dap.student import DepthStudent, load_student
    if args.out.exists():
        raise ValueError(f"checkpoint already exists: {args.out}")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    depth, state, data_info = _dataset(args.data)
    model = load_student(args.init, allow_pose_only=True) if args.init else DepthStudent()
    proprio, partner = state[:, :31], state[:, 31:]
    # Scaling is measured on the training corpus only. Retain it on warm
    # starts: these buffers are part of the network's coordinate system.
    if not args.init:
        for key, values in [("proprio", proprio), ("partner", partner)]:
            getattr(model, key + "_mean").copy_(values.mean(0))
            getattr(model, key + "_scale").copy_(values.std(0, unbiased=False).clamp_min(.03))
    teacher_hash = None
    if args.teacher:
        actor = torch.load(args.teacher, map_location="cpu", weights_only=True)
        model.actor.load_state_dict(actor["actor"])
        model.actor_mean.copy_(actor["mean"])
        model.actor_scale.copy_(actor["scale"])
        teacher_hash = actor["source"]
    model.to(device)
    model.encoder.to(memory_format=torch.channels_last)
    model.train()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr, weight_decay=.001)
    # A training-time auxiliary pose loss supplies a spatial teaching signal.
    # The action target is computed on exactly the same arm state and the
    # latency-matched partner truth, by the identical frozen teacher actor.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace = []
    for epoch in range(args.epochs):
        avg = np.zeros(3)
        for idx in np.array_split(rng.permutation(len(state)),
                                  int(np.ceil(len(state) / args.batch_size))):
            x = depth[idx].to(device).float()
            if args.shift:
                dy, dx = rng.integers(-args.shift, args.shift + 1, size=2) + args.shift
                pad = F.pad(x, (args.shift,) * 4, mode="replicate")
                x = pad[:, :, dy:dy + x.shape[2], dx:dx + x.shape[3]]
            x = x.contiguous(memory_format=torch.channels_last)
            prop, truth = proprio[idx].to(device), partner[idx].to(device)
            estimated, action = model.predict(x, prop)
            pose_loss = F.smooth_l1_loss((estimated - truth) / model.partner_scale,
                                         torch.zeros_like(truth))
            if args.teacher:
                with torch.no_grad():
                    target = model.action(prop, truth)
                act_loss = F.mse_loss(action, target)
            else:
                act_loss = pose_loss.new_zeros(())
            loss = pose_loss + act_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            avg += len(idx) * np.array([float(pose_loss.detach()), float(act_loss.detach()),
                                       float(loss.detach())])
        epoch_stats = dict(zip(("pose", "action", "loss"), (avg / len(state)).tolist()))
        trace.append(epoch_stats)
        print(f"epoch {epoch+1}/{args.epochs}: {epoch_stats}", flush=True)
    metadata = {"teacher": teacher_hash, "seed": args.seed, "epochs": args.epochs,
                "lr": args.lr, "shift": args.shift, "batch_size": args.batch_size,
                "device": str(device), "torch": str(torch.__version__), "data": data_info,
                "init_sha256": _digest(args.init) if args.init else None,
                "states": len(state), "trace": trace}
    tmp = args.out.with_suffix(".tmp.pt")
    torch.save({"format": 1, "weights": model.cpu().state_dict(),
                "teacher": teacher_hash}, tmp)
    tmp.replace(args.out)
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


def _validate_one(seed):
    from dap.experiment import Condition, ExperimentConfig
    from dap.policy_eval import run_policy_trial
    cfg = ExperimentConfig(name="student_validation", split="val")
    row = run_policy_trial(cfg, Condition("student", 1.), seed, _policy).to_dict()
    if row["error"]:
        raise RuntimeError(f"validation seed {seed}: {row['error']}")
    return row


def _validation_worker(path):
    import torch
    from dap.student import load_student
    global _policy
    torch.set_num_threads(1)
    _policy = load_student(path)


def validate(args):
    from dap.scoring import dap_loss
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers, initializer=_validation_worker,
                             initargs=(args.checkpoint,)) as pool:
        rows = list(pool.map(_validate_one, range(args.first, args.first + args.count)))
    table = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    result = {"split": "val", "checkpoint_sha256": _digest(args.checkpoint),
              "success": int(table["success"].sum()), "loss": float(dap_loss(table).mean()),
              "rows": rows}
    args.out.write_text(json.dumps(result, indent=2))
    print(f"Validation: {result['success']}/{len(rows)} successes; mean loss {result['loss']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rollout", choices=("expert", "teacher", "student", "ik"), required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--bank", type=Path)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--count", type=int, default=1024)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(run=collect)
    p = sub.add_parser("export-teacher")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(run=export_actor)
    p = sub.add_parser("fit")
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--teacher", type=Path, help="exported actor; omit for pose-only pretraining")
    p.add_argument("--init", type=Path, help="previous warm start or DAgger checkpoint")
    p.add_argument("--seed", type=int, default=231)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--shift", type=int, default=1)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    p.set_defaults(run=fit)
    p = sub.add_parser("validate")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(run=validate)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
