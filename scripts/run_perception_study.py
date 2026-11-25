"""Run a resumable shard of the preregistered perception study.

python scripts/run_perception_study.py nominal servo_depth --workers 8
python scripts/run_perception_study.py wide student --checkpoint models/student.pt
"""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dap.experiment import CONFIGS, PERCEPTION_METHODS, Condition  # noqa: E402

CORE_FILES = (
    "dap/bank.py", "dap/baselines.py", "dap/camera.py", "dap/depth_noise.py",
    "dap/envs/dap_env.py", "dap/eval_scenarios.py", "dap/estimator.py", "dap/experiment.py", "dap/feasibility.py",
    "dap/ik.py", "dap/metrics.py", "dap/partner.py", "dap/perception.py", "dap/planner.py",
    "dap/policy_eval.py", "dap/scenario.py", "dap/scene.py", "dap/scoring.py", "dap/trial.py",
    "models/dap_hand.sdf", "scripts/run_experiment.py", "scripts/run_perception_study.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(cfg, method, checkpoint):
    from dap.baselines import SERVO_DEPTH_CAMERA
    from dap.eval_scenarios import DEFAULT_ARCHIVE

    extra_files, artifacts = [], {}
    packages = ["drake", "numpy", "scipy", "gymnasium"]
    if method == "student":
        extra_files.append("dap/student.py")
        artifacts = {"weights": digest(checkpoint)}
        packages.append("torch")
    elif method == "teacher":
        extra_files.append("dap/teacher.py")
        artifacts = {suffix: digest(checkpoint.with_suffix(suffix)) for suffix in (".zip", ".norm.pkl")}
        packages += ["torch", "stable_baselines3"]
    return dict(config=dataclasses.asdict(cfg), method=method,
                camera=dataclasses.asdict(SERVO_DEPTH_CAMERA), depth_noise="med",
                sources={f: digest(ROOT / f) for f in (*CORE_FILES, *extra_files)},
                checkpoints=artifacts,
                scenario_archive=(digest(DEFAULT_ARCHIVE / f"{cfg.scenario}.jsonl.gz")
                                  if cfg.split == "eval" else None),
                versions={p: importlib.metadata.version(p) for p in packages},
                git=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())


def atomic_json(path, record, *, immutable=False):
    """Publish complete records, including when workers start on overlapping shards."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            json.dump(record, temp, allow_nan=False, sort_keys=True)
            temp.write("\n")
            temp.flush()
            if immutable:
                try:
                    os.link(temp_path, path)
                except FileExistsError:
                    # Unrelated new commits do not change the byte-identical experiment.
                    previous = json.loads(path.read_text())
                    if {k: v for k, v in previous.items() if k != "git"} != {
                        k: v for k, v in json.loads(temp_path.read_text()).items() if k != "git"
                    }:
                        raise ValueError(f"{path}: different experiment or checkpoint")
            else:
                os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def policy_for(method, checkpoint):
    import torch

    torch.set_num_threads(1)
    if method == "student":
        from dap.student import load_student

        policy = load_student(checkpoint)
        if policy.obs_mode != "pixels":
            raise ValueError("the deployed student must consume the pixels observation")
        return policy
    from dap.teacher import load_teacher

    return load_teacher(checkpoint)


def run_seed(job):
    from dap.metrics import TrialResult
    from dap.scenario import SCENARIO_PRESETS, ScenarioError, draw_peak_speed
    from dap.eval_scenarios import load_frozen_scenario

    cfg, method, seed, checkpoint = job
    condition = Condition(method, 1.)
    frozen = load_frozen_scenario(cfg, seed) if cfg.split == "eval" else None
    policy = policy_for(method, checkpoint) if checkpoint is not None else None
    try:
        if policy is None:
            from dap.baselines import CAMERA_BASELINES, SERVO_DEPTH_CAMERA
            from dap.trial import run_trial
            from scripts.run_experiment import _factory

            camera = SERVO_DEPTH_CAMERA if method in CAMERA_BASELINES else None
            result = run_trial(cfg, condition, seed, camera=camera, controller_factory=_factory(method),
                               frozen_scenario=frozen)
        else:
            from dap.policy_eval import run_policy_trial

            result = run_policy_trial(cfg, condition, seed, policy, allow_privileged=method == "teacher",
                                      frozen_scenario=frozen)
    except Exception as exc:
        result = TrialResult(seed=seed, **condition.key())
        result.outcome = "error"
        result.error_kind = "scenario" if isinstance(exc, ScenarioError) else "harness"
        result.target_peak_speed = draw_peak_speed(cfg.rngs(seed)["scenario"], SCENARIO_PRESETS[cfg.scenario])
        result.error = traceback.format_exc(limit=5)
    return dict(result.to_dict(), scenario_sha256=frozen.sha256 if frozen else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=("nominal", "wide"))
    parser.add_argument("method", choices=PERCEPTION_METHODS)
    parser.add_argument("--checkpoint", type=Path, help="student .pt, or teacher checkpoint prefix")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "perception_study")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0, help="first seed index within the preset")
    parser.add_argument("--count", type=int, help="number of seeds in this shard")
    parser.add_argument("--validation", action="store_true", help="smoke check on a separate validation stream")
    args = parser.parse_args()
    if (args.method in ("student", "teacher")) != (args.checkpoint is not None):
        parser.error("supply --checkpoint exactly for student and teacher")
    cfg = CONFIGS[f"perception_{args.preset}"]
    if args.validation:
        cfg = dataclasses.replace(cfg, split="val", first_seed=400)
    count = cfg.n_seeds - args.offset if args.count is None else args.count
    if args.workers < 1 or args.offset < 0 or count <= 0 or args.offset + count > cfg.n_seeds:
        parser.error("invalid worker count or shard range")
    out = args.out / cfg.split / cfg.scenario / args.method
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "manifest.json", manifest(cfg, args.method, args.checkpoint), immutable=True)
    jobs = []
    for seed in cfg.seeds()[args.offset:args.offset + count]:
        path = out / f"{seed}.json"
        if not path.exists() or json.loads(path.read_text()).get("error_kind") == "harness":
            jobs.append((cfg, args.method, seed, args.checkpoint))
    if not jobs:
        print("All requested records already exist.", flush=True)
        return
    with multiprocessing.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        for i, record in enumerate(pool.imap_unordered(run_seed, jobs), 1):
            atomic_json(out / f"{record['seed']}.json", record)
            if record["error_kind"] == "harness":
                raise RuntimeError(record["error"])
            if i % 20 == 0 or i == len(jobs):
                print(f"{args.preset} {args.method}: {i}/{len(jobs)} recorded", flush=True)


if __name__ == "__main__":
    main()
