"""Train the PPO teacher on the training bank (docs/teacher_design.md).

    python scripts/train_teacher.py --name pilot --steps 2000000 --workers 8

Every --eval-every steps the deterministic teacher is checkpointed and scored
by dap.policy_eval on VALIDATION seeds (default 0-99, the selection set;
seeds 100-299 are kept for the check against servo_gt_cam). Checkpoints are
chosen by validation success, then mean loss; evaluation seeds are never
touched here.

With --warm-start N, the actor and critic are first fitted to N episodes of
the servo expert on training scenarios (dap.teacher.warm_start), and that
policy is scored as ckpt_0. With --kick-coef, PPO's loss also pulls the
policy toward that expert's actions on the policy's own states, annealed over
--kick-steps (kickstarting; ParallelPPO.train). With --dagger-rounds R, R
DAgger rounds toward the expert follow the warm start (dap.teacher.dagger),
each scored as ckpt_dagger<k>; --steps 0 then stops there.

Writes results/teacher/<name>/: config.json (arguments, PPO and reward
settings, git commit, bank SHA-256), progress.csv (SB3's training log),
episodes.jsonl (every training episode), ckpt_<steps>.zip/.norm.pkl, and
val.jsonl (one row per validation trial, with the checkpoint's step count and
the deterministic teacher's mean |a_t - a_t-1|; not a harness score).
"""

import argparse
import dataclasses
import hashlib
import json
import sys
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402

from dap.bank import provenance  # noqa: E402
from dap.experiment import Condition, ExperimentConfig  # noqa: E402
from dap.reward import RewardConfig  # noqa: E402
from dap.scoring import dap_loss  # noqa: E402
from dap.teacher import ParallelPPO, collect_expert, dagger, warm_start  # noqa: E402


class _Recorder:
    """Passes the teacher through and keeps its actions (for smoothness)."""

    obs_mode = "privileged"

    def __init__(self, teacher):
        self.teacher, self.actions = teacher, []

    def __call__(self, obs):
        a = self.teacher(obs)
        self.actions.append(a)
        return a


def _eval_one(args):
    ckpt, seed, n_val = args
    from dap.policy_eval import run_policy_trial
    from dap.teacher import load_teacher

    cfg = ExperimentConfig(name="teacher_val", split="val", n_seeds=n_val)
    policy = _Recorder(load_teacher(ckpt))
    row = run_policy_trial(cfg, Condition("teacher", 1.0), seed, policy,
                           allow_privileged=True).to_dict()
    a = np.array([np.zeros(7), *policy.actions])  # the env starts from a zero action
    row["mean_action_change"] = float(np.mean(np.linalg.norm(np.diff(a, axis=0), axis=1)))
    return row


class ValidationCheckpoints(BaseCallback):
    """Checkpoints at the start of a rollout, so ckpt_<n> is the policy after
    the PPO update on all n steps (on_step comes before that update)."""

    def __init__(self, out, every, seeds, workers):
        super().__init__()
        self.out, self.every, self.seeds, self.workers = out, every, seeds, workers
        self.next, self.last = every, None

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        if self.model.num_timesteps >= self.next:
            self.next += self.every
            self.evaluate()

    def evaluate(self, name=None):
        """Scores the current policy as ckpt_<name> (default: the step count)."""
        n = self.last = self.model.num_timesteps if name is None else name
        ckpt = self.out / f"ckpt_{n}"
        self.model.save_checkpoint(ckpt)
        jobs = [(str(ckpt), s, len(self.seeds)) for s in self.seeds]
        with get_context("spawn").Pool(self.workers) as pool:
            rows = pool.map(_eval_one, jobs)
        with open(self.out / "val.jsonl", "a") as f:
            for r in rows:
                f.write(json.dumps(dict(r, checkpoint=n)) + "\n")
        table = {k: np.array([r[k] for r in rows]) for k in rows[0]}
        succ, loss = float(np.mean(table["success"])), float(np.nanmean(dap_loss(table)))
        self.logger.record("val/success", succ)
        self.logger.record("val/loss", loss)
        self.logger.record("val/mean_action_change", float(np.mean(table["mean_action_change"])))
        for o in ("contact", "collision", "timeout"):
            self.logger.record(f"val/frac_{o}", float(np.mean(table["outcome"] == o)))
        print(f"[val] {n} steps: success {succ:.3f}, loss {loss:.2f} on {len(rows)} "
              "validation seeds", flush=True)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, required=True, help="PPO steps (0: none)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bank", type=Path, default=ROOT / "banks" / "train_nominal_20000.pkl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500_000)
    ap.add_argument("--val-seeds", type=int, default=100, help="validation seeds 0..N-1")
    # PPO (docs/teacher_design.md section 2)
    ap.add_argument("--n-steps", type=int, default=256, help="steps per worker per iteration")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--target-kl", type=float, default=None,
                    help="stop an update's epochs early past this approx-KL (SB3; default off)")
    # Reward (dap/reward.py)
    ap.add_argument("--k", type=float, default=RewardConfig.k, help="shaping strength")
    ap.add_argument("--w-action-rate", type=float, default=RewardConfig.w_action_rate)
    # Warm start from the servo expert (dap.teacher.ServoExpert)
    ap.add_argument("--warm-start", type=int, default=0, metavar="EPISODES",
                    help="expert episodes on training scenarios to fit the actor and critic "
                         "to before PPO (0: none)")
    ap.add_argument("--warm-epochs", type=int, default=30)
    ap.add_argument("--warm-std", type=float, default=0.2, help="exploration std after it")
    ap.add_argument("--dagger-rounds", type=int, default=0,
                    help="DAgger rounds toward the expert after the warm start (dap.teacher.dagger)")
    ap.add_argument("--dagger-steps", type=int, default=50_000, help="policy steps per round")
    ap.add_argument("--dagger-epochs", type=int, default=10)
    # Kickstarting toward the same expert (ParallelPPO.train)
    ap.add_argument("--kick-coef", type=float, default=0.0,
                    help="initial weight of the expert's cross-entropy term (0: off)")
    ap.add_argument("--kick-steps", type=int, default=1_000_000,
                    help="steps over which that weight anneals linearly to 0")
    args = ap.parse_args()

    out = ROOT / "results" / "teacher" / args.name
    if out.exists():
        raise SystemExit(f"{out} exists; pick another --name")
    out.mkdir(parents=True)
    reward = RewardConfig(gamma=args.gamma, k=args.k, w_action_rate=args.w_action_rate)
    ppo = dict(n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
               learning_rate=args.lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
               target_kl=args.target_kl,
               clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
               policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                  activation_fn=th.nn.Tanh, log_std_init=float(np.log(0.5))))
    meta = dict(args={k: str(v) for k, v in vars(args).items()},
                ppo={k: str(v) for k, v in ppo.items()}, reward=str(dataclasses.asdict(reward)),
                provenance=provenance(), bank_sha256=_sha256(args.bank),
                started=datetime.now().isoformat(timespec="seconds"))
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"bank sha256 {meta['bank_sha256']}", flush=True)

    if args.dagger_rounds and not args.warm_start:
        raise SystemExit("--dagger-rounds needs --warm-start (its round 0)")
    model = ParallelPPO(args.workers, dict(bank=str(args.bank), reward=reward), seed=args.seed,
                        kick_coef=args.kick_coef, kick_steps=args.kick_steps,
                        expert_labels=args.dagger_rounds > 0, **ppo)
    model.set_logger(configure(str(out), ["stdout", "csv"]))
    callback = ValidationCheckpoints(out, args.eval_every, range(args.val_seeds), args.workers)
    if args.warm_start:
        episodes = collect_expert(args.warm_start, dict(bank=str(args.bank), reward=reward),
                                  args.workers, args.seed)
        stats = warm_start(model, episodes, epochs=args.warm_epochs,
                           log_std=float(np.log(args.warm_std)), seed=args.seed)
        stats["expert_success"] = float(np.mean([e["success"] for e in episodes]))
        stats["expert_outcomes"] = {o: sum(e["outcome"] == o for e in episodes)
                                    for o in ("contact", "collision", "timeout")}
        (out / "warm_start.json").write_text(json.dumps(stats, indent=2))
        print(f"warm start: {stats}", flush=True)
        callback.init_callback(model)
        callback.evaluate()  # the warm-started policy, before any PPO
        model.logger.dump(0)
        for k in range(1, args.dagger_rounds + 1):
            stats = dagger(model, episodes, args.dagger_steps, epochs=args.dagger_epochs,
                           seed=args.seed + k)
            with open(out / "dagger.jsonl", "a") as f:
                f.write(json.dumps(dict(stats, round=k)) + "\n")
            print(f"dagger round {k}: {stats}", flush=True)
            callback.evaluate(f"dagger{k}")
            model.logger.dump(0)
    try:
        if args.steps:
            model.learn(args.steps, callback=callback)
            if callback.last != model.num_timesteps:  # the final policy
                callback.evaluate()
                model.logger.dump(model.num_timesteps)
    finally:
        model.close()
        with open(out / "episodes.jsonl", "w") as f:
            for e in model.episode_log:
                f.write(json.dumps(e) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
