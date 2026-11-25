"""The teacher's bar check (docs/teacher_design.md section 2): a teacher
checkpoint, servo_gt_cam (the bar) and the ServoExpert (the teacher's own
reference) on the same validation seeds of the `teacher_bar` config
(100-299, disjoint from the 0-99 checkpoints are chosen on), then
scripts/analyze.py's tables (Wilson intervals, exact McNemar, paired loss).
A check, not a result: the teacher and the expert see ground truth.

    python scripts/check_teacher.py results/teacher/dagger_15/ckpt_dagger13 --workers 8

Writes results/teacher_bar/<run>_<checkpoint>/: config.json, trials.jsonl,
and analyze.py's summary.csv, paired.csv, by_speed.csv and figures.
"""

import argparse
import json
import sys
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dap.experiment import CONFIGS, Condition  # noqa: E402
from scripts.analyze import DEFAULT_SPEED_BINS, analyze  # noqa: E402

PLANNERS = ("teacher", "servo_gt_cam", "expert")


def _run_one(args):
    planner, seed, ckpt = args
    from dap.policy_eval import run_policy_trial

    cfg, cond = CONFIGS["teacher_bar"], Condition(planner, 1.0)
    if planner == "servo_gt_cam":
        from dap.baselines import SERVO_DEPTH_CAMERA, servo_gt_cam_factory
        from dap.trial import run_trial

        return run_trial(cfg, cond, seed, camera=SERVO_DEPTH_CAMERA,
                         controller_factory=servo_gt_cam_factory).to_dict()
    if planner == "teacher":
        from dap.teacher import load_teacher

        policy = load_teacher(ckpt)
    else:
        from dap.teacher import ServoExpert

        policy = ServoExpert()
    return run_policy_trial(cfg, cond, seed, policy, allow_privileged=True).to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path, help="path without .zip (ParallelPPO.save_checkpoint)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = CONFIGS["teacher_bar"]
    out = ROOT / "results" / "teacher_bar" / f"{args.checkpoint.parent.name}_{args.checkpoint.name}"
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(
        {"config": cfg.name, "seeds": [cfg.first_seed, cfg.first_seed + cfg.n_seeds - 1],
         "checkpoint": str(args.checkpoint), "planners": PLANNERS,
         "started": datetime.now().isoformat(timespec="seconds")}, indent=2))
    jobs = [(p, s, str(args.checkpoint)) for p in PLANNERS for s in cfg.seeds()]
    with get_context("spawn").Pool(args.workers) as pool, open(out / "trials.jsonl", "w") as f:
        for i, row in enumerate(pool.imap_unordered(_run_one, jobs), 1):
            f.write(json.dumps(row) + "\n")
            if i % 100 == 0:
                print(f"{i}/{len(jobs)} trials", flush=True)
    analyze(out, [float(v) for v in DEFAULT_SPEED_BINS.split(",")])


if __name__ == "__main__":
    main()
