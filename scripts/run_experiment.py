"""Run a named experiment sweep, one JSON line per trial.

    python scripts/run_experiment.py smoke --dry-run
    python scripts/run_experiment.py replan_vs_once --workers 4
    python scripts/run_experiment.py speed_limits --seeds 5      # quick look

Writes results/<name>/config.json and results/<name>/trials.jsonl. Re-running
skips trials already in trials.jsonl, so an interrupted sweep resumes.
A trial that raises is recorded with its `error` and the sweep moves on
(use --fail-fast while debugging).
"""

import argparse
import dataclasses
import json
import subprocess
import sys
import traceback
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dap.experiment import CONFIGS, Condition  # noqa: E402

_scene = None


def _init_worker():
    global _scene
    from dap.trial import Scene

    _scene = Scene()


def _run_one(args):
    cfg, cond, seed, fail_fast = args
    from dap.metrics import TrialResult
    from dap.trial import run_trial

    try:
        return run_trial(cfg, cond, seed, _scene).to_dict()
    except Exception:
        if fail_fast:
            raise
        r = TrialResult(seed=seed, **dataclasses.asdict(cond))
        r.error = traceback.format_exc(limit=5)
        return r.to_dict()


def _trial_key(cond_dict, seed):
    return (json.dumps(cond_dict, sort_keys=True), seed)


def _done_keys(path):
    done = set()
    if path.exists():
        cond_fields = [f.name for f in dataclasses.fields(Condition)]
        for line in path.read_text().splitlines():
            row = json.loads(line)
            done.add(_trial_key({k: row[k] for k in cond_fields}, row["seed"]))
    return done


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=sorted(CONFIGS))
    ap.add_argument("--seeds", type=int, help="override n_seeds")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="list trials, run nothing")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore existing trials.jsonl")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    if args.seeds is not None:
        cfg = dataclasses.replace(cfg, n_seeds=args.seeds)

    out_dir = args.out / cfg.name
    trials_path = out_dir / "trials.jsonl"
    done = set() if args.fresh else _done_keys(trials_path)
    todo = [(c, s) for c, s in cfg.trials() if _trial_key(c.key(), s) not in done]

    print(f"{cfg.name}: {cfg.n_trials()} trials "
          f"({sum(1 for _ in cfg.conditions())} conditions x {cfg.n_seeds} seeds), "
          f"{len(done)} already done, {len(todo)} to run")
    if args.dry_run:
        for cond in cfg.conditions():
            print("  ", cond)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": dataclasses.asdict(cfg),
        "git": _git_commit(),
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))

    mode = "w" if args.fresh else "a"
    jobs = [(cfg, c, s, args.fail_fast) for c, s in todo]
    with open(trials_path, mode) as f:
        if args.workers > 1:
            with Pool(args.workers, initializer=_init_worker) as pool:
                rows = pool.imap_unordered(_run_one, jobs)
                _write_rows(rows, f, len(jobs))
        else:
            _init_worker()
            _write_rows(map(_run_one, jobs), f, len(jobs))
    print(f"wrote {trials_path}")


def _write_rows(rows, f, total):
    n_ok = n_err = 0
    for i, row in enumerate(rows, 1):
        f.write(json.dumps(row) + "\n")
        f.flush()
        n_err += bool(row["error"])
        n_ok += bool(row["success"])
        if i % 10 == 0 or i == total:
            print(f"  {i}/{total}  success={n_ok}  errors={n_err}", flush=True)


if __name__ == "__main__":
    main()
