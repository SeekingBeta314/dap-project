"""Run a named experiment sweep, one JSON line per trial.

    python scripts/run_experiment.py smoke --dry-run
    python scripts/run_experiment.py smoke                      # the config's own controllers
    python scripts/run_experiment.py first_look --controller servo_depth --workers 4
    python scripts/run_experiment.py first_look --policy mypkg.policies:make --policy-name student

Writes results/<name>/config.json and results/<name>/trials.jsonl. Re-running
skips trials already completed in trials.jsonl, so an interrupted sweep
resumes (crashed trials are retried; the analysis keeps the last row per
trial). A trial that raises is recorded with its `error` and `error_kind`
and the sweep moves on (use --fail-fast while debugging).
"""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import inspect
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

BASELINES = ("hold", "blind", "servo", "servo_depth", "servo_gt", "servo_gt_cam",
             "oracle")  # dap/baselines.py


def _factory(name):
    from dap import baselines

    if name not in BASELINES:
        raise ValueError(f"unknown controller {name!r}; expected one of {BASELINES}")
    return getattr(baselines, f"{name}_factory")


def _load_policy_factory(spec):
    """"module:function" -> function(cfg, cond) returning a policy
    (dap.policy_eval)."""
    import importlib

    module, _, name = spec.partition(":")
    if not name:
        raise SystemExit(f"--policy must look like module:function, got {spec!r}")
    return getattr(importlib.import_module(module), name)


def _run_one(args):
    # Each trial samples its own scenario, so it builds its own scene.
    cfg, cond, seed, controller, policy_spec, fail_fast = args
    from dap.metrics import TrialResult
    from dap.trial import run_trial

    try:
        if policy_spec is not None:
            from dap.policy_eval import run_policy_trial

            policy = _load_policy_factory(policy_spec)(cfg, cond)
            return run_policy_trial(cfg, cond, seed, policy).to_dict()
        from dap.baselines import CAMERA_BASELINES, SERVO_DEPTH_CAMERA

        controller = controller or cond.planner
        camera = SERVO_DEPTH_CAMERA if controller in CAMERA_BASELINES else None
        return run_trial(cfg, cond, seed, camera=camera,
                         controller_factory=_factory(controller)).to_dict()
    except Exception as e:
        if fail_fast:
            raise
        from dap.scenario import SCENARIO_PRESETS, ScenarioError, draw_peak_speed

        # Controller crashes are caught inside run_trial (a trial outcome);
        # anything reaching here is a sampling failure or a harness bug.
        r = TrialResult(seed=seed, **dataclasses.asdict(cond))
        r.outcome = "error"
        r.error_kind = "scenario" if isinstance(e, ScenarioError) else "harness"
        # The drawn partner speed is the scenario stream's first draw.
        r.target_peak_speed = draw_peak_speed(cfg.rngs(seed)["scenario"],
                                              SCENARIO_PRESETS[cfg.scenario])
        r.error = traceback.format_exc(limit=5)
        return r.to_dict()


def _trial_key(cond_dict, seed):
    return (json.dumps(cond_dict, sort_keys=True), seed)


def _done_keys(path):
    """Trials already completed. Crashed trials are not "done": a resume
    retries them, and the analysis keeps the last row per trial."""
    done = set()
    if path.exists():
        cond_fields = [f.name for f in dataclasses.fields(Condition)]
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = _trial_key({k: row[k] for k in cond_fields}, row["seed"])
            if row.get("error"):
                done.discard(key)
            else:
                done.add(key)
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


def _runtime_info(controller, policy):
    """Identify imported project code and bundled inference weights for this sweep."""
    if policy:
        from dap import policy_eval  # noqa: F401; load evaluator dependencies
        factory = _load_policy_factory(policy)
    else:
        from dap import baselines, trial  # load the actual controller and harness dependencies
        factory = baselines if controller is None else _factory(controller)
    paths = {"scripts/run_experiment.py": Path(__file__),
             "models/dap_hand.sdf": ROOT / "models/dap_hand.sdf"}
    for name, module in sys.modules.copy().items():
        if name == "dap" or name.startswith("dap."):
            src = getattr(module, "__file__", None)
            if src:
                path = Path(src).resolve()
                paths[str(path.relative_to(ROOT))] = path
    if policy:
        src = inspect.getsourcefile(factory)
        if src:
            paths["policy_factory"] = Path(src)
        bundled = {"dap.pixel_policy:pixel_factory": "depth.npz",
                   "dap.pixel_policy:proprio_factory": "proprio.npz"}
        if policy in bundled:
            key = f"models/policies/{bundled[policy]}"
            paths[key] = ROOT / key
    return {
        "python": sys.version.split()[0],
        "packages": {name: importlib.metadata.version(name)
                     for name in ("drake", "numpy", "gymnasium")},
        "sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest()
                   for k, p in sorted(paths.items())},
    }


def _resume_metadata(previous, current):
    """Extending the seed range preserves provenance of surviving trials."""
    old_cfg, new_cfg = dict(previous["config"]), dict(current["config"])
    old_n, new_n = old_cfg.pop("n_seeds"), new_cfg.pop("n_seeds")
    compatible = (new_n >= old_n and old_cfg == new_cfg
                  and all(previous.get(k) == current.get(k)
                          for k in ("runtime", "controller", "policy")))
    if not compatible:
        raise SystemExit("Existing results have a different or unverified run configuration. "
                         "Use --out for a separate run or --fresh to replace the trials.")
    return {**previous, "config": current["config"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=sorted(CONFIGS))
    ap.add_argument("--seeds", type=int, help="override n_seeds")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true", help="list trials, run nothing")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore existing trials.jsonl")
    ap.add_argument("--controller", choices=BASELINES,
                    help="a baseline or reference (dap/baselines.py) replaces the config's "
                         "controllers; results go to <name>_<controller>")
    ap.add_argument("--policy", metavar="MODULE:FUNCTION",
                    help="evaluate a learned policy (dap/policy_eval.py) instead of a "
                         "controller; FUNCTION(cfg, cond) returns the policy")
    ap.add_argument("--policy-name", default="policy",
                    help="label for the policy's rows; results go to <name>_<policy-name>")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    if args.seeds is not None:
        cfg = dataclasses.replace(cfg, n_seeds=args.seeds)
    if args.policy is not None:
        if args.controller is not None:
            raise SystemExit("--policy and --controller are exclusive")
        cfg = dataclasses.replace(cfg, name=f"{cfg.name}_{args.policy_name}",
                                  planners=(args.policy_name,))
        # Fail before the sweep starts, not once per trial.
        policy = _load_policy_factory(args.policy)(cfg, next(cfg.conditions()))
        if policy.obs_mode == "privileged":
            raise SystemExit("a privileged policy sees ground truth: evaluate it as a reference "
                             "with dap.policy_eval.run_policy_trial(..., allow_privileged=True)")
    elif args.controller is not None:
        cfg = dataclasses.replace(cfg, name=f"{cfg.name}_{args.controller}",
                                  planners=(args.controller,))

    if args.policy is None:
        for cond in cfg.conditions():
            _factory(args.controller or cond.planner)  # fail before the sweep starts

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

    if not todo and done:
        print(f"Previously completed results: {trials_path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": dataclasses.asdict(cfg),
        "controller": args.controller,
        "policy": args.policy,
        "git": _git_commit(),
        "runtime": _runtime_info(args.controller, args.policy),
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    config_path = out_dir / "config.json"
    if done:
        if not config_path.exists():
            raise SystemExit("Missing run metadata. Use --out or --fresh.")
        meta = _resume_metadata(json.loads(config_path.read_text()), json.loads(json.dumps(meta)))
    config_path.write_text(json.dumps(meta, indent=2))

    mode = "w" if args.fresh else "a"
    jobs = [(cfg, c, s, args.controller, args.policy, args.fail_fast) for c, s in todo]
    with open(trials_path, mode) as f:
        if args.workers > 1:
            with Pool(args.workers) as pool:
                rows = pool.imap_unordered(_run_one, jobs)
                _write_rows(rows, f, len(jobs))
        else:
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
