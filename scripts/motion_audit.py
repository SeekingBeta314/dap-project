"""Distinguish successful contacts during the moving reach from contacts on hold.

python scripts/motion_audit.py --reconstruct --workers 8
python scripts/motion_audit.py

The first command rebuilds nominal partner trajectories and checks their geometry
and planned meeting times against the 300-seed speed experiment. The second
verifies the reconstructed covariates against all six archived experiments and
classifies successful first-contact samples on their common nominal-noise seeds.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from plot_outcomes import METHODS
from publish_study import SCENARIO_FIELDS, audit_run

DEFAULT_OUT = Path("docs/results")


def sample(seed):
    from dap.experiment import trial_rngs
    from dap.metrics import TrialResult
    from dap.scenario import SCENARIO_PRESETS, sample_scenario
    from dap.trial import record_scenario

    sc = sample_scenario(trial_rngs(seed)["scenario"], SCENARIO_PRESETS["nominal"])
    record = TrialResult(seed, "audit", 1)
    record_scenario(record, sc)
    return {"seed": seed, "t_end": sc.partner.t_end,
            **{field: getattr(record, field) for field in SCENARIO_FIELDS}}


def field_values(times):
    if not times.index.is_unique or not np.isfinite(times.t_end).all():
        raise ValueError("Invalid reconstruction seeds or stop times")
    if (times.t_end < times.t_meet_true).any():
        raise ValueError("Partner stop precedes its planned meeting")
    return {seed: row.to_numpy(dtype=float) for seed, row
            in times[list(SCENARIO_FIELDS)].iterrows()}


def reconstruct(runs, out, workers):
    meta, trials = audit_run(runs / "speed_limits", {})
    if meta["config"]["scenario"] != "nominal":
        raise ValueError("This reconstruction requires the nominal scenario preset")
    seeds = [r["seed"] for r in trials if r["error_kind"] != "scenario"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        times = pd.DataFrame(pool.map(sample, seeds)).set_index("seed")
    audit_run(runs / "speed_limits", field_values(times))
    out.mkdir(exist_ok=True, parents=True)
    times.sort_index().to_csv(out / "motion_times.csv")


def success_counts(rows, times, seeds, dt):
    passed = [r for s, r in rows.items() if s in seeds and r["success"]]
    t = np.array([r["t_event"] for r in passed], dtype=float)
    ends = np.array([times.loc[r["seed"], "t_end"] for r in passed])
    errors = np.array([r["time_err"] for r in passed], dtype=float)
    moving = t < ends
    # A first-contact sample at or after the stop may bracket the actual stop
    # with the previous 10 ms sample. Report that boundary subset explicitly.
    straddled = ~moving & (t - dt < ends)
    return {
        "paired_trials": len(seeds),
        "aligned_touches": len(passed),
        "before_stop": int(moving.sum()),
        "at_or_after_stop": int((~moving).sum()),
        "stop_in_final_sample_interval": int(straddled.sum()),
        "before_stop_and_within_100ms": int((moving & (np.abs(errors) <= .1)).sum()),
        "before_stop_and_within_200ms": int((moving & (np.abs(errors) <= .2)).sum()),
        "all_within_100ms": int((np.abs(errors) <= .1).sum()),
        "all_within_200ms": int((np.abs(errors) <= .2).sum()),
    }


def summarize(runs, out):
    times = pd.read_csv(out / "motion_times.csv").set_index("seed")
    expected = field_values(times)
    rows_by_method = {}
    for directory, name, _ in METHODS:
        cfg, trials = audit_run(runs / directory, expected)
        if cfg["config"]["scenario"] != "nominal":
            raise ValueError(f"{directory}: expected nominal scenarios")
        rows = {r["seed"]: r for r in trials if r["planner"] == name
                and r["noise_scale"] == 1 and r["error_kind"] != "scenario"}
        if not rows.keys() <= set(times.index):
            raise ValueError(f"{directory}: incomplete motion time reconstruction")
        rows_by_method[name] = (rows, cfg["config"]["dt"])
    seeds = set.intersection(*(set(rows) for rows, _ in rows_by_method.values()))
    if not seeds:
        raise ValueError("No common seeds")
    table = pd.DataFrame([{"planner": name, **success_counts(rows, times, seeds, dt)}
                          for name, (rows, dt) in rows_by_method.items()])
    table.to_csv(out / "motion_at_success.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reconstruct", action="store_true", help="resample partner trajectories")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.reconstruct:
        reconstruct(args.runs, args.out, args.workers)
    else:
        summarize(args.runs, args.out)
