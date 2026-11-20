"""Audit complete study runs and publish the raw evidence and paired tables.

python scripts/publish_study.py --runs results --out docs/results \
    --policies speed_limits_pixels speed_limits_proprio

Can rebuild the tables from the archived study by setting --runs docs/results.
"""

import argparse
import gzip
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import DEFAULT_SPEED_BINS, analyze, load, paired, summary
from dap.stats import wilson

SCENARIO_FIELDS = (
    "target_peak_speed", "partner_peak_speed", "partner_speed_at_meet",
    "t_meet_true", "reachable_in_time", "partner_joint_speed_ratio",
    "layout_distance", "layout_lateral", "layout_yaw_deg", "layout_height",
    "scenario_attempts",
)


def audit_run(source, scenario_values):
    meta = json.loads((source / "config.json").read_text())
    cfg = meta["config"]
    first = cfg["first_seed"]
    expected = set(itertools.product(cfg["planners"], cfg["noise_scales"],
                                     range(first, first + cfg["n_seeds"])))
    path = source / "trials.jsonl"
    if path.exists():
        lines = path.read_text().splitlines()
    else:
        with gzip.open(source / "trials.jsonl.gz", "rt") as handle:
            lines = handle.read().splitlines()
    rows = {}
    for line in lines:
        r = json.loads(line)
        rows[r["planner"], r["noise_scale"], r["seed"]] = r
    if expected != rows.keys():
        raise ValueError(f"{source}: missing or unexpected seed/condition rows")
    for key, row in rows.items():
        # A harness failure cannot substantiate a published comparison. Retain
        # controller failures as trial outcomes, as specified in docs/setup.md.
        if row["error_kind"] not in {"", "controller", "scenario"}:
            raise ValueError(f"{source} {key}: {row['error_kind']} failure")
        # Sampling failures are also seed-specific and must agree across runs.
        values = None if row["error_kind"] == "scenario" else np.asarray(
            [row[f] for f in SCENARIO_FIELDS], float)
        previous = scenario_values.setdefault(row["seed"], values)
        if values is None and previous is None:
            continue
        if (values is None or previous is None or not np.isfinite(values).all() or
                not np.allclose(previous, values, rtol=1e-7, atol=1e-7)):
            raise ValueError(f"{source} {key}: scenario differs between methods")
    return meta, [rows[k] for k in sorted(rows)]


def timing_sensitivity(data):
    """Aligned touches additionally restricted to an absolute timing window."""
    rows = []
    for planner, trials in data.groupby("planner"):
        for window in (0.1, 0.2, None):
            passed = trials.success.copy()
            if window is not None:
                passed &= trials.time_err.abs() <= window
            k, n = int(passed.sum()), len(trials)
            lo, hi = wilson(k, n)
            rows.append(dict(planner=planner, max_abs_time_err_s=window, k=k, n=n,
                             rate=k / n, ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def publish(runs, out, policies):
    names = ["replan_vs_once", "speed_limits", "speed_limits_oracle", "speed_limits_hold",
             *policies]
    scenario_values, nominal, configs = {}, [], []
    edges = [float(v) for v in DEFAULT_SPEED_BINS.split(",")]
    for name in names:
        meta, rows = audit_run(runs / name, scenario_values)
        cfg = meta["config"]
        settings = {k: cfg[k] for k in ("scenario", "dt", "t_extra", "tolerances")}
        if configs and configs[0] != settings:
            raise ValueError(f"{name}: incompatible scenario, timing, or tolerances")
        configs.append(settings)
        dest = out / name
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
        raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
        (dest / "trials.jsonl.gz").write_bytes(gzip.compress(raw, mtime=0))
        analyze(dest, edges, verbose=False)
        data, _ = load(dest)
        if name == "replan_vs_once":
            data = data[(data.noise_scale == 1) & (data.planner == "once")]
        if not len(data) or data.noise_scale.nunique() != 1 or data.noise_scale.iloc[0] != 1:
            raise ValueError(f"{name}: expected one nominal-noise method")
        nominal.append(data)
    # Each method receives the same number of identical trials in the joint
    # comparison even when its original sweep used a longer range of seeds.
    common = set.intersection(*(set(df.seed) for df in nominal))
    data = pd.concat([df[df.seed.isin(common)] for df in nominal], ignore_index=True)
    if not common or data.duplicated(["planner", "seed"]).any():
        raise ValueError("methods must have unique names and overlapping seed sets")
    crashes = data[data.controller_crash]
    summary(data, crashes).to_csv(out / "matched_summary.csv", index=False)
    paired(data).to_csv(out / "matched_paired.csv", index=False)
    timing_sensitivity(data).to_csv(out / "matched_timing.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("docs/results"))
    parser.add_argument("--policies", nargs="+", required=True,
                        help="sweep directory names for trained policies")
    args = parser.parse_args()
    publish(args.runs, args.out, args.policies)
