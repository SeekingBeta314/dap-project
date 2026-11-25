"""Reproduce the report figure from the archived, seed-paired trials."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Directory, row label and presentation label for each evaluated method.
METHODS = (
    ("speed_limits_pixels", "pixels", "Depth + arm state"),
    ("speed_limits_proprio", "proprio", "Arm state only"),
    ("speed_limits", "replan", "Replan"),
    ("replan_vs_once", "once", "Plan once"),
    ("speed_limits_oracle", "oracle", "Oracle"),
    ("speed_limits_hold", "hold", "Hold still"),
)


def plot(root, output):
    records = []
    for dirname, method, _ in METHODS:
        with gzip.open(root / dirname / "trials.jsonl.gz", "rt") as f:
            rows = [json.loads(line) for line in f]
        records.append({r["seed"]: r for r in rows if r["planner"] == method
                        and r["noise_scale"] == 1 and r["error_kind"] != "scenario"})
    seeds = sorted(set.intersection(*(set(rows) for rows in records)))
    if not seeds:
        raise ValueError("Missing paired trials")
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    labels = ["Aligned touch", "Misaligned touch", "Collision", "Timeout", "Controller error"]
    colors = ["#176b60", "#8ac7a9", "#cc6a3f", "#cfdae0", "#292635"]
    rates = []
    for rows in records:
        rs = [rows[s] for s in seeds]
        codes = [0 if r["success"] else
                 {"contact": 1, "collision": 2, "timeout": 3, "error": 4}[r["outcome"]]
                 for r in rs]
        rates.append(100 * np.bincount(codes, minlength=5) / len(seeds))
    left = np.zeros(len(METHODS))
    for label, color, pct in zip(labels, colors, np.array(rates).T):
        if np.any(pct):
            ax.barh(np.arange(len(METHODS)), pct, left=left, label=label, color=color,
                    edgecolor="white", linewidth=.6, height=.65)
        left += pct
    ax.set_yticks(np.arange(len(METHODS)), [m[2] for m in METHODS])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"Share of {len(seeds)} matched trials (%)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.0), ncol=3)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("docs/results"))
    parser.add_argument("--out", type=Path, default=Path("docs/outcomes.png"))
    args = parser.parse_args()
    plot(args.runs, args.out)
