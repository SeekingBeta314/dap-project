"""Summarize descriptive replay measurements giving each held-out trial one vote."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def summarize(directory):
    data = pd.concat([pd.read_csv(directory / f"{preset}.csv") for preset in ("nominal", "wide")])
    if data.duplicated(["preset", "seed"]).any():
        raise ValueError("expected one replay per scenario")
    data["position_cm"] = 100 * data.last_200ms_median_position_m
    data["orientation_deg"] = data.last_200ms_median_orientation_deg
    table = (data.groupby(["preset", "success"])
             .agg(trials=("seed", "count"), with_estimates=("position_cm", "count"),
                  terminal_median_cm=("position_cm", "median"),
                  terminal_median_deg=("orientation_deg", "median"),
                  success_before_stop=("success_before_stop", "sum")))
    table.to_csv(directory / "pose_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.1), layout="constrained")
    for ax, metric, xlabel in zip(axes, ("position_cm", "orientation_deg"),
                                  ("Palm-center error (cm)", "Palm orientation error (°)")):
        ticks, labels = [], []
        for y, (preset, success) in enumerate((("nominal", True), ("nominal", False),
                                               ("wide", True), ("wide", False))):
            values = data.loc[(data.preset == preset) & (data.success == success), metric].dropna()
            ticks.append(y)
            labels.append(f'{preset.title()}\n{"aligned" if success else "failed"}')
            if values.empty:
                continue
            c = "#226c71" if success else "#ac4e32"
            lo, median, hi = np.percentile(values, [10, 50, 90])
            ax.plot((lo, hi), (y, y), color=c, lw=2.3)
            ax.scatter(median, y, color=c, s=27, zorder=3)
        ax.set(xlabel=xlabel, yticks=ticks, yticklabels=labels if ax is axes[0] else [],
               ylim=(3.5, -.5), xlim=(0, None))
        ax.grid(axis="x", color="#ddd", lw=.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    for fmt in ("png", "pdf"):
        fig.savefig(directory / f"pose_errors.{fmt}", dpi=230)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path, nargs="?", default=ROOT / "docs/perception_results/diagnostics")
    summarize(p.parse_args().directory)
