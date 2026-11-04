"""Summarize and plot one experiment's trials.jsonl.

    python scripts/analyze.py results/replan_vs_once --x speed_scale --hue planner
    python scripts/analyze.py results/velocity_match --x speed_scale --hue velocity_match

Prints a per-condition summary table and writes figures to <dir>/figures/:
success rate (with 95% Wilson intervals) and the median of each error metric
against --x, one line per --hue value, faceted by --facet if given.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CONDITION_COLS = ["planner", "speed_scale", "noise_scale", "n_checkpoints", "velocity_match"]
ERROR_METRICS = {
    "pos_err": "palm position error at contact (m)",
    "time_err": "timing error at contact (s)",
    "angle_err": "palm angle error at contact (deg)",
    "vel_mismatch": "velocity mismatch before contact (m/s)",
    "min_arm_dist": "closest arm-arm distance (m)",
    "max_checkpoint_err": "worst checkpoint miss (m)",
    "plan_ms_median": "median planning time (ms)",
}


def load(results_dir):
    df = pd.read_json(Path(results_dir) / "trials.jsonl", lines=True)
    df["max_checkpoint_err"] = df["checkpoint_errs"].apply(
        lambda v: max(v) if len(v) else np.nan)
    df["plan_ms_median"] = df["plan_times_ms"].apply(
        lambda v: float(np.median(v)) if len(v) else np.nan)
    df["crashed"] = df["error"].astype(bool)
    return df


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def summary(df):
    rows = []
    for key, g in df.groupby(CONDITION_COLS):
        ok = g[~g.crashed]
        k, n = int(ok.success.sum()), len(ok)
        lo, hi = wilson(k, n)
        rows.append({
            **dict(zip(CONDITION_COLS, key)),
            "n": n, "crashed": int(g.crashed.sum()),
            "success": k / n if n else np.nan, "ci_lo": lo, "ci_hi": hi,
            "contact": ok.contact.mean(),
            **{f"med_{m}": ok[m].median() for m in ERROR_METRICS},
            "infeasible_per_trial": ok.n_infeasible.mean(),
        })
    return pd.DataFrame(rows)


def plot_success(table, x, hue, facet, out):
    facets = sorted(table[facet].unique()) if facet else [None]
    fig, axes = plt.subplots(1, len(facets), figsize=(4.5 * len(facets), 3.6),
                             sharey=True, squeeze=False)
    for ax, fv in zip(axes[0], facets):
        sub = table if fv is None else table[table[facet] == fv]
        for hv, g in sub.groupby(hue):
            g = g.groupby(x).agg(success=("success", "mean"), lo=("ci_lo", "mean"),
                                 hi=("ci_hi", "mean")).reset_index()
            ax.errorbar(g[x], g.success, yerr=[g.success - g.lo, g.hi - g.success],
                        marker="o", capsize=3, label=f"{hue}={hv}")
        ax.set_xlabel(x)
        ax.set_ylim(0, 1.02)
        if fv is not None:
            ax.set_title(f"{facet}={fv}")
    axes[0][0].set_ylabel("clasp success rate")
    axes[0][-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "success.png", dpi=150)
    plt.close(fig)


def plot_errors(df, x, hue, out):
    ok = df[~df.crashed & df.contact]
    metrics = [m for m in ERROR_METRICS if ok[m].notna().any()]
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.4), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for hv, g in ok.groupby(hue):
            med = g.groupby(x)[m].median()
            q1 = g.groupby(x)[m].quantile(0.25)
            q3 = g.groupby(x)[m].quantile(0.75)
            ax.plot(med.index, med.values, marker="o", label=f"{hue}={hv}")
            ax.fill_between(med.index, q1.values, q3.values, alpha=0.2)
        ax.set_xlabel(x)
        ax.set_title(ERROR_METRICS[m], fontsize=9)
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "errors.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--x", default="speed_scale", choices=CONDITION_COLS)
    ap.add_argument("--hue", default="planner", choices=CONDITION_COLS)
    ap.add_argument("--facet", choices=CONDITION_COLS)
    args = ap.parse_args()

    df = load(args.results_dir)
    table = summary(df)
    out = args.results_dir / "figures"
    out.mkdir(exist_ok=True)

    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    table.to_csv(args.results_dir / "summary.csv", index=False)
    if df.crashed.any():
        print(f"\n{int(df.crashed.sum())} crashed trials; first error:\n"
              f"{df[df.crashed].error.iloc[0]}")

    plot_success(table, args.x, args.hue, args.facet, out)
    plot_errors(df, args.x, args.hue, out)
    print(f"\nwrote {args.results_dir / 'summary.csv'} and {out}/")


if __name__ == "__main__":
    main()
