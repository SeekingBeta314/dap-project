"""Summarize one experiment's trials.jsonl (paired analysis).

    python scripts/analyze.py results/replan_vs_once
    python scripts/analyze.py results/speed_limits --speed-bins 0.15,0.3,0.45,0.6,0.75,0.9

Writes to <dir>/:
  summary.csv    per condition: trials, crash counts by kind, success /
                 contact / collision / timeout rates with 95% Wilson
                 intervals, mean loss with a bootstrap CI, medians of the
                 contact scores
  paired.csv     every pair of planners at each noise level, on the same
                 seeds: exact McNemar on success, paired bootstrap of the
                 loss and of the contact scores, Holm-adjusted p-values
  by_speed.csv   success and mean loss per realized partner peak-speed bin,
                 plus crashes per bin of the drawn speed
  figures/       success and loss vs. partner speed

Crash policy (docs/setup.md section 9). A crashed trial is never silently
dropped:
  - "controller" crashes are failures of the method: they count as trials
    with success = False and the collision-tier loss;
  - "scenario" crashes (no scenario could be sampled for the seed) are the
    same for every condition; those seeds are excluded from every condition
    and counted;
  - "harness" crashes are bugs in our code: excluded, counted, and printed
    first, because results with harness crashes should not be trusted.
If a trial appears more than once (a resumed sweep retried it), the last
row counts.
"""

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dap.scoring import LossWeights, dap_loss  # noqa: E402
from dap.stats import (  # noqa: E402
    bin_index,
    binned_rate,
    bootstrap_mean_ci,
    covering_edges,
    holm,
    mcnemar_exact,
    paired_bootstrap,
    wilson,
)

CONDITION_COLS = ["planner", "noise_scale"]
CONTACT_METRICS = ["pos_err", "rot_err", "angle_err", "finger_err", "time_err", "rel_speed"]
PAIRED_CONTACT_METRICS = ["rot_err", "pos_err", "time_err"]
DEFAULT_SPEED_BINS = "0.15,0.3,0.45,0.6,0.75,0.9"


def load(results_dir, weights=LossWeights()):
    """Read trials.jsonl and apply the crash policy. Returns (df, crashes):
    `df` has one row per counted (condition, seed), with columns
    `controller_crash` and `loss`; `crashes` lists every crashed row."""
    path = Path(results_dir) / "trials.jsonl"
    if not path.exists():
        path = path.with_suffix(".jsonl.gz")
    df = pd.read_json(path, lines=True)
    df = df.drop_duplicates(CONDITION_COLS + ["seed"], keep="last").reset_index(drop=True)
    if "error_kind" not in df:
        df["error_kind"] = ""
    df["error_kind"] = df["error_kind"].fillna("")
    crashed = (df["outcome"] == "error") | df["error"].fillna("").astype(bool)
    df.loc[crashed & (df.error_kind == ""), "error_kind"] = "harness"
    crashes = df[crashed].copy()

    # Scenario crashes: drop the seed everywhere. Harness crashes: drop the row.
    bad_seeds = set(crashes.loc[crashes.error_kind == "scenario", "seed"])
    keep = ~df.seed.isin(bad_seeds) & ~(crashed & (df.error_kind == "harness"))
    df = df[keep].sort_values(CONDITION_COLS + ["seed"]).copy()
    df["controller_crash"] = df.error_kind == "controller"
    df["success"] = df.success.fillna(False).astype(bool) & ~df.controller_crash
    df["loss"] = dap_loss(df, weights) if len(df) else []
    df.loc[df.controller_crash, "loss"] = weights.collision_penalty
    return df, crashes


def _cond(frame, key):
    return frame[(frame.planner == key[0]) & (frame.noise_scale == key[1])]


def summary(df, crashes):
    keys = set(map(tuple, df[CONDITION_COLS].drop_duplicates().itertuples(index=False)))
    keys |= set(map(tuple, crashes[CONDITION_COLS].drop_duplicates().itertuples(index=False)))
    rows = []
    for key in sorted(keys):
        g, c = _cond(df, key), _cond(crashes, key)
        n = len(g)
        row = {**dict(zip(CONDITION_COLS, key)), "n": n,
               "controller_crashes": int(g.controller_crash.sum()),
               "scenario_crashes": int((c.error_kind == "scenario").sum()),
               "harness_crashes": int((c.error_kind == "harness").sum())}
        for name, col in [("success", g.success),
                          ("contact", g.outcome == "contact"),
                          ("collision", g.outcome == "collision"),
                          ("timeout", g.outcome == "timeout")]:
            k = int(col.sum())
            lo, hi = wilson(k, n)
            row.update({name: k / n if n else np.nan, f"{name}_lo": lo, f"{name}_hi": hi})
        m, lo, hi = bootstrap_mean_ci(g.loss)
        row.update({"loss": m, "loss_lo": lo, "loss_hi": hi})
        contact = g[g.outcome == "contact"]
        row.update({f"med_{c}": contact[c].median() for c in CONTACT_METRICS})
        row["reachable_frac"] = g.reachable_in_time.mean() if n else np.nan
        row["step_ms_p95"] = g.step_ms_p95.median() if n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired(df):
    rows = []
    planners = sorted(df.planner.unique())
    for noise, g in df.groupby("noise_scale"):
        for a, b in itertools.combinations(planners, 2):
            A = g[g.planner == a].set_index("seed")
            B = g[g.planner == b].set_index("seed")
            seeds = A.index.intersection(B.index)
            A, B = A.loc[seeds], B.loc[seeds]
            mc = mcnemar_exact(A.success.values, B.success.values)
            bl = paired_bootstrap(A.loss.values, B.loss.values)
            row = {
                "noise_scale": noise, "a": a, "b": b, "n_pairs": len(seeds),
                "success_a": A.success.mean(), "success_b": B.success.mean(),
                "a_only": mc["a_only"], "b_only": mc["b_only"], "p_mcnemar": mc["p"],
                "loss_diff": bl["mean_diff"], "loss_diff_lo": bl["lo"],
                "loss_diff_hi": bl["hi"], "p_loss": bl["p_boot"],
            }
            # Contact scores exist only where both made contact.
            both = (A.outcome == "contact").values & (B.outcome == "contact").values
            row["n_both_contact"] = int(both.sum())
            for m in PAIRED_CONTACT_METRICS:
                x, y = A[m].to_numpy(float), B[m].to_numpy(float)
                if m == "time_err":
                    x, y = np.abs(x), np.abs(y)
                r = paired_bootstrap(np.where(both, x, np.nan), np.where(both, y, np.nan))
                row.update({f"{m}_diff": r["mean_diff"], f"{m}_lo": r["lo"],
                            f"{m}_hi": r["hi"], f"p_{m}": r["p_boot"]})
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        p_cols = ["p_mcnemar", "p_loss"] + [f"p_{m}" for m in PAIRED_CONTACT_METRICS]
        # Holm over every test in this analysis (NaN = not computable).
        adj = holm(out[p_cols].to_numpy(float).ravel()).reshape(len(out), len(p_cols))
        for j, c in enumerate(p_cols):
            out[f"{c}_holm"] = adj[:, j]
    return out


def by_speed(df, crashes, edges):
    """Success / loss per bin of realized partner peak speed (outer edges
    widened so no trial falls outside), and crashes per bin of the drawn
    speed (a crashed trial may have no realized one)."""
    rows = []
    if len(df):
        # Realized speed; a trial without one (shouldn't happen) uses the drawn one.
        speed = df.partner_peak_speed.fillna(df.target_peak_speed)
        e = covering_edges(edges, speed)
        for key, g in df.assign(speed=speed).groupby(CONDITION_COLS):
            idx = bin_index(g.speed, e)
            if (idx < 0).any():
                raise ValueError("trials without any partner speed")
            for i, r in enumerate(binned_rate(g.speed, g.success, e)):
                rows.append({**dict(zip(CONDITION_COLS, key)), "kind": "trials", **r,
                             "mean_loss": g.loss[idx == i].mean(),
                             "mean_vmax_frac": g.vmax_frac[idx == i].mean(),
                             "reachable_frac": g.reachable_in_time[idx == i].mean()})
    if len(crashes):
        e = covering_edges(edges, crashes.target_peak_speed)
        for key, g in crashes.groupby(CONDITION_COLS + ["error_kind"]):
            idx = bin_index(g.target_peak_speed, e)
            for i, (lo, hi) in enumerate(zip(e[:-1], e[1:])):
                rows.append({**dict(zip(CONDITION_COLS, key[:2])),
                             "kind": f"{key[2]}_crashes (drawn speed)",
                             "bin_lo": lo, "bin_hi": hi, "n": int((idx == i).sum())})
    return pd.DataFrame(rows)


def plot_by_speed(table, out):
    t = table[table.kind == "trials"] if len(table) else table
    if not len(t):
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for key, g in t[t.n > 0].groupby(CONDITION_COLS):
        x = (g.bin_lo + g.bin_hi) / 2
        label = ", ".join(f"{c}={v}" for c, v in zip(CONDITION_COLS, key))
        yerr = np.clip([g.rate - g.ci_lo, g.ci_hi - g.rate], 0, None)
        axes[0].errorbar(x, g.rate, yerr=yerr, marker="o", capsize=3, label=label)
        axes[1].plot(x, g.mean_loss, marker="o", label=label)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("dap success rate (95% Wilson)")
    axes[1].set_ylabel("mean loss")
    for ax in axes:
        ax.set_xlabel("partner peak palm speed (m/s)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "by_speed.png", dpi=150)
    plt.close(fig)


def analyze(results_dir, edges, verbose=True):
    """Run the whole analysis; returns (summary, paired or None, by_speed)."""
    def say(*a):
        if verbose:
            print(*a)

    df, crashes = load(results_dir)
    out = Path(results_dir) / "figures"
    out.mkdir(exist_ok=True)

    harness = crashes[crashes.error_kind == "harness"]
    if len(harness):
        say(f"WARNING: {len(harness)} harness crashes (bugs; results not trustworthy). "
            f"First:\n{harness.error.iloc[0]}\n")
    if len(crashes):
        say("crashes by kind:", crashes.error_kind.value_counts().to_dict())

    table = summary(df, crashes)
    say(table.round(3).to_string(index=False))
    table.to_csv(Path(results_dir) / "summary.csv", index=False)

    pt = None
    if len(df) and df.planner.nunique() > 1:
        pt = paired(df)
        say("\npaired comparisons (same seeds):")
        say(pt.round(4).to_string(index=False))
        pt.to_csv(Path(results_dir) / "paired.csv", index=False)

    speed = by_speed(df, crashes, edges)
    if len(speed):
        say("\nby partner peak speed:")
        say(speed.round(3).to_string(index=False))
        speed.to_csv(Path(results_dir) / "by_speed.csv", index=False)
        plot_by_speed(speed, out)
    say(f"\nwrote results to {results_dir}")
    return table, pt, speed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--speed-bins", default=DEFAULT_SPEED_BINS,
                    help="comma-separated bin edges for partner peak speed (m/s); the "
                         "outer edges are widened to include every trial")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    analyze(args.results_dir, [float(v) for v in args.speed_bins.split(",")])


if __name__ == "__main__":
    main()
