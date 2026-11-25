"""Regenerate the checkpoint and independent validation figure from the archive."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
GREEN = "#176d60"
GRAY = "#536575"
LIGHT = "#d6dfe2"


def main():
    selection = json.loads((HERE / "selection.json").read_text())
    runs = selection["validation"]
    rounds = [int(run["checkpoint"].replace("dagger", "")) for run in runs]
    rates = [run["successes"] / run["n"] for run in runs]
    with (HERE / "bar_summary.csv").open() as source:
        checks = {row["planner"]: row for row in csv.DictReader(source)}

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig, (train, check) = plt.subplots(
        1, 2, figsize=(8.6, 3.25), gridspec_kw={"width_ratios": (1.65, 1)}, layout="constrained"
    )
    for ax in (train, check):
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", color=LIGHT, linewidth=.7)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_axisbelow(True)

    train.plot(rounds, rates, color=GREEN, linewidth=1.8, marker="o", markersize=4)
    train.set_xticks(range(0, 13, 2))
    train.set_xlabel("DAgger round  (0 = behavior cloning)")
    train.set_ylabel("Aligned first contacts")
    train.set_title("Checkpoint selection on 32 scenarios", loc="left", fontsize=11)
    for x in (0, len(runs) - 1):
        train.annotate(
            f"{runs[x]['successes']}/{runs[x]['n']}", (rounds[x], rates[x]),
            xytext=(4, 6 if x == 0 else -15), textcoords="offset points",
            ha="left" if x == 0 else "right", color=GREEN, fontsize=9, weight="bold"
        )

    systems = [("servo_gt_cam", "Pose servo"), ("teacher", "Teacher"), ("expert", "Expert")]
    for x, (name, _) in enumerate(systems):
        row = checks[name]
        rate, lo, hi = (float(row[field]) for field in ("success", "success_lo", "success_hi"))
        color = GREEN if name == "teacher" else GRAY
        check.errorbar(x, rate, yerr=[[rate - lo], [hi - rate]], fmt="o", color=color,
                       capsize=4, markersize=6, linewidth=1.7)
        check.text(x, lo - .042, f"{rate:.1%}", ha="center", va="top", color=color, fontsize=9)
    check.set_xlim(-.5, len(systems) - .5)
    check.set_xticks(range(len(systems)), [label for _, label in systems], fontsize=9)
    check.set_title("200 independent scenarios", loc="left", fontsize=11)
    for suffix in ("pdf", "png"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(HERE / f"training.{suffix}", dpi=200, facecolor="white", metadata=metadata)
    plt.close(fig)


if __name__ == "__main__":
    main()
