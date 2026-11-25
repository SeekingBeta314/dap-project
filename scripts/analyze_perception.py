"""Audit and publish the two-preset study, or rebuild it from the published raw archive.

python scripts/analyze_perception.py results/perception_study --out docs/perception_results
python scripts/analyze_perception.py docs/perception_results --out results/rebuilt_perception
"""

import argparse
import dataclasses
import gzip
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dap.eval_scenarios import DEFAULT_ARCHIVE, load_frozen_scenario  # noqa: E402
from dap.experiment import CONFIGS, PERCEPTION_METHODS  # noqa: E402
from dap.stats import holm, mcnemar_exact, paired_bootstrap, wilson  # noqa: E402
from analyze import DEFAULT_SPEED_BINS, by_speed, load, summary  # noqa: E402
from publish_study import audit_run, timing_sensitivity  # noqa: E402
from run_perception_study import CORE_FILES, digest  # noqa: E402

PRESETS = ("nominal", "wide")
LABEL = dict(student="Depth student", servo_depth="Classical depth", teacher="Privileged teacher",
             servo_gt_cam="True pose at camera rate", servo_gt="Current true pose",
             servo="Noisy pose sensor", blind="Blind prior", oracle="Meeting-pose oracle")
OUTCOMES = {"aligned": "Aligned touch", "misaligned": "Misaligned touch", "collision": "Collision",
            "near_miss": "Timeout ≤5 cm", "far_timeout": "Timeout >5 cm", "crash": "Controller error"}


def student_sources(path, manifest, seeds):
    receipt = path / "deployment" / "sources.json"
    identity = json.loads(receipt.read_text())
    if identity["weights"] != manifest["checkpoints"]["weights"] or (
            identity["sources"]["dap/student.py"] != manifest["sources"]["dap/student.py"]):
        raise ValueError("student source or checkpoint disagrees with its deployment receipt")
    covered = set()
    for shard in receipt.parent.glob("*-*.json"):
        report = json.loads(shard.read_text())
        if report["source_receipt"] != digest(receipt):
            raise ValueError("student shard has a different deployment receipt")
        covered.update(range(report["first"], report["last"] + 1))
    if covered != set(seeds):
        raise ValueError("incomplete student source attestation")
    return identity["sources"]


def read_raw(runs, preset):
    cfg = CONFIGS[f"perception_{preset}"]
    seed_names = {f"{s}.json" for s in cfg.seeds()}
    if (runs / "eval").is_dir():
        manifests, records = {}, []
        for method in PERCEPTION_METHODS:
            path = runs / "eval" / preset / method
            if not path.is_dir():
                raise ValueError(f"missing run for {preset} {method}")
            found = {p.name for p in path.glob("*.json")} - {"manifest.json"}
            if found != seed_names:
                raise ValueError(f"{preset} {method}: missing or unexpected seed files")
            manifests[method] = json.loads((path / "manifest.json").read_text())
            if method == "student":
                extra = student_sources(path, manifests[method], cfg.seeds())
                manifests[method]["sources"].update(extra)
            for seed in cfg.seeds():
                row = json.loads((path / f"{seed}.json").read_text())
                if (row["seed"], row["planner"]) != (seed, method):
                    raise ValueError(f"{path}: seed or method label differs from path")
                records.append(row)
    else:
        path = runs / preset
        manifests = json.loads((path / "config.json").read_text())["methods"]
        with gzip.open(path / "trials.jsonl.gz", "rt") as f:
            records = [json.loads(line) for line in f]
    if set(manifests) != set(PERCEPTION_METHODS) or len(records) != cfg.n_trials():
        raise ValueError(f"{preset}: expected {cfg.n_trials()} records and eight manifests")
    expected = json.loads(json.dumps(dataclasses.asdict(cfg)))
    for m, meta in manifests.items():
        if meta["method"] != m or meta["config"] != expected:
            raise ValueError(f"{preset} {m}: manifest disagrees with frozen design")
    archive = runs / "scenarios" if (runs / "scenarios").is_dir() else DEFAULT_ARCHIVE
    archive_hashes = {meta["scenario_archive"] for meta in manifests.values()}
    if archive_hashes != {digest(archive / f"{preset}.jsonl.gz")}:
        raise ValueError(f"{preset}: scenario archive checksum differs from the run")
    for row in records:
        frozen = load_frozen_scenario(cfg, row["seed"], archive)
        if row.get("scenario_sha256") != frozen.sha256:
            raise ValueError(f"{preset} seed {row['seed']}: scenario hash mismatch")
        if (row["outcome"] == "error") != bool(row["error_kind"]) or (
                row["error_kind"] == "" and bool(row["error"])):
            raise ValueError(f"unclassified error in {preset}, seed {row['seed']}")
        if row["outcome"] not in {"error", "contact", "timeout", "collision"}:
            raise ValueError(f"unknown outcome {row['outcome']}")
    records.sort(key=lambda r: (r["planner"], r["seed"]))
    encoded = "".join(json.dumps(r, allow_nan=False) + "\n" for r in records).encode()
    return {"config": expected, "methods": manifests}, gzip.compress(encoded, mtime=0)


def verify_freeze(all_meta):
    same_sources, weights = None, {}
    for meta in all_meta:
        for method, m in meta["methods"].items():
            common = ({f: m["sources"][f] for f in CORE_FILES}, m["camera"], m["depth_noise"],
                      {k: m["versions"][k] for k in ("drake", "numpy", "scipy", "gymnasium")})
            if same_sources is not None and same_sources != common:
                raise ValueError("simulation sources, runtime, camera or depth noise differ")
            same_sources = common
            identity = (m["sources"], m["checkpoints"], m["versions"])
            if weights.setdefault(method, identity) != identity:
                raise ValueError(f"{method}: implementation or checkpoint changed between presets")


def compare(data, a, b):
    A = data[data.planner == a].set_index("seed").sort_index()
    B = data[data.planner == b].set_index("seed").sort_index()
    if A.empty or not A.index.equals(B.index):
        raise ValueError(f"{a}/{b}: cannot pair all counted seeds")
    mc = mcnemar_exact(A.success, B.success)
    result = dict(a=a, b=b, n_pairs=len(A), successes_a=int(A.success.sum()),
                  successes_b=int(B.success.sum()), a_only=mc["a_only"], b_only=mc["b_only"],
                  both=int((A.success & B.success).sum()), neither=int((~A.success & ~B.success).sum()),
                  success_diff=mc["diff"], p_mcnemar=mc["p"])
    for col in ("success", "loss"):
        ci = paired_bootstrap(A[col], B[col])
        result.update({f"{col}_diff": ci["mean_diff"], f"{col}_diff_lo": ci["lo"],
                       f"{col}_diff_hi": ci["hi"]})
    return result


def outcomes(data):
    rows = []
    for method, g in data.groupby("planner"):
        timeout = g.outcome == "timeout"
        if g.loc[timeout, "min_palm_dist"].isna().any():
            raise ValueError("timeout without a measured closest approach")
        counts = dict(aligned=int(g.success.sum()),
                      misaligned=int(((g.outcome == "contact") & ~g.success).sum()),
                      collision=int((g.outcome == "collision").sum()),
                      near_miss=int((timeout & (g.min_palm_dist <= .05)).sum()),
                      far_timeout=int((timeout & (g.min_palm_dist > .05)).sum()),
                      crash=int(g.controller_crash.sum()))
        if sum(counts.values()) != len(g):
            raise ValueError(f"{method}: outcomes do not account for all trials")
        rows.append(dict(planner=method, n=len(g), **counts))
    return pd.DataFrame(rows)


def describe_trajectories(data, cfg, archive):
    """Exploratory target switches and timing of aligned contacts relative to the final knot."""
    switched, stop_times = {}, {}
    for seed in cfg.seeds():
        spec = load_frozen_scenario(cfg, seed, archive).spec
        if spec is not None:
            switched[seed] = spec["reference"]["switch"] is not None
            stop_times[seed] = float(spec["knot_times"][-1])
    primary = data[data.planner.isin(("student", "servo_depth"))]
    rows = []
    for (method, flag), g in primary.groupby(["planner", primary.seed.map(switched)]):
        n, k = len(g), int(g.success.sum())
        lo, hi = wilson(k, n)
        rows.append(dict(planner=method, target_switch=flag, n=n, k=k,
                         success=k/n, success_lo=lo, success_hi=hi))
    motion = []
    for method, g in data.groupby("planner"):
        hits = g[g.success]
        stop = hits.seed.map(stop_times)
        if hits.t_event.isna().any() or stop.isna().any():
            raise ValueError("an aligned hit has no event or scheduled stop time")
        before = int((hits.t_event < stop).sum())
        motion.append(dict(planner=method, n_trials=len(g), aligned=int(len(hits)),
                           before_final_knot=before, at_or_after_final_knot=len(hits)-before,
                           median_stop_minus_event_s=(stop - hits.t_event).median()))
    return pd.DataFrame(rows), pd.DataFrame(motion)


def plots(tables, out):
    colors = {"nominal": "#166978", "wide": "#b54d20"}
    methods = list(PERCEPTION_METHODS)
    order = [LABEL[m] for m in methods]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.6, 4.5), layout="constrained")
    for preset, offset in zip(PRESETS, (-.14, .14)):
        g = tables["rates"].query("preset == @preset").set_index("planner").loc[methods]
        err = np.maximum(0, [g.success - g.success_lo, g.success_hi - g.success]) * 100
        ax.errorbar(g.success * 100, y + offset, xerr=err, fmt="o", ms=5, capsize=3,
                    color=colors[preset], label=preset.title())
    ax.set(yticks=y, yticklabels=order, xlabel="Aligned first contacts (% of trials, 95% Wilson interval)",
           xlim=(-3, 103))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.13), ncol=2)
    save_fig(fig, out, "success")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True, layout="constrained")
    palette = ["#087e80", "#6e9dc2", "#c86240", "#c4ac57", "#d6d7d4", "#4b4650"]
    for ax, preset in zip(axes, PRESETS):
        g = tables["outcomes"].query("preset == @preset").set_index("planner").loc[methods]
        left = np.zeros(len(methods))
        for (key, label), color in zip(OUTCOMES.items(), palette):
            values = 100. * g[key].to_numpy() / g.n.to_numpy()
            ax.barh(y, values, left=left, color=color, height=.65, label=label)
            left += values
        ax.set(yticks=y, yticklabels=order, xlim=(0, 100), title=preset.title(),
               xlabel="Share of first-event outcomes (%)")
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False)
    save_fig(fig, out, "outcomes")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), sharey=True, layout="constrained")
    for ax, preset in zip(axes, PRESETS):
        speed = tables["by_speed"].query("preset == @preset and kind == 'trials' and n > 0")
        for method, color, marker in (("student", "#166978", "o"), ("servo_depth", "#b54d20", "s")):
            g = speed[speed.planner == method]
            x = (g.bin_lo + g.bin_hi) / 2
            err = np.maximum(0, [g.rate - g.ci_lo, g.ci_hi - g.rate]) * 100
            ax.errorbar(x, g.rate * 100, yerr=err, marker=marker, lw=1.3, ms=5,
                        capsize=3, color=color, label=LABEL[method])
        ax.set(title=preset.title(), xlabel="Realized peak partner speed (m/s)", ylim=(-2, 102))
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Aligned first contacts (%, 95% Wilson interval)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
    save_fig(fig, out, "speed")


def save_fig(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def ieee_table(paired, dest):
    """A self-contained, single-column IEEEtran table (no additional packages)."""
    rows = []
    for r in paired[paired.a == "student"].itertuples():
        p = f"{r.p_holm:.3f}" if r.p_holm >= .001 else r"<0.001"
        rows.append(f"{r.preset.title()} & {r.successes_a}/{r.n_pairs} & "
                    f"{r.successes_b}/{r.n_pairs} & {100*r.success_diff:+.1f} & ${p}$ " + r"\\")
    body = "\n".join(rows)
    dest.write_text(
        r"""\begin{table}[t]
\centering
\caption{Aligned first contacts on paired evaluation seeds. The difference is
student minus classical depth in percentage points. Two-sided exact McNemar
$p$-values are Holm-adjusted over the two presets.}
\label{tab:perception-paired}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrr}
\hline
Preset & Student & Classical & Diff. (pp) & Adj. $p$ \\
\hline
""" + body + r"""
\hline
\end{tabular}
\end{table}
"""
    )


def publish(runs, out):
    archive = runs / "scenarios" if (runs / "scenarios").is_dir() else DEFAULT_ARCHIVE
    inputs = {preset: read_raw(runs, preset) for preset in PRESETS}
    verify_freeze([meta for meta, _ in inputs.values()])
    parts = {k: [] for k in ("rates", "outcomes", "paired", "by_speed", "by_switch", "motion_at_success", "timing")}
    with tempfile.TemporaryDirectory() as tmp:
        for preset, (meta, compressed) in inputs.items():
            path = Path(tmp) / preset
            path.mkdir()
            (path / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
            (path / "trials.jsonl.gz").write_bytes(compressed)
            audit_run(path, {})  # Exact grid and matched ground-truth scenario covariates.
            data, crashes = load(path)
            contrasts = [compare(data, a, b) for a, b in (
                ("student", "servo_depth"), ("teacher", "student"), ("servo_gt_cam", "servo_depth"))]
            parts["rates"].append(summary(data, crashes).assign(preset=preset))
            parts["outcomes"].append(outcomes(data).assign(preset=preset))
            parts["paired"].append(pd.DataFrame(contrasts).assign(preset=preset))
            edges = list(map(float, DEFAULT_SPEED_BINS.split(",")))
            parts["by_speed"].append(by_speed(data, crashes, edges).assign(preset=preset))
            parts["timing"].append(timing_sensitivity(data).assign(preset=preset))
            switches, motion = describe_trajectories(data, CONFIGS[f"perception_{preset}"], archive)
            parts["by_switch"].append(switches.assign(preset=preset))
            parts["motion_at_success"].append(motion.assign(preset=preset))
        tables = {name: pd.concat(items, ignore_index=True) for name, items in parts.items()}
        test = tables["paired"].a == "student"
        tables["paired"].loc[test, "p_holm"] = holm(tables["paired"].loc[test, "p_mcnemar"].to_numpy())
        (out / "figures").mkdir(parents=True, exist_ok=True)
        plots(tables, out / "figures")
        ieee_table(tables["paired"], out / "paired_table.tex")
        for name, table in tables.items():
            table.to_csv(out / f"{name}.csv", index=False)
        for preset, (meta, compressed) in inputs.items():
            scenario_file = archive / f"{preset}.jsonl.gz"
            scenario_out = out / "scenarios" / scenario_file.name
            if scenario_file.resolve() != scenario_out.resolve():
                scenario_out.parent.mkdir(exist_ok=True)
                scenario_out.write_bytes(scenario_file.read_bytes())
            dest = out / preset
            dest.mkdir(exist_ok=True)
            (dest / "config.json").write_text(json.dumps(meta, indent=2) + "\n")
            (dest / "trials.jsonl.gz").write_bytes(compressed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "perception_results")
    args = parser.parse_args()
    publish(args.runs, args.out)
