"""Compare a separate, frozen-student rerun against the published evaluation."""

import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyze_perception import student_sources  # noqa: E402
from dap.experiment import CONFIGS  # noqa: E402
from publish_study import SCENARIO_FIELDS  # noqa: E402


def archive_rows(path):
    with gzip.open(path, "rt") as f:
        return [r for line in f if (r := json.loads(line))["planner"] == "student"]


def ordered(records, seeds):
    if len(records) != len(seeds) or sorted(r["seed"] for r in records) != list(seeds):
        raise ValueError("duplicate, missing or unexpected seed")
    return sorted(records, key=lambda r: r["seed"])


def event(row):
    if row["outcome"] == "contact":
        return "aligned" if row["success"] else "misaligned"
    return row["outcome"]


def build(official, repetition, out):
    out.mkdir(parents=True, exist_ok=True)
    paired, summary = [], []
    for preset in ("nominal", "wide"):
        seeds = CONFIGS[f"perception_{preset}"].seeds()
        original_meta = json.loads((official / preset / "config.json").read_text())["methods"]["student"]
        original = ordered(archive_rows(official / preset / "trials.jsonl.gz"), seeds)
        source = repetition / "eval" / preset / "student"
        if source.is_dir():
            repeat_meta = json.loads((source / "manifest.json").read_text())
            repeat_meta["sources"].update(student_sources(source, repeat_meta, seeds))
            repeat = [json.loads((source / f"{s}.json").read_text()) for s in seeds]
        else:
            repeat_meta = json.loads((repetition / f"{preset}_manifest.json").read_text())
            repeat = archive_rows(repetition / f"{preset}_repeat.jsonl.gz")
        repeat = ordered(repeat, seeds)
        if {k: v for k, v in original_meta.items() if k != "git"} != {
                k: v for k, v in repeat_meta.items() if k != "git"}:
            raise ValueError(f"{preset}: simulation, sensor, software or deployed model differs")

        subset = []
        for a, b in zip(original, repeat, strict=True):
            if (a["scenario_sha256"] != b["scenario_sha256"] or
                    not np.allclose([a[k] for k in SCENARIO_FIELDS], [b[k] for k in SCENARIO_FIELDS],
                                    atol=1e-7, rtol=1e-7)):
                raise ValueError(f"{preset} {a['seed']}: scenario inputs differ")
            if a["error_kind"] or b["error_kind"]:
                raise ValueError(f"{preset} {a['seed']}: trial error")
            row = dict(preset=preset, seed=a["seed"], scenario_sha256=a["scenario_sha256"],
                       original_class=event(a), repeat_class=event(b))
            for metric in ("rot_err", "t_event", "time_err", "min_palm_dist"):
                row[f"original_{metric}"] = a[metric]
                row[f"repeat_{metric}"] = b[metric]
            subset.append(row)
        paired.extend(subset)
        df = pd.DataFrame(subset)
        a = df.original_class.eq("aligned")
        b = df.repeat_class.eq("aligned")
        shared = df.original_rot_err.notna() & df.repeat_rot_err.notna()
        rotation = (df.loc[shared, "repeat_rot_err"] - df.loc[shared, "original_rot_err"]).abs()
        event_time = (df.repeat_t_event - df.original_t_event).abs().dropna()
        summary.append(dict(preset=preset, n=len(df), original_successes=int(a.sum()),
                            repeat_successes=int(b.sum()), original_only=int((a & ~b).sum()),
                            repeat_only=int((b & ~a).sum()), binary_flips=int((a != b).sum()),
                            class_flips=int((df.original_class != df.repeat_class).sum()),
                            both_contact=int(shared.sum()), median_abs_rotation_diff_deg=rotation.median(),
                            p95_abs_rotation_diff_deg=rotation.quantile(.95), max_abs_rotation_diff_deg=rotation.max(),
                            event_in_both=len(event_time), median_abs_event_time_diff_ms=event_time.median()*1000,
                            max_abs_event_time_diff_ms=event_time.max()*1000))
        (out / f"{preset}_manifest.json").write_text(json.dumps(repeat_meta, indent=2) + "\n")
        raw = "".join(json.dumps(r, allow_nan=False) + "\n" for r in repeat).encode()
        (out / f"{preset}_repeat.jsonl.gz").write_bytes(gzip.compress(raw, mtime=0))
    frame = pd.DataFrame(paired)
    frame.to_csv(out / "per_seed.csv", index=False)
    frame.groupby(["preset", "original_class", "repeat_class"]).size().rename("n").to_csv(out / "transitions.csv")
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repetition", type=Path, help="repeat run folder or published repeat archive")
    parser.add_argument("--official", type=Path, default=ROOT / "docs" / "perception_results")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "perception_results" / "portability")
    args = parser.parse_args()
    build(args.official, args.repetition, args.out)
