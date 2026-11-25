"""Reproduce the two validation cohorts used to freeze the depth student.

Run `confirm` after collecting each candidate's selection validation JSON,
then `publish` to archive the rows and frozen checkpoint. No evaluation
streams are available through this script.
"""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dap.scoring import dap_loss
from scripts.train_student import _digest, validate

FAMILIES = {"global": ("finalwarm", "round1", "round2"),
            "spatial": ("spatial", "spatial2")}


def checked_result(runs, name, first, filename):
    score = json.loads((runs / filename).read_text())
    rows = score["rows"]
    expected = list(range(first, first + 64))
    if (score["split"] != "val" or sorted(r["seed"] for r in rows) != expected
            or any(r.get("error") for r in rows)):
        raise ValueError(f"incomplete validation cohort in {filename}")
    if score["checkpoint_sha256"] != _digest(runs / (name + ".pt")):
        raise ValueError(f"checkpoint changed since {filename}")
    table = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    successes = int(table["success"].sum())
    loss = float(dap_loss(table).mean())
    if score["success"] != successes or not np.isclose(score["loss"], loss):
        raise ValueError(f"aggregate does not match validation rows in {filename}")
    return {"name": name, "n": len(rows), "successes": successes, "mean_loss": loss,
            "sha256": score["checkpoint_sha256"]}, rows


def best(results):
    return max(results, key=lambda r: (r["successes"], -r["mean_loss"]))["name"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("confirm", "publish"))
    ap.add_argument("--runs", type=Path, default=ROOT / "banks" / "phase2student")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "policies" / "student.pt")
    ap.add_argument("--archive", type=Path, default=ROOT / "docs/perception_results/training/student")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    runs = args.runs
    archive_rows, summaries, winners = [], {}, []
    def read(name, first, filename):
        summary, rows = checked_result(runs, name, first, filename)
        archive_rows.extend({"candidate": name, "cohort_first": first, **row} for row in rows)
        return summary
    diagnostic = read("warm", 300, "initial-val.json")
    for family, names in FAMILIES.items():
        summaries[family] = [read(name, 300, f"val-{name}.json") for name in names]
        winners.append(best(summaries[family]))
    confirm = []
    for name in winners:
        filename = f"confirm-{name}.json"
        if args.stage == "confirm" and not (runs / filename).exists():
            validate(argparse.Namespace(checkpoint=runs / (name + ".pt"), first=400, count=64,
                                        workers=args.workers, out=runs / filename))
        confirm.append(read(name, 400, filename))
    selected = best(confirm)
    print(json.dumps({"shortlist": winners, "confirmation": confirm, "selected": selected}),
          flush=True)
    if args.stage == "confirm":
        return
    archive = args.archive
    archive.mkdir(exist_ok=True, parents=True)
    args.checkpoint.parent.mkdir(exist_ok=True, parents=True)
    src = runs / f"{selected}.pt"
    if args.checkpoint.exists() and _digest(src) != _digest(args.checkpoint):
        raise ValueError(f"would replace a different frozen checkpoint at {args.checkpoint}")
    tmp = args.checkpoint.with_suffix(".tmp")
    shutil.copyfile(src, tmp)
    tmp.replace(args.checkpoint)
    record = {"rule": "Most aligned contacts, then smaller mean benchmark loss, then listed order.",
              "split": "validation", "development_seeds_inclusive": [300, 363],
              "initial_diagnostic": diagnostic, "families": summaries,
              "confirmation_seeds_inclusive": [400, 463], "confirmation": confirm,
              "selected": selected, "sha256": _digest(src)}
    (archive / "selection.json").write_text(json.dumps(record, indent=2) + "\n")
    with gzip.open(archive / "validation.jsonl.gz", "wt") as f:
        for row in archive_rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    models = ["pose", "warm", *FAMILIES["global"], *FAMILIES["spatial"]]
    fits = {name: json.loads((runs / f"{name}.json").read_text()) for name in models}
    with gzip.open(archive / "fits.json.gz", "wt") as f:
        json.dump(fits, f, separators=(",", ":"))
    behavior = archive / "behavior"
    behavior.mkdir(exist_ok=True)
    # These intermediate policies generated training states. Preserve them so
    # a reproducer can regenerate those trajectories directly.
    for name in ("initial-actor", "warm", "round1", "spatial"):
        if name != selected:
            shutil.copyfile(runs / f"{name}.pt", behavior / f"{name}.pt")
    datasets = {}
    for name in ("warm", "ik", "dagger", "second", "spatialdata"):
        manifest = json.loads((runs / name / "manifest.json").read_text())
        paths = sorted((runs / name).glob("[0-9]"*8 + ".npz"))
        frames = 0
        for path in paths:
            with np.load(path, allow_pickle=False) as row:
                frames += len(row["privileged"])
        datasets[name] = {**manifest, "episodes": len(paths), "labeled_states": frames,
                          "first_seed": int(paths[0].stem), "last_seed": int(paths[-1].stem)}
    (archive / "datasets.json").write_text(json.dumps(datasets, indent=2) + "\n")


if __name__ == "__main__":
    main()
