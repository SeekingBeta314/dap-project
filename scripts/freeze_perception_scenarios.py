"""Solve each reserved scenario once and retain the actual inputs used by every controller."""

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dap.bank import scenario_spec  # noqa: E402
from dap.eval_scenarios import DEFAULT_ARCHIVE, freeze_record  # noqa: E402
from dap.experiment import CONFIGS  # noqa: E402
from dap.scenario import SCENARIO_PRESETS, ScenarioError, sample_scenario  # noqa: E402


def solve(job):
    preset, seed = job
    cfg = CONFIGS[f"perception_{preset}"]
    try:
        scenario = sample_scenario(cfg.rngs(seed)["scenario"], SCENARIO_PRESETS[preset])
        spec = scenario_spec(scenario)
    except ScenarioError:
        spec = None
    return freeze_record(preset, cfg.split, seed, spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=("nominal", "wide"))
    parser.add_argument("--out", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f"{args.preset}.jsonl.gz"
    if dest.exists():
        parser.error(f"{dest} already exists; use a separate --out to regenerate")
    jobs = [(args.preset, s) for s in CONFIGS[f"perception_{args.preset}"].seeds()]
    with ProcessPoolExecutor(args.workers) as pool, tempfile.NamedTemporaryFile(dir=args.out) as tmp:
        with gzip.GzipFile(fileobj=tmp, mode="wb", mtime=0, filename="") as gz:
            for r in pool.map(solve, jobs):
                gz.write((json.dumps(r, separators=(",", ":")) + "\n").encode())
        tmp.flush()
        dest.hardlink_to(tmp.name)
    print(f"Frozen {len(jobs)} {args.preset} scenarios at {dest}.")
