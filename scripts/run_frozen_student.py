"""Verify the entire student deployment before and after a held-out evaluation shard."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dap.experiment import CONFIGS  # noqa: E402
from run_perception_study import atomic_json, digest  # noqa: E402


def deployment(checkpoint):
    sources = [ROOT / "dap" / "student.py"]
    spatial = ROOT / "dap" / "student_spatial.py"
    if spatial.exists():
        sources.append(spatial)
    return {"weights": digest(checkpoint), "sources": {
        str(src.relative_to(ROOT)): digest(src) for src in sources}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=("nominal", "wide"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "perception_study")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    cfg = CONFIGS[f"perception_{args.preset}"]
    count = cfg.n_seeds - args.offset if args.count is None else args.count
    if not (0 <= args.offset < cfg.n_seeds and 0 < count <= cfg.n_seeds - args.offset):
        parser.error("invalid range")
    method_dir = args.out / "eval" / args.preset / "student"
    attest = method_dir / "deployment"
    attest.mkdir(parents=True, exist_ok=True)
    identity = deployment(args.checkpoint)
    snapshot = attest / "sources.json"
    if not snapshot.exists() and any(method_dir.glob("[0-9]*.json")):
        raise SystemExit("existing student trials have no source receipt")
    atomic_json(snapshot, identity, immutable=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_perception_study.py"),
                    args.preset, "student", "--checkpoint", str(args.checkpoint), "--out", str(args.out),
                    "--workers", str(args.workers), "--offset", str(args.offset), "--count", str(count)],
                   check=True)
    if deployment(args.checkpoint) != identity:
        raise SystemExit("deployment changed during evaluation")
    seeds = cfg.seeds()[args.offset:args.offset+count]
    atomic_json(attest / f"{seeds[0]}-{seeds[-1]}.json",
                dict(source_receipt=digest(snapshot), first=seeds[0], last=seeds[-1]), immutable=True)
