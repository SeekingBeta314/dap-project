"""Build a scenario bank for RL training (dap/bank.py).

    python scripts/build_bank.py --n 20000 --preset nominal --workers 8
    python scripts/build_bank.py --n 200 --preset wide --out banks/wide_small.pkl

Scenarios come from training streams (dap.experiment.train_rngs), disjoint
from every evaluation seed. Seeds whose sampling gives up are skipped and
listed in the bank (`failed_seeds`). The bank records what built it
(`provenance`), and the script prints the file's SHA-256. Banks are not
committed (banks/ is in .gitignore): share the file and check its SHA-256
(rebuilding on another machine is not known to give the same scenarios).
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dap.bank import build_bank, save_bank  # noqa: E402
from dap.scenario import SCENARIO_PRESETS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="number of scenarios")
    ap.add_argument("--preset", default="nominal", choices=sorted(SCENARIO_PRESETS))
    ap.add_argument("--first-seed", type=int, default=0, help="first training seed")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, help="default: banks/train_<preset>_<n>.pkl")
    args = ap.parse_args()

    out = args.out or ROOT / "banks" / f"train_{args.preset}_{args.n}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def progress(done, total):
        el = time.perf_counter() - t0
        print(f"  {done}/{total} scenarios  {el:.0f} s", flush=True)

    bank = build_bank(args.n, args.preset, args.first_seed, args.workers, progress)
    save_bank(bank, out)
    print(f"wrote {out}: {len(bank['specs'])} scenarios, {len(bank['failed_seeds'])} seeds "
          f"skipped, {bank['build_s']:.0f} s")
    print(f"skipped seeds: {bank['failed_seeds']}")
    print(f"provenance: {bank['provenance']}")
    print(f"sha256: {hashlib.sha256(out.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
