"""Open the two-arm scene in Meshcat and draw one random partner path.

    python scripts/show_scene.py [--seed N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from pydrake.all import Rgba, StartMeshcat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dap.partner import PartnerPath, sample_params  # noqa: E402
from dap.scene import build_scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meshcat = StartMeshcat()
    diagram, plant, _ = build_scene(meshcat)
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)

    path = PartnerPath(sample_params(np.random.default_rng(args.seed)))
    ts = np.linspace(0, path.params.duration, 60)
    pts = np.column_stack([path.p(t) for t in ts])
    meshcat.SetLine("partner_path", pts, 2.0, Rgba(0.9, 0.3, 0.2, 1))

    print(f"Meshcat: {meshcat.web_url()}  (Ctrl-C to quit)")
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
