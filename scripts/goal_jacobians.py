"""Save the palm's spatial Jacobian at each training demonstration endpoint."""
import argparse
from pathlib import Path
import sys

import numpy as np
from pydrake.all import JacobianWrtVariable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dap.scene import OURS, P_PALM_FACE, Scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scene = Scene(static=False)
    plant = scene.plant
    world = plant.world_frame()
    labels = {}
    for directory in args.data:
        for path in directory.glob("[0-9]*.npz"):
            seed = int(path.stem)
            if seed in labels:
                raise ValueError(f"duplicate seed {seed} in input directories")
            with np.load(path) as row:
                scene.set_ours(row["goal"])
            jac = plant.CalcJacobianSpatialVelocity(
                scene.plant_context, JacobianWrtVariable.kQDot,
                scene.palm_frames[OURS], P_PALM_FACE, world, world)
            labels[seed] = jac[:, scene.idx[OURS]]
    np.savez(args.out, seeds=list(labels), jacobian=np.stack(list(labels.values())).astype("float32"))


if __name__ == "__main__":
    main()
