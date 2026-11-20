"""Generate supervised head-camera examples using the disjoint training RNGs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dap.camera import HeadCameraConfig
from dap.depth_noise import DepthNoiseModel
from dap.experiment import train_rngs
from dap.partner import min_jerk
from dap.pixel_policy import fine_depth, pool_depth
from dap.scenario import SCENARIO_PRESETS, ScenarioError, sample_scenario
from dap.scene import OURS, Q_READY


def collect(args):
    seed, out, independent, fine = args
    dest = out / f"{seed:06d}.npz"
    if dest.exists():
        with np.load(dest, allow_pickle=False) as row:
            if (bool(row.get("independent_poses", False)) != independent
                    or bool(row.get("fine_view", False)) != fine):
                raise ValueError(f"{dest} was generated with different collection settings")
        return True
    rngs = train_rngs(seed)
    camera = HeadCameraConfig().scaled(.25)
    try:
        sc = sample_scenario(rngs["scenario"], SCENARIO_PRESETS["nominal"], camera=camera)
    except ScenarioError:
        return False
    noise = DepthNoiseModel.for_camera(camera, "med")
    noise.reset(rngs["camera"], (camera.height, camera.width))
    rng = rngs["sensor"]
    v_max = sc.scene.v_max[sc.scene.idx[OURS]]
    lo = sc.scene.q_lower[sc.scene.idx[OURS]]
    hi = sc.scene.q_upper[sc.scene.idx[OURS]]
    states, frames = [], []
    for t in np.linspace(.05, sc.partner.t_end + .85, 20):
        # Broaden the expert trajectory to include the states a student visits:
        # vary its progress, mix in a nominal posture and perturb its joints.
        s = min_jerk(t / (sc.t_meet * rng.uniform(.6, 1.5)))
        goal = Q_READY + (sc.q_ours_meet - Q_READY) * rng.uniform(0., 1.)
        if independent and rng.random() < .65:
            # Another plausible reaching posture, drawn independently of the
            # expert's selected IK branch, prevents a target leak through q.
            goal = Q_READY + rng.normal(0, [.30, .35, .50, .40, .85, .35, .85])
        q = np.clip(sc.q_ours0 + s * (goal - sc.q_ours0) + rng.normal(0, .10, 7), lo, hi)
        sc.scene.set_ours(q)
        sc.scene.set_partner(sc.partner.q(max(0, t - rng.uniform(.034, .067))))
        image = noise.apply(sc.scene.render_depth())
        frames.append(fine_depth(image) if fine else pool_depth(image))
        states.append(np.r_[sc.q_ours0, q, t])
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, image=np.round(np.array(frames) * 1000).astype(np.uint16),
                        state=np.asarray(states, np.float32), goal=sc.q_ours_meet,
                        v_max=v_max, independent_poses=independent, fine_view=fine)
    tmp.replace(dest)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "banks/pixel_data")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--independent-poses", action="store_true",
                        help="randomize most training arm postures independently of the label")
    parser.add_argument("--fine-view", action="store_true", help="store a higher-resolution central crop")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, args.out, args.independent_poses, args.fine_view) for seed in range(args.first, args.first + args.count)]
    if args.workers == 1:
        done = sum(map(collect, jobs))
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            done = sum(pool.map(collect, jobs))
    print(f"Collected {done} scenarios out of {len(jobs)} attempted.")


if __name__ == "__main__":
    main()
