"""Look at one sampled scenario.

    python scripts/show_scene.py --seed 3                    # Meshcat: start poses + partner path
    python scripts/show_scene.py --seed 3 --animate          # also play the partner motion
    python scripts/show_scene.py --seed 3 --scenario wide
    python scripts/show_scene.py --seed 3 --depth out.png --no-meshcat   # head-camera depth only

Meshcat draws the partner palm path (red), the certified meeting point
(green), and the nominal meeting prior (grey).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dap.camera import HeadCameraConfig  # noqa: E402
from dap.scenario import SCENARIO_PRESETS, sample_scenario  # noqa: E402
from dap.scene import Scene  # noqa: E402


def save_depth(scenario, path, noise):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dap.depth_noise import DepthNoiseModel

    cfg = scenario.scene.camera_cfg
    clean = scenario.scene.render_depth()
    clean_show = np.where(np.isfinite(clean) & (clean <= cfg.z_far), clean, 0)
    model = DepthNoiseModel.for_camera(cfg, noise)
    panels = [("clean", clean_show)]
    if model is not None:
        model.reset(np.random.default_rng(0), clean.shape)
        panels.append((f"noise={noise}", model.apply(clean)))
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 3.6))
    for ax, (title, img) in zip(np.atleast_1d(axes), panels):
        im = ax.imshow(np.where(img > 0, img, np.nan), cmap="viridis", vmin=0.3, vmax=cfg.z_far)
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(im, ax=axes, label="depth (m), white = no return")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenario", default="nominal", choices=sorted(SCENARIO_PRESETS))
    ap.add_argument("--speed-scale", type=float, default=1.0)
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--depth", type=Path, help="save the head-camera depth image here")
    ap.add_argument("--noise", default="med", help="depth noise preset for --depth")
    ap.add_argument("--no-meshcat", action="store_true")
    args = ap.parse_args()

    camera = HeadCameraConfig() if args.depth else None
    sc = sample_scenario(np.random.default_rng(args.seed), SCENARIO_PRESETS[args.scenario],
                         args.speed_scale, camera=camera)
    lay = sc.layout
    print(f"layout: distance={lay.distance:.2f} m lateral={lay.lateral:+.2f} m "
          f"yaw={lay.yaw_deg:+.0f} deg height={lay.height:+.2f} m")
    print(f"t_meet={sc.t_meet:.2f} s, dap quality={sc.witness.quality}, "
          f"attempts={sc.attempts}, rejections={sc.rejections}, sampled in {sc.sample_s:.2f} s")

    if args.depth:
        save_depth(sc, args.depth, args.noise)
    if args.no_meshcat:
        return

    from pydrake.all import Rgba, Sphere, StartMeshcat

    meshcat = StartMeshcat()
    view = Scene(lay, meshcat=meshcat)
    view.set_ours(sc.q_ours0)
    view.set_partner(sc.partner.q(0.0))
    view.diagram.ForcedPublish(view.context)
    meshcat.SetLine("partner_path", sc.partner.P.T, 2.0, Rgba(0.9, 0.3, 0.2, 1))
    for name, p, color in [("meet", sc.partner.p_meet, Rgba(0.2, 0.8, 0.3, 1)),
                           ("prior", sc.meet_prior, Rgba(0.5, 0.5, 0.5, 0.6))]:
        meshcat.SetObject(name, Sphere(0.015), color)
        from pydrake.all import RigidTransform

        meshcat.SetTransform(name, RigidTransform(p))
    print(f"Meshcat: {meshcat.web_url()}")
    if args.animate:
        for t in np.arange(0, sc.partner.t_end, 0.02):
            view.set_partner(sc.partner.q(t))
            view.diagram.ForcedPublish(view.context)
            time.sleep(0.02)
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
