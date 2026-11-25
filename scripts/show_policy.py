"""Save an evaluation trial as a GIF with the policy's actual depth observation."""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dap.camera import HeadCameraConfig
from dap.experiment import Condition, ExperimentConfig
from dap.policy_eval import make_eval_env
from dap.scene import Scene
from dap.trial import score_log
from dap.metrics import TrialResult
from run_experiment import _load_policy_factory


def frame(env, obs, spectator, seed):
    spectator.set_ours(env.q)
    spectator.set_partner(env.motion.q(env.t))
    side = Image.fromarray(spectator.render_rgb())
    # White means a missing depth return. The metric scale is fixed across frames.
    depth = obs["depth"][-1]
    gray = np.where(depth > 0, 220 * (1 - np.clip(depth / 3, 0, 1)), 255)
    camera = Image.fromarray(gray.astype("uint8")).resize(side.size, Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (side.width * 2, side.height + 44), "white")
    canvas.paste(side, (0, 24))
    canvas.paste(camera, (side.width, 24))
    pen = ImageDraw.Draw(canvas)
    pen.text((8, 5), f"Side view   |   held-out seed {seed}", fill="black")
    pen.text((side.width + 8, 5), "Newest depth frame available to the policy", fill="black")
    pen.text((8, side.height + 28), f"Simulation time {env.t:.2f} s", fill="black")
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", help="module:factory, as in run_experiment.py")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = ExperimentConfig(name="replay")
    cond = Condition("pixels", 1.0)
    policy = _load_policy_factory(args.policy)(cfg, cond)
    env = make_eval_env(cfg, cond, policy.obs_mode)
    obs, _ = env.reset(options={"trial_seed": args.seed, "split": cfg.split})
    if hasattr(policy, "reset"):
        policy.reset()
    spectator = Scene(env.scenario.layout, camera=HeadCameraConfig(
        width=320, height=180, rgb=True, position=(0.35, -1.75, 1.05),
        target=(0.62, 0, 0.48)))
    frames = [frame(env, obs, spectator, args.seed)]
    durations = []
    done = env.event is not None
    while not done:
        start = env.t
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        frames.append(frame(env, obs, spectator, args.seed))
        durations.append(round((env.t - start) * 1000))
        done = terminated or truncated
    result = TrialResult(args.seed, cond.planner, cond.noise_scale)
    score_log(result, env.log, env.motion, cfg.tolerances, env.v_max, env.event)
    label = f"Outcome: {result.outcome}"
    if result.contact:
        label += f"   |   rotation error {result.rot_err:.1f} deg   |   success {result.success}"
    ImageDraw.Draw(frames[-1]).text((env.camera_cfg.width, frames[-1].height - 16),
                                    label, fill="black")
    frames[0].save(args.output, save_all=True, append_images=frames[1:],
                   duration=durations + [1500], loop=0, disposal=2)
    print(label)


if __name__ == "__main__":
    main()
