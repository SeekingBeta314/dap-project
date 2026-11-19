"""Gymnasium environment for the dap task. Infrastructure only: the reward is
a hook (`reward_fn`) that returns 0 unless you pass one.

    import gymnasium as gym
    import dap.envs  # registers the env
    env = gym.make("dap/PalmMeet-v0", obs_mode="state+depth")

Design (sources in docs/setup.md, "RL environment"):
  - Plain `gymnasium.Env` wrapping the same kinematic loop as dap/trial.py
    (not pydrake.gym.DrakeGymEnv, which needs a Simulator-driven Diagram and
    in Drake 1.57/1.58 ignores `info_handler` and rejects reset options).
  - Policy at 20 Hz over a 100 Hz inner loop (robosuite / ManiSkill /
    LIBERO all control at 20 Hz).
  - Action: Box(-1, 1, (8,)). The first 7 are joint velocity commands as a
    fraction of each joint's limit; the 8th maps [-1, 1] to the finger
    hinge target [open, closed].
  - Observation: Dict with "state" (float32 vector, layout in STATE_LAYOUT)
    and, in "state+depth" mode, "depth" (1, H, W) float32 metres from the
    head camera with D435-style noise and one frame of latency (0 = no
    return). The partner part of "state" is the Kalman filter estimate from
    the stand-in pose sensor, not ground truth; ground truth is in `info`.
  - Episode ends: terminated on palm contact (dap.metrics.palm_contact) or
    any collision; truncated
    `t_extra` s after the partner's motion ends.
"""

from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dap.camera import FrameBuffer, HeadCameraConfig
from dap.clasp import HINGE_CLOSED
from dap.depth_noise import DepthNoiseModel
from dap.estimator import PalmEstimator
from dap.metrics import palm_angle_error_deg, palm_contact
from dap.partner import NOISE_ANG, NOISE_POS, NoisyPoseSensor
from dap.scenario import SCENARIO_PRESETS, sample_scenario
from dap.scene import OURS, Scene

# Order and size of the "state" observation vector.
STATE_LAYOUT = (
    ("q", 7),  # our joint positions (rad)
    ("qdot", 7),  # our joint velocities (rad/s)
    ("hinge", 1),  # our finger hinge angle (rad, 0 = open)
    ("palm_p", 3),  # our palm face center (m, world)
    ("palm_n", 3),  # our palm normal (world)
    ("partner_p_est", 3),  # KF estimate of the partner palm face center
    ("partner_v_est", 3),  # KF estimate of its velocity (m/s)
    ("partner_n_est", 3),  # filtered partner palm normal
    ("t", 1),  # time since the episode started (s)
    ("prev_action", 8),
)
STATE_DIM = sum(n for _, n in STATE_LAYOUT)

POS_BOUND = 3.0  # m, world positions are clipped to +- this
VEL_BOUND = 5.0  # m/s
T_BOUND = 10.0  # s


class DapEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, obs_mode="state", scenario="nominal", speed_scale=1.0, noise_scale=1.0,
                 control_hz=20.0, sim_dt=0.01, t_extra=1.0, contact_dist=0.01,
                 camera: HeadCameraConfig | None = None, camera_scale=0.25,
                 depth_noise="med", reward_fn=None, render_mode=None):
        if obs_mode not in ("state", "state+depth"):
            raise ValueError(f"obs_mode must be 'state' or 'state+depth', got {obs_mode!r}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.obs_mode = obs_mode
        self.scenario_cfg = SCENARIO_PRESETS[scenario]
        self.speed_scale = speed_scale
        self.noise_scale = noise_scale
        self.sim_dt = sim_dt
        self.n_sub = max(1, int(round(1.0 / control_hz / sim_dt)))
        self.t_extra = t_extra
        self.contact_dist = contact_dist
        self.reward_fn = reward_fn
        self.render_mode = render_mode
        self.depth_noise = depth_noise

        self.use_camera = obs_mode == "state+depth" or render_mode == "rgb_array"
        cam = camera or HeadCameraConfig()
        if camera_scale != 1:
            cam = cam.scaled(camera_scale)
        if render_mode == "rgb_array":
            cam = replace(cam, rgb=True)
        self.camera_cfg = cam

        self.action_space = spaces.Box(-1.0, 1.0, (8,), np.float32)
        obs = {"state": spaces.Box(*self._state_bounds(), dtype=np.float32)}
        if obs_mode == "state+depth":
            obs["depth"] = spaces.Box(0.0, cam.z_far, (1, cam.height, cam.width), np.float32)
        self.observation_space = spaces.Dict(obs)
        self.scenario = None

    # -- spaces -----------------------------------------------------------
    @staticmethod
    def _state_bounds():
        # Joint and velocity limits come from the model (same in every layout).
        scene = Scene(static=False)
        idx = scene.idx[OURS]
        q_lo, q_hi = scene.q_lower[idx], scene.q_upper[idx]
        v_hi = scene.v_max[idx]
        lo, hi = [], []
        bounds = {
            "q": (q_lo, q_hi), "qdot": (-v_hi, v_hi), "hinge": ([HINGE_CLOSED], [0.0]),
            "palm_p": (-POS_BOUND, POS_BOUND), "palm_n": (-1.0, 1.0),
            "partner_p_est": (-POS_BOUND, POS_BOUND),
            "partner_v_est": (-VEL_BOUND, VEL_BOUND), "partner_n_est": (-1.0, 1.0),
            "t": (0.0, T_BOUND), "prev_action": (-1.0, 1.0),
        }
        for name, n in STATE_LAYOUT:
            a, b = bounds[name]
            lo.append(np.broadcast_to(np.asarray(a, float), (n,)))
            hi.append(np.broadcast_to(np.asarray(b, float), (n,)))
        return np.concatenate(lo).astype(np.float32), np.concatenate(hi).astype(np.float32)

    # -- gym API ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        sc = sample_scenario(rng, self.scenario_cfg, self.speed_scale,
                             camera=self.camera_cfg if self.use_camera else None)
        self.scenario = sc
        self.scene = sc.scene
        self.motion = sc.partner
        sub = np.random.default_rng(rng.integers(2**63))
        self.sensor = NoisyPoseSensor(self.motion, sub, NOISE_POS * self.noise_scale,
                                      NOISE_ANG * self.noise_scale)
        self.estimator = PalmEstimator(pos_noise=self.sensor.noise_pos)

        self.t = 0.0
        self.q = sc.q_ours0.copy()
        self.qdot = np.zeros(7)
        self.hinge = 0.0
        self.prev_action = np.zeros(8)
        self.v_max = self.scene.v_max[self.scene.idx[OURS]]
        self.q_lo = self.scene.q_lower[self.scene.idx[OURS]]
        self.q_hi = self.scene.q_upper[self.scene.idx[OURS]]
        self.t_end = self.motion.t_end + self.t_extra

        if self.obs_mode == "state+depth":
            self.noise = DepthNoiseModel.for_camera(self.camera_cfg, self.depth_noise)
            if self.noise is not None:
                self.noise.reset(np.random.default_rng(rng.integers(2**63)),
                                 (self.camera_cfg.height, self.camera_cfg.width))
            self.frames = FrameBuffer(self.camera_cfg.fps, self.camera_cfg.latency_frames)

        self._advance_world()
        return self._obs(), self._info()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.prev_action = a.copy()
        qdot_cmd = a[:7] * self.v_max
        hinge_target = (a[7] + 1.0) / 2.0 * HINGE_CLOSED
        terminated = False
        for _ in range(self.n_sub):
            self.t += self.sim_dt
            q_new = np.clip(self.q + qdot_cmd * self.sim_dt, self.q_lo, self.q_hi)
            self.qdot = (q_new - self.q) / self.sim_dt
            self.q = q_new
            self.hinge = hinge_target  # TODO: rate-limit with the hinge velocity limit
            self._advance_world()
            if self._contact() or self._collision():
                terminated = True
                break
        truncated = (not terminated) and self.t >= self.t_end
        info = self._info()
        reward = float(self.reward_fn(info)) if self.reward_fn is not None else 0.0
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array" or self.scenario is None:
            return None
        return self.scene.render_rgb().astype(np.uint8)

    # -- internals ----------------------------------------------------------
    def _advance_world(self):
        """Pose both arms at self.t, update the estimator and the camera."""
        self.scene.set_partner(self.motion.q(self.t))
        self.scene.set_ours(self.q, self.hinge)
        self.estimator.update(self.t, *self.sensor.measure(self.t))
        if self.obs_mode == "state+depth" and self.frames.due(self.t):
            depth = self.scene.render_depth()
            depth = self.noise.apply(depth) if self.noise is not None else np.nan_to_num(
                np.where(depth > self.camera_cfg.z_far, 0.0, depth), posinf=0.0)
            self.frames.push(self.t, depth.astype(np.float32))

    def _contact_state(self):
        p_ours, _ = self.scene.palm(OURS)
        return palm_contact(p_ours, self.motion.p(self.t), self.motion.normal(self.t),
                            self.contact_dist)

    def _contact(self):
        return self._contact_state()[0]

    def _collision(self):
        return self.scene.clearance() < 0.0

    def _obs(self):
        palm_p, palm_n = self.scene.palm(OURS)
        p_est, v_est, _ = self.estimator.predict(0.0)
        parts = {
            "q": self.q, "qdot": self.qdot, "hinge": [self.hinge],
            "palm_p": palm_p, "palm_n": palm_n,
            "partner_p_est": p_est, "partner_v_est": v_est,
            "partner_n_est": self.estimator.normal,
            "t": [self.t], "prev_action": self.prev_action,
        }
        lo, hi = self.observation_space["state"].low, self.observation_space["state"].high
        state = np.concatenate([np.asarray(parts[k], float).ravel() for k, _ in STATE_LAYOUT])
        obs = {"state": np.clip(state, lo, hi).astype(np.float32)}
        if self.obs_mode == "state+depth":
            got = self.frames.latest(self.t)
            shape = (self.camera_cfg.height, self.camera_cfg.width)
            img = got[1] if got is not None else np.zeros(shape, np.float32)
            obs["depth"] = np.array(img, dtype=np.float32, copy=True)[None]
        return obs

    def _info(self):
        p_ours, n_ours = self.scene.palm(OURS)
        p_partner = self.motion.p(self.t)
        n_partner = self.motion.normal(self.t)
        clearance = self.scene.clearance()
        contact, normal_gap, in_plane = palm_contact(p_ours, p_partner, n_partner,
                                                     self.contact_dist)
        return {
            "t": self.t,
            "palm_gap": float(np.linalg.norm(p_ours - p_partner)),  # face-center distance
            "normal_gap": normal_gap,  # along the partner palm normal (< 0 = passed through)
            "in_plane_offset": in_plane,
            "angle_err_deg": palm_angle_error_deg(n_ours, n_partner),
            "min_arm_dist": float(self.scene.min_arm_distance()),
            "clearance": float(clearance),
            "contact": bool(contact),
            "collision": bool(clearance < 0.0),
            "partner_p": p_partner.copy(),
            "partner_v": self.motion.v(self.t).copy(),
            "partner_n": n_partner.copy(),
            "t_meet": float(self.motion.t_meet),
            "scenario_attempts": int(self.scenario.attempts),
        }

