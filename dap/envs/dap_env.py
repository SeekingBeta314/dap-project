"""Gymnasium environment for the dap task. Infrastructure only: the reward is
a hook (`reward_fn`) that returns 0 unless you pass one.

    import gymnasium as gym
    import dap.envs  # registers the env
    env = gym.make("dap/PalmMeet-v0", obs_mode="pixels", bank="banks/train_nominal.pkl")

Design (sources in docs/setup.md, "RL environment"):
  - Plain `gymnasium.Env` wrapping the same kinematic loop as dap/trial.py
    (not pydrake.gym.DrakeGymEnv, which needs a Simulator-driven Diagram and
    in Drake 1.57/1.58 ignores `info_handler` and rejects reset options).
    It uses the trial's arm limits (dap.trial.limit_command) and the
    scorer's event rule (dap.scoring.EventMonitor), so an episode ends
    exactly where the same motion would end a trial.
  - Policy at 20 Hz over a 100 Hz inner loop (robosuite / ManiSkill /
    LIBERO all control at 20 Hz).
  - Action: Box(-1, 1, (7,)): joint velocity commands as a fraction of each
    joint's limit. The hands are rigid; the task is purely kinematic.
    Non-finite actions are treated as 0 and counted in info.
  - Observation modes (`obs_mode`):
      "pixels"      {"proprio", "depth"}: our arm's own state (PROPRIO_LAYOUT)
                    and the last `frame_stack` head-camera depth frames.
                    Nothing about the partner except what the camera sees.
                    This is what an end-to-end pixel policy gets.
      "privileged"  {"state"}: our arm's state plus the partner palm's TRUE
                    pose and velocity (PRIVILEGED_LAYOUT). For training a
                    teacher or a critic only; never a deployed policy. By
                    default (`privileged_delay=True`) the partner part is
                    the true state when the newest delivered camera frame
                    was captured, i.e. as old as what the pixel policy
                    sees; `privileged_delay=False` gives the current state.
      "state"       {"state"}: our arm plus the stand-in sensor's Kalman
                    estimate of the partner palm (STATE_LAYOUT).
      "state+depth" "state" plus the newest depth frame.
    Depth frames are (H, W) float32 metres with D435-style noise, one frame
    of latency and 0 = no return. `info["privileged"]` always carries the
    privileged vector (labels for distillation); `info` is ground truth and
    must never be fed to a deployed policy.
  - Scenarios: `reset(options={"trial_seed": s})` samples experiment seed s
    exactly as a trial does (evaluation, dap.policy_eval); add `"split":
    "val"` for validation seed s (dap.experiment.SPLITS). Otherwise, with
    `bank=` (dap.bank, training seeds disjoint from evaluation) a reset
    rebuilds a random banked scenario in tens of ms; without a bank it
    samples a fresh one from a training stream (about 1 s).
  - Episode ends: terminated at the first contact or collision (the same
    rule as the scorer); truncated at the same horizon as a trial
    (dap.trial.horizon_steps). With `record_log=True` the env keeps the
    trial's ground-truth log (dap.trial.TrialLog) so dap.policy_eval can
    score an episode with dap.scoring.evaluate, exactly like a trial.
"""

from collections import deque
from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dap.bank import ScenarioBank
from dap.camera import FrameBuffer, HeadCameraConfig, sensed_depth
from dap.depth_noise import DepthNoiseModel
from dap.estimator import PalmEstimator
from dap.experiment import SPLITS, train_rngs
from dap.partner import NOISE_ANG, NOISE_POS, NoisyPoseSensor
from dap.scenario import SCENARIO_PRESETS, sample_scenario
from dap.scene import OURS, Scene
from dap.scoring import EventMonitor, Tolerances
from dap.trial import TrialLog, horizon_steps, limit_command

OBS_MODES = ("pixels", "privileged", "state", "state+depth")

# Our arm's own state: what the robot knows without looking at the partner.
# palm_* follow from q by forward kinematics.
PROPRIO_LAYOUT = (
    ("q", 7),  # our joint positions (rad)
    ("qdot", 7),  # our joint velocities (rad/s)
    ("palm_p", 3),  # our palm face center (m, world)
    ("palm_n", 3),  # our palm normal (world)
    ("palm_x", 3),  # our finger direction (world)
    ("t", 1),  # time since the episode started (s)
    ("prev_action", 7),
)
# Ground truth about the partner palm (training-time only).
PARTNER_TRUTH_LAYOUT = (
    ("partner_p", 3),  # palm face center (m, world)
    ("partner_v", 3),  # its velocity (m/s)
    ("partner_n", 3),  # palm normal
    ("partner_x", 3),  # finger direction
)
PRIVILEGED_LAYOUT = PROPRIO_LAYOUT + PARTNER_TRUTH_LAYOUT

# Order and size of the "state" observation vector (stand-in sensor modes).
STATE_LAYOUT = (
    ("q", 7),
    ("qdot", 7),
    ("palm_p", 3),
    ("palm_n", 3),
    ("partner_p_est", 3),  # KF estimate of the partner palm face center
    ("partner_v_est", 3),  # KF estimate of its velocity (m/s)
    ("partner_n_est", 3),  # filtered partner palm normal
    ("t", 1),
    ("prev_action", 7),
)


def _dim(layout):
    return sum(n for _, n in layout)


STATE_DIM = _dim(STATE_LAYOUT)
PROPRIO_DIM = _dim(PROPRIO_LAYOUT)
PRIVILEGED_DIM = _dim(PRIVILEGED_LAYOUT)

# Depth frames stacked in "pixels" mode: one frame carries no velocity.
# Ours, in line with frame stacking in pixel RL (docs/setup.md, section 8).
FRAME_STACK = 3

POS_BOUND = 3.0  # m, world positions are clipped to +- this
VEL_BOUND = 5.0  # m/s
T_BOUND = 10.0  # s


class DapEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, obs_mode="state", scenario=None, noise_scale=1.0,
                 control_hz=20.0, sim_dt=0.01, t_extra=1.0,
                 touch_dist=Tolerances().touch_dist, rot_tol=Tolerances().rot_tol,
                 camera: HeadCameraConfig | None = None, camera_scale=0.25, depth_noise="med",
                 frame_stack=FRAME_STACK, privileged_delay=True, bank=None, record_log=False,
                 reward_fn=None, render_mode=None):
        if obs_mode not in OBS_MODES:
            raise ValueError(f"obs_mode must be one of {OBS_MODES}, got {obs_mode!r}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.bank = None
        if bank is not None:
            self.bank = bank if isinstance(bank, ScenarioBank) else ScenarioBank(bank)
            if scenario not in (None, self.bank.preset):
                raise ValueError(f"bank was sampled from the {self.bank.preset!r} preset, "
                                 f"not {scenario!r}")
            scenario = self.bank.preset
        self.scenario_name = scenario or "nominal"
        self.obs_mode = obs_mode
        self.scenario_cfg = SCENARIO_PRESETS[self.scenario_name]
        self.noise_scale = noise_scale
        self.sim_dt = sim_dt
        self.n_sub = max(1, int(round(1.0 / control_hz / sim_dt)))
        self.t_extra = t_extra
        self.touch_dist = touch_dist
        self.rot_tol = rot_tol
        self.reward_fn = reward_fn
        self.render_mode = render_mode
        self.depth_noise = depth_noise
        self.frame_stack = int(frame_stack)
        self.privileged_delay = bool(privileged_delay)
        self.record_log = record_log

        self.uses_depth = obs_mode in ("pixels", "state+depth")
        self.uses_estimator = obs_mode in ("state", "state+depth")
        self.use_camera = self.uses_depth or render_mode == "rgb_array"
        cam = camera or HeadCameraConfig()
        if camera_scale != 1:
            cam = cam.scaled(camera_scale)
        if render_mode == "rgb_array":
            cam = replace(cam, rgb=True)
        self.camera_cfg = cam

        self.action_space = spaces.Box(-1.0, 1.0, (7,), np.float32)
        bounds = self._bounds()
        H, W = cam.height, cam.width
        if obs_mode == "pixels":
            obs = {"proprio": self._box(PROPRIO_LAYOUT, bounds),
                   "depth": spaces.Box(0.0, cam.z_far, (self.frame_stack, H, W), np.float32)}
        elif obs_mode == "privileged":
            obs = {"state": self._box(PRIVILEGED_LAYOUT, bounds)}
        else:
            obs = {"state": self._box(STATE_LAYOUT, bounds)}
            if obs_mode == "state+depth":
                obs["depth"] = spaces.Box(0.0, cam.z_far, (1, H, W), np.float32)
        self.observation_space = spaces.Dict(obs)
        self._privileged_box = self._box(PRIVILEGED_LAYOUT, bounds)
        self.scenario = None
        self.log = None

    # -- spaces -----------------------------------------------------------
    @staticmethod
    def _bounds():
        # Joint and velocity limits come from the model (same in every layout).
        scene = Scene(static=False)
        idx = scene.idx[OURS]
        v_hi = scene.v_max[idx]
        pos, unit = (-POS_BOUND, POS_BOUND), (-1.0, 1.0)
        return {
            "q": (scene.q_lower[idx], scene.q_upper[idx]), "qdot": (-v_hi, v_hi),
            "palm_p": pos, "palm_n": unit, "palm_x": unit,
            "partner_p_est": pos, "partner_v_est": (-VEL_BOUND, VEL_BOUND),
            "partner_n_est": unit,
            "partner_p": pos, "partner_v": (-VEL_BOUND, VEL_BOUND),
            "partner_n": unit, "partner_x": unit,
            "t": (0.0, T_BOUND), "prev_action": unit,
        }

    @staticmethod
    def _box(layout, bounds):
        lo, hi = [], []
        for name, n in layout:
            a, b = bounds[name]
            lo.append(np.broadcast_to(np.asarray(a, float), (n,)))
            hi.append(np.broadcast_to(np.asarray(b, float), (n,)))
        return spaces.Box(np.concatenate(lo).astype(np.float32),
                          np.concatenate(hi).astype(np.float32), dtype=np.float32)

    # -- gym API ------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        """`options={"trial_seed": s}` replays experiment trial seed `s`
        (dap.experiment.trial_rngs: its scenario, stand-in sensor noise and
        depth noise), so a policy is evaluated on exactly the seeds the other
        controllers use; `options={"trial_seed": s, "split": "val"}` replays
        validation seed `s` instead. Otherwise a banked (or freshly sampled)
        training scenario."""
        super().reset(seed=seed)
        rng = self.np_random
        camera = self.camera_cfg if self.use_camera else None
        trial_seed = (options or {}).get("trial_seed")
        camera_rng = None
        if trial_seed is not None:
            split = options.get("split", "eval")
            streams = SPLITS[split](int(trial_seed))
            sc = sample_scenario(streams["scenario"], self.scenario_cfg, camera=camera)
            sensor_rng, camera_rng = streams["sensor"], streams["camera"]
            self.scenario_id = ("trial" if split == "eval" else split, int(trial_seed))
        elif self.bank is not None:
            i = int(rng.integers(len(self.bank)))
            sc = self.bank.get(i, camera)
            sensor_rng = np.random.default_rng(rng.integers(2**63))
            self.scenario_id = ("bank", self.bank.specs[i].get("seed", i))
        else:
            # A training stream: np_random(s) is the same stream as the
            # evaluation scenario stream of seed s (numpy drops trailing
            # zero words of the entropy), so it must not sample directly.
            sc = sample_scenario(train_rngs(int(rng.integers(2**63)))["scenario"],
                                 self.scenario_cfg, camera=camera)
            sensor_rng = np.random.default_rng(rng.integers(2**63))
            self.scenario_id = ("sampled", None)
        self.scenario = sc
        self.scene = sc.scene
        self.motion = sc.partner
        if self.uses_estimator:
            self.sensor = NoisyPoseSensor(self.motion, sensor_rng, NOISE_POS * self.noise_scale,
                                          NOISE_ANG * self.noise_scale)
            self.estimator = PalmEstimator(pos_noise=self.sensor.noise_pos)

        self.t = 0.0
        self.k = 0  # inner steps taken; t = k * sim_dt (no float drift)
        self.q = sc.q_ours0.copy()
        self.qdot = np.zeros(7)
        self.prev_action = np.zeros(7)
        self.n_bad_actions = 0
        self.v_max = self.scene.v_max[self.scene.idx[OURS]]
        self.q_lo = self.scene.q_lower[self.scene.idx[OURS]]
        self.q_hi = self.scene.q_upper[self.scene.idx[OURS]]
        # Same horizon as a trial: the last sample is at k_end * sim_dt.
        self.k_end = horizon_steps(self.motion.t_end, self.t_extra, self.sim_dt)
        self.monitor = EventMonitor(self.touch_dist)
        self.event = None
        self.log = TrialLog() if self.record_log else None

        if self.uses_depth:
            self.noise = DepthNoiseModel.for_camera(self.camera_cfg, self.depth_noise)
            if self.noise is not None:
                if camera_rng is None:
                    camera_rng = np.random.default_rng(rng.integers(2**63))
                self.noise.reset(camera_rng, (self.camera_cfg.height, self.camera_cfg.width))
        # Camera timing in every mode: each capture holds its depth frame
        # (None when not rendered) and the partner truth it saw, which the
        # delayed privileged vector uses.
        self.frames = FrameBuffer(self.camera_cfg.fps, self.camera_cfg.latency_frames)
        self.stack = deque(maxlen=self.frame_stack)  # delivered depth frames, newest last
        self._last_capture = None

        self._advance_world()
        self.first_capture = (self.t, (None, self._partner_now()))  # the capture at t = 0
        return self._obs(), self._info()

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (7,) or not np.all(np.isfinite(a)):
            self.n_bad_actions += 1
            a = np.zeros(7)
        a = np.clip(a, -1.0, 1.0)
        self.prev_action = a.copy()
        qdot_cmd = a * self.v_max
        for _ in range(self.n_sub):
            # Move (within the arm's limits), then measure the world at t + dt.
            q_new, _ = limit_command(self.q + qdot_cmd * self.sim_dt, self.q, self.v_max,
                                     self.sim_dt, self.q_lo, self.q_hi)
            self.qdot = (q_new - self.q) / self.sim_dt
            self.q = q_new
            self.k += 1
            self.t = self.k * self.sim_dt
            self._advance_world()
            if self.event is not None or self.k >= self.k_end:
                break
        terminated = self.event is not None
        truncated = (not terminated) and self.k >= self.k_end
        info = self._info()
        reward = float(self.reward_fn(info)) if self.reward_fn is not None else 0.0
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array" or self.scenario is None:
            return None
        return self.scene.render_rgb().astype(np.uint8)

    # -- internals ----------------------------------------------------------
    def _advance_world(self):
        """Pose both arms at self.t, measure the ground truth, run the event
        rule, log it, and update the stand-in estimator and the camera."""
        self.scene.set_partner(self.motion.q(self.t))
        self.scene.set_ours(self.q)
        m = self.scene.measure()
        self.truth, self.partner_v = m, self.motion.v(self.t)
        self.event = self.monitor.step(m["palm_p"], m["palm_R"], m["partner_p"],
                                       m["partner_R"], m["clearance"], m["palm_palm"])
        if self.log is not None:  # the same rows dap.trial.run_trial records
            self.log.append(t=self.t, q=self.q.copy(), partner_v=self.partner_v, **m)
        if self.uses_estimator:
            self.estimator.update(self.t, *self.sensor.measure(self.t))
        if self.frames.due(self.t):
            depth = None
            if self.uses_depth:
                depth = sensed_depth(self.scene.render_depth(), self.noise,
                                     self.camera_cfg.z_far)
            self.frames.push(self.t, (depth, self._partner_now()))
        got = self.frames.latest(self.t)
        if self.uses_depth and got is not None and got[0] != self._last_capture:
            self._last_capture = got[0]
            self.stack.append(got[1][0])

    def _partner_now(self):
        """True partner palm face (position, velocity, frame) now."""
        return self.truth["partner_p"], self.partner_v, self.truth["partner_R"]

    def _partner_seen(self):
        """(time, (p, v, R)) behind the privileged vector: the newest
        delivered capture's partner truth (before any is delivered, the
        capture at t = 0), or the current truth without `privileged_delay`."""
        if not self.privileged_delay:
            return self.t, self._partner_now()
        t_capture, (_, partner) = self.frames.latest(self.t) or self.first_capture
        return t_capture, partner

    def _parts(self):
        m = self.truth
        _, (p, v, R) = self._partner_seen()
        return {
            "q": self.q, "qdot": self.qdot,
            "palm_p": m["palm_p"], "palm_n": m["palm_R"][:, 2], "palm_x": m["palm_R"][:, 0],
            "t": [self.t], "prev_action": self.prev_action,
            "partner_p": p, "partner_v": v, "partner_n": R[:, 2], "partner_x": R[:, 0],
        }

    @staticmethod
    def _vector(parts, layout, box):
        v = np.concatenate([np.asarray(parts[k], float).ravel() for k, _ in layout])
        return np.clip(v, box.low, box.high).astype(np.float32)

    def _depth_stack(self):
        shape = (self.camera_cfg.height, self.camera_cfg.width)
        frames = list(self.stack)
        pad = [np.zeros(shape, np.float32)] * (self.frame_stack - len(frames))
        return np.stack(pad + frames).astype(np.float32, copy=True)  # oldest first

    def _obs(self):
        parts = self._parts()
        space = self.observation_space
        if self.obs_mode == "pixels":
            return {"proprio": self._vector(parts, PROPRIO_LAYOUT, space["proprio"]),
                    "depth": self._depth_stack()}
        if self.obs_mode == "privileged":
            return {"state": self._vector(parts, PRIVILEGED_LAYOUT, space["state"])}
        p_est, v_est, _ = self.estimator.predict(0.0)
        parts.update(partner_p_est=p_est, partner_v_est=v_est,
                     partner_n_est=self.estimator.normal)
        obs = {"state": self._vector(parts, STATE_LAYOUT, space["state"])}
        if self.obs_mode == "state+depth":
            got = self.frames.latest(self.t)
            shape = (self.camera_cfg.height, self.camera_cfg.width)
            img = got[1][0] if got is not None else np.zeros(shape, np.float32)
            obs["depth"] = np.array(img, dtype=np.float32, copy=True)[None]
        return obs

    def _info(self):
        """Ground truth for reward design, distillation labels and logging
        (never a deployed policy's input)."""
        m, g = self.truth, self.monitor.last
        contact = self.event == "contact"
        return {
            "t": self.t,
            "event": self.event,  # None | "contact" | "collision"
            "contact": contact,
            "collision": self.event == "collision",
            "success": bool(contact and g["rot_err"] <= self.rot_tol),
            "palm_dist": float(np.linalg.norm(m["palm_p"] - m["partner_p"])),  # face centers
            "normal_gap": g["gap"],  # along the partner palm normal (< 0 = passed through)
            "in_plane_offset": g["in_plane"],
            "rot_err_deg": g["rot_err"],  # 0 = exactly antiparallel
            "angle_err_deg": g["angle_err"],
            "finger_err_deg": g["finger_err"],
            "clearance": m["clearance"],
            "palm_palm": m["palm_palm"],
            "arm_dist": m["arm_dist"],  # at this sample
            "partner_p": m["partner_p"].copy(),
            "partner_v": self.partner_v.copy(),
            "partner_n": m["partner_R"][:, 2].copy(),
            "privileged": self._vector(self._parts(), PRIVILEGED_LAYOUT, self._privileged_box),
            "privileged_t": float(self._partner_seen()[0]),  # when its partner part is from
            "t_meet": float(self.motion.t_meet),
            "partner_peak_speed": float(self.motion.peak_speed),
            "n_bad_actions": self.n_bad_actions,
            "scenario_attempts": int(self.scenario.attempts),
            "scenario_id": self.scenario_id,
        }
