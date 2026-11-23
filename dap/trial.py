"""One trial: sample a scenario, run a controller, score the dap.

The loop is kinematic and causal. At each step k (time t = k dt):
  1. the controller sees our measured joint positions q(t) and the partner
     estimate (stand-in sensor at time t), plus any head-camera depth frame
     delivered by t if the trial has a camera, and returns a command for
     t + dt;
  2. the harness moves our arm toward it, at most v_max * dt per joint and
     within the joint position limits (so the arm cannot teleport), and
     moves the partner arm along its sampled joint trajectory to t + dt;
  3. the ground truth at t + dt is logged and checked by the same event rule
     the scorer uses (dap.scoring.EventMonitor).
The trial ends at the first contact or collision, or `t_extra` s after the
partner's motion ends. It is scored afterwards from the log alone
(dap.scoring.evaluate); the controller's own claims play no part.
"""

import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from dap.camera import FrameBuffer, sensed_depth
from dap.depth_noise import DepthNoiseModel
from dap.estimator import PalmEstimator
from dap.experiment import Condition, ExperimentConfig
from dap.metrics import TrialResult
from dap.partner import NOISE_ANG, NOISE_POS, NoisyPoseSensor
from dap.planner import controller_model
from dap.scenario import SCENARIO_PRESETS, sample_scenario
from dap.scene import OURS
from dap.scoring import EventMonitor, evaluate


@dataclass
class TrialLog:
    """Ground-truth history, one row per step (including t = 0)."""

    t: list = field(default_factory=list)
    q: list = field(default_factory=list)
    palm_p: list = field(default_factory=list)  # our palm face center
    palm_R: list = field(default_factory=list)  # our palm frame (x fingers, z normal)
    partner_p: list = field(default_factory=list)
    partner_R: list = field(default_factory=list)
    partner_v: list = field(default_factory=list)  # partner palm face velocity
    clearance: list = field(default_factory=list)  # every pair but palm-palm
    palm_palm: list = field(default_factory=list)  # palm box vs. palm box
    arm_dist: list = field(default_factory=list)  # ours vs. partner, palm pair excluded

    def append(self, **row):
        for k, v in row.items():
            getattr(self, k).append(v)

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


def record_scenario(result: TrialResult, sc):
    """Scenario covariates every trial row carries (also dap.policy_eval)."""
    motion = sc.partner
    result.scenario_attempts = sc.attempts
    result.target_peak_speed = sc.target_peak_speed
    result.partner_peak_speed = motion.peak_speed
    result.partner_speed_at_meet = motion.speed_at_meet
    result.t_meet_true = motion.t_meet
    result.reachable_in_time = bool(sc.reachable_in_time)
    result.partner_joint_speed_ratio = sc.partner_joint_speed_ratio
    for k in ("distance", "lateral", "yaw_deg", "height"):
        setattr(result, f"layout_{k}", getattr(sc.layout, k))


def score_log(result: TrialResult, log: TrialLog, motion, tolerances, v_max, live_event):
    """Score a ground-truth log into `result` (dap.scoring.evaluate) and
    check that the live event rule and the offline scorer agree."""
    for key, val in evaluate(log.arrays(), motion.t_meet, tolerances, v_max).items():
        setattr(result, key, val)
    if result.outcome != (live_event or "timeout"):
        raise RuntimeError(f"live event {live_event!r} != scored outcome {result.outcome!r}")


def record_crash(result: TrialResult, what, wall0):
    """A crash inside the controller (or policy) is a failure of the method,
    recorded as a trial outcome with every scenario covariate kept. Call
    from the `except` block."""
    result.outcome, result.error_kind = "error", "controller"
    result.error = f"{what}:\n{traceback.format_exc(limit=5)}"
    result.wall_time_s = time.perf_counter() - wall0


def record_timing(result: TrialResult, step_ms, wall0):
    """Harness timing: the compute time of each controller (or policy) call,
    and the trial's wall time."""
    result.n_steps = len(step_ms)
    if step_ms:
        result.step_ms_mean = float(np.mean(step_ms))
        result.step_ms_p95 = float(np.percentile(step_ms, 95))
        result.step_ms_max = float(np.max(step_ms))
    result.wall_time_s = time.perf_counter() - wall0


def horizon_steps(t_end, t_extra, dt):
    """Number of control steps in a trial (and an RL episode): the last
    sample is at horizon_steps * dt >= t_end + t_extra."""
    return int(np.ceil((t_end + t_extra) / dt - 1e-9))


def limit_command(q_cmd, q, v_max, dt, q_lo, q_hi):
    """Where the arm actually goes: at most v_max * dt per joint toward the
    command, within the position limits. Returns (q_next, ok); a
    non-finite or wrongly shaped command holds the arm (ok = False)."""
    q_cmd = np.asarray(q_cmd, dtype=float) if q_cmd is not None else None
    if q_cmd is None or q_cmd.shape != q.shape or not np.all(np.isfinite(q_cmd)):
        return q.copy(), False
    step = v_max * dt
    return np.clip(np.clip(q_cmd, q - step, q + step), q_lo, q_hi), True


def run_trial(cfg: ExperimentConfig, cond: Condition, seed: int, camera=None,
              *, controller_factory, return_log=False):
    """`controller_factory(cfg, cond, scenario)` builds the controller; it
    gets the scenario so reference baselines (e.g. the oracle) can use
    ground truth. A real controller must not.

    With `camera` (a dap.camera.HeadCameraConfig), the controller also gets
    the head camera's depth frames, rendered as dap.envs.DapEnv renders them:
    captured every 1/fps s with D435-style noise (the env's default `med`
    preset, the trial's camera stream) and delivered `latency_frames` later,
    through
    `controller.observe(t_capture, depth, q_capture)`, where q_capture is
    our joint positions at capture."""
    rngs = cfg.rngs(seed)
    result = TrialResult(seed=seed, planner=cond.planner, noise_scale=cond.noise_scale)
    wall0 = time.perf_counter()

    sc = sample_scenario(rngs["scenario"], SCENARIO_PRESETS[cfg.scenario], camera=camera)
    scene, motion = sc.scene, sc.partner
    record_scenario(result, sc)

    # Stand-in for the camera pipeline (docs/perception_pipeline.md).
    sensor = NoisyPoseSensor(motion, rngs["sensor"], NOISE_POS * cond.noise_scale,
                             NOISE_ANG * cond.noise_scale)
    estimator = PalmEstimator(pos_noise=sensor.noise_pos)

    frames = None
    if camera is not None:
        noise = DepthNoiseModel.for_camera(camera, "med")
        noise.reset(rngs["camera"], (camera.height, camera.width))
        frames, last_capture = FrameBuffer(camera.fps, camera.latency_frames), None

    idx = scene.idx[OURS]
    v_max, q_lo, q_hi = scene.v_max[idx], scene.q_lower[idx], scene.q_upper[idx]
    def controller_crashed(what):
        record_crash(result, what, wall0)
        return (result, log) if return_log else result

    log = TrialLog()
    try:
        controller = controller_factory(cfg, cond, sc)
        controller.reset(controller_model(scene, cfg.dt), sc.q_ours0.copy())
    except Exception:
        return controller_crashed("controller setup failed")

    monitor = EventMonitor(cfg.tolerances.touch_dist)
    q = sc.q_ours0.copy()

    def record(t):
        scene.set_ours(q)
        scene.set_partner(motion.q(t))
        m = scene.measure()
        log.append(t=t, q=q.copy(), partner_v=motion.v(t), **m)
        return monitor.step(m["palm_p"], m["palm_R"], m["partner_p"], m["partner_R"],
                            m["clearance"], m["palm_palm"])

    step_ms = []
    n_steps = horizon_steps(motion.t_end, cfg.t_extra, cfg.dt)
    event = record(0.0)
    for k in range(n_steps):
        if event is not None:
            break
        t = k * cfg.dt
        estimator.update(t, *sensor.measure(t))
        new_frame = None
        if frames is not None:
            if frames.due(t):
                frames.push(t, (sensed_depth(scene.render_depth(), noise, camera.z_far), q.copy()))
            got = frames.latest(t)
            if got is not None and got[0] != last_capture:
                last_capture, new_frame = got[0], got
        c0 = time.perf_counter()
        try:
            if new_frame is not None:
                controller.observe(new_frame[0], *new_frame[1])
            q_cmd = controller.step(t, q.copy(), estimator)
        except Exception:
            return controller_crashed(f"controller failed at t={t:.2f}")
        step_ms.append(1e3 * (time.perf_counter() - c0))
        q, ok = limit_command(q_cmd, q, v_max, cfg.dt, q_lo, q_hi)
        result.n_bad_commands += int(not ok)
        event = record((k + 1) * cfg.dt)

    score_log(result, log, motion, cfg.tolerances, v_max, event)
    record_timing(result, step_ms, wall0)
    return (result, log) if return_log else result
