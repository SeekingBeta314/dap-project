"""One trial: sample a scenario, run our controller, score the dap.

The loop is kinematic: our arm is wherever the controller commands, and the
partner arm follows its sampled joint trajectory. Swapping in a full Drake
simulation later only changes `run_trial`; `score` works off the log.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from dap.checkpoints import sample_checkpoints
from dap.estimator import PalmEstimator
from dap.experiment import Condition, ExperimentConfig, trial_rngs
from dap.metrics import TrialResult, palm_angle_error_deg, palm_contact
from dap.partner import NOISE_ANG, NOISE_POS, NoisyPoseSensor
from dap.planner import ReplanningController
from dap.scenario import SCENARIO_PRESETS, sample_scenario


@dataclass
class TrialLog:
    """Ground-truth history, one row per control step."""

    t: list = field(default_factory=list)
    q: list = field(default_factory=list)
    palm_p: list = field(default_factory=list)  # our palm face center
    palm_n: list = field(default_factory=list)  # our palm normal
    partner_p: list = field(default_factory=list)
    partner_n: list = field(default_factory=list)
    partner_v: list = field(default_factory=list)
    arm_dist: list = field(default_factory=list)  # our arm/hand vs. partner arm/hand
    clearance: list = field(default_factory=list)  # min over all unfiltered pairs
    phase: list = field(default_factory=list)

    def append(self, **row):
        for k, v in row.items():
            getattr(self, k).append(v)

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


def run_trial(cfg: ExperimentConfig, cond: Condition, seed: int, camera=None):
    rngs = trial_rngs(seed)
    result = TrialResult(
        seed=seed,
        planner=cond.planner,
        speed_scale=cond.speed_scale,
        noise_scale=cond.noise_scale,
        n_checkpoints=cond.n_checkpoints,
        velocity_match=cond.velocity_match,
    )
    wall0 = time.perf_counter()

    sc = sample_scenario(rngs["scenario"], SCENARIO_PRESETS[cfg.scenario], cond.speed_scale,
                         camera=camera)
    scene, motion = sc.scene, sc.partner
    result.scenario_attempts = sc.attempts
    result.t_meet_true = motion.t_meet

    # Stand-in for the camera pipeline (docs/perception_pipeline.md).
    sensor = NoisyPoseSensor(motion, rngs["sensor"], NOISE_POS * cond.noise_scale,
                             NOISE_ANG * cond.noise_scale)
    estimator = PalmEstimator(pos_noise=sensor.noise_pos)

    scene.set_ours(sc.q_ours0)
    scene.set_partner(motion.q(0.0))
    p0, _ = scene.palm()
    # Checkpoints run toward the nominal meeting point (known from the
    # layout), not the true one, so they leak nothing about the partner path.
    checkpoints = sample_checkpoints(rngs["checkpoints"], cond.n_checkpoints, p0, sc.meet_prior)

    controller = ReplanningController(
        replan_hz=cfg.replan_hz,
        handoff_dist=cfg.handoff_dist,
        replan=(cond.planner == "replan"),
        velocity_match=cond.velocity_match,
    )
    controller.reset(scene.plant, scene.plant_context, sc.q_ours0.copy(), checkpoints)

    log = TrialLog()
    t_end = motion.t_end + cfg.t_extra
    for t in np.arange(0.0, t_end, cfg.dt):
        scene.set_partner(motion.q(t))
        estimator.update(t, *sensor.measure(t))
        q, _ = controller.step(t, estimator)
        scene.set_ours(q, controller.hinge)
        palm_p, palm_n = scene.palm()
        log.append(
            t=t, q=q, palm_p=palm_p, palm_n=palm_n,
            partner_p=motion.p(t), partner_n=motion.normal(t), partner_v=motion.v(t),
            arm_dist=scene.min_arm_distance(), clearance=scene.clearance(),
            phase=controller.phase,
        )
        if controller.phase == "done":
            break

    score(cfg, log, checkpoints, controller, result)
    result.wall_time_s = time.perf_counter() - wall0
    return result


def score(cfg, log: TrialLog, checkpoints, controller, result: TrialResult):
    """Fill `result` from the ground-truth log. Pure function of its inputs."""
    a = log.arrays()
    t = a["t"]

    result.plan_times_ms = list(controller.plan_times_ms)
    result.n_replans = controller.n_replans
    result.n_infeasible = controller.n_infeasible
    result.t_meet_pred = controller.t_meet_pred
    result.clasp_held = bool(controller.clasp_held)

    if a["arm_dist"].size:
        result.min_arm_dist = float(np.min(a["arm_dist"]))
    if a["clearance"].size:
        result.collision = bool(np.min(a["clearance"]) < 0.0)

    # Checkpoints, in order: closest approach after the previous one was hit.
    start = 0
    for c in checkpoints:
        d = np.linalg.norm(a["palm_p"][start:] - c, axis=1)
        i = int(np.argmin(d))
        result.checkpoint_errs.append(float(d[i]))
        start += i

    # Contact = first sample where the palm faces close along the partner's
    # palm normal while overlapping in the palm plane (metrics.palm_contact).
    k = None
    for i in range(len(t)):
        hit, _, in_plane = palm_contact(a["palm_p"][i], a["partner_p"][i], a["partner_n"][i],
                                        cfg.contact_dist)
        if hit:
            k = i
            break
    if k is None:
        return result
    result.contact = True
    result.t_contact = float(t[k])
    result.time_err = result.t_contact - result.t_meet_pred

    # Position error in the partner's palm plane; angle between the normals.
    n_partner = a["partner_n"][k]
    result.pos_err = in_plane
    result.angle_err = palm_angle_error_deg(a["palm_n"][k], n_partner)

    # Velocity mismatch over the window before contact (finite-difference ours).
    v_ours = np.gradient(a["palm_p"], t, axis=0)
    w = (t >= t[k] - cfg.vel_window) & (t <= t[k])
    result.vel_mismatch = float(
        np.mean(np.linalg.norm(v_ours[w] - a["partner_v"][w], axis=1))
    )

    result.success = (
        result.pos_err <= cfg.success_pos_tol
        and result.angle_err <= cfg.success_angle_tol
        and result.clasp_held
        and not result.collision
    )
    return result
