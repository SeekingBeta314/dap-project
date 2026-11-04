"""One trial: sample a partner approach, run our controller, score the dap.

The loop is kinematic: our arm is wherever the controller commands, and the
partner palm is wherever PartnerPath says. Swapping in a full Drake
simulation later only changes `run_trial`; `score` works off the log.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from dap.checkpoints import sample_checkpoints
from dap.estimator import PalmEstimator
from dap.experiment import Condition, ExperimentConfig, trial_rngs
from dap.metrics import TrialResult, palm_angle_error_deg
from dap.partner import NoisyPoseSensor, PartnerPath, sample_params
from dap.planner import ReplanningController
from dap.scene import Q_READY, build_scene

# Palm face center in the palm frame (the palm box is 2 cm thick; see dap_hand.sdf).
P_PALM_FACE = np.array([0.0, 0.0, 0.02])


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
    phase: list = field(default_factory=list)

    def append(self, **row):
        for k, v in row.items():
            getattr(self, k).append(v)

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


class Scene:
    """Built once per worker process and reused across trials."""

    def __init__(self):
        self.diagram, self.plant, self.scene_graph = build_scene()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        self.iiwa = self.plant.GetModelInstanceByName("iiwa")
        self.hand = self.plant.GetModelInstanceByName("iiwa_hand")
        self.palm_frame = self.plant.GetFrameByName("palm", self.hand)

    def set_ours(self, q, hinge=0.0):
        self.plant.SetPositions(self.plant_context, self.iiwa, q)
        self.plant.SetPositions(self.plant_context, self.hand, [hinge])

    def palm(self):
        X = self.plant.CalcRelativeTransform(
            self.plant_context, self.plant.world_frame(), self.palm_frame
        )
        return X @ P_PALM_FACE, X.rotation().matrix()[:, 2]


def run_trial(cfg: ExperimentConfig, cond: Condition, seed: int, scene=None):
    scene = scene or Scene()
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

    path = PartnerPath(sample_params(rngs["partner"], cond.speed_scale, cond.noise_scale))
    sensor = NoisyPoseSensor(path, rngs["sensor"])
    estimator = PalmEstimator(pos_noise=path.params.noise_pos)

    scene.set_ours(Q_READY)
    p0, _ = scene.palm()
    checkpoints = sample_checkpoints(
        rngs["checkpoints"], cond.n_checkpoints, p0, path.params.meet
    )

    controller = ReplanningController(
        replan_hz=cfg.replan_hz,
        handoff_dist=cfg.handoff_dist,
        replan=(cond.planner == "replan"),
        velocity_match=cond.velocity_match,
    )
    controller.reset(scene.plant, scene.plant_context, Q_READY.copy(), checkpoints)

    # TODO: drive the partner arm along `path` so min_arm_dist and collisions
    # can be measured (see dap/partner.py).

    log = TrialLog()
    t_end = path.params.duration + cfg.t_extra
    for t in np.arange(0.0, t_end, cfg.dt):
        estimator.update(t, *sensor.measure(t))
        q, _ = controller.step(t, estimator)
        scene.set_ours(q, controller.hinge)
        palm_p, palm_n = scene.palm()
        log.append(
            t=t, q=q, palm_p=palm_p, palm_n=palm_n,
            partner_p=path.p(t), partner_n=path.normal(t), partner_v=path.v(t),
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

    # Checkpoints, in order: closest approach after the previous one was hit.
    start = 0
    for c in checkpoints:
        d = np.linalg.norm(a["palm_p"][start:] - c, axis=1)
        i = int(np.argmin(d))
        result.checkpoint_errs.append(float(d[i]))
        start += i

    # Contact = first time the palm faces come within contact_dist.
    gap = np.linalg.norm(a["palm_p"] - a["partner_p"], axis=1)
    hits = np.flatnonzero(gap <= cfg.contact_dist)
    if hits.size == 0:
        return result
    k = int(hits[0])
    result.contact = True
    result.t_contact = float(t[k])
    result.time_err = result.t_contact - result.t_meet_pred

    # Position error in the partner's palm plane; angle between the normals.
    n_partner = a["partner_n"][k]
    offset = a["palm_p"][k] - a["partner_p"][k]
    result.pos_err = float(np.linalg.norm(offset - np.dot(offset, n_partner) * n_partner))
    result.angle_err = palm_angle_error_deg(a["palm_n"][k], n_partner)

    # Velocity mismatch over the window before contact (finite-difference ours).
    v_ours = np.gradient(a["palm_p"], t, axis=0)
    w = (t >= t[k] - cfg.vel_window) & (t <= t[k])
    result.vel_mismatch = float(
        np.mean(np.linalg.norm(v_ours[w] - a["partner_v"][w], axis=1))
    )

    # TODO: also require "no other collision" once the partner arm is driven.
    result.success = (
        result.pos_err <= cfg.success_pos_tol
        and result.angle_err <= cfg.success_angle_tol
        and result.clasp_held
    )
    return result
