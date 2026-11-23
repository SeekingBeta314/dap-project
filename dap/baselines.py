"""Baselines and references. All implement the controller contract in
dap/planner.py and are scored by dap.trial.run_trial like anything else.

    run_trial(cfg, cond, seed, controller_factory=servo_factory)

| name       | sees the partner through        | role                                   |
| ---------- | ------------------------------- | -------------------------------------- |
| hold       | nothing                         | floor: never moves                     |
| blind      | nothing (a pose fit offline)    | does the task need perception at all?  |
| servo      | the stand-in pose sensor        | classical reactive baseline            |
| servo_depth| head-camera depth (perception)  | the servo on the student's input       |
| servo_gt   | ground truth (current state)    | reference: the servo with perfect state|
| servo_gt_cam | ground truth at camera timing | reference: servo_depth with perfect    |
|            |                                 | perception (the teacher's counterpart) |
| oracle     | ground truth incl. the future   | reference: knows the meeting in advance|

The oracle, servo_gt and servo_gt_cam read the scenario, so they are
references, not methods. servo_depth and servo_gt_cam need the head camera:

    run_trial(cfg, cond, seed, camera=SERVO_DEPTH_CAMERA, controller_factory=servo_depth_factory)
"""

from functools import lru_cache

import numpy as np
from pydrake.all import JacobianWrtVariable, RotationMatrix

from dap.camera import HeadCameraConfig
from dap.estimator import PalmEstimator
from dap.partner import min_jerk
from dap.scene import OURS, P_PALM_FACE

ROT_Y_PI = RotationMatrix.MakeYRotation(np.pi).matrix()  # ours = partner @ ROT_Y_PI


class OracleController:
    """Knows the scenario: moves our arm with a joint-space min-jerk profile
    from its start to `scenario.q_ours_meet` (our side of the certified dap),
    arriving at the partner's true meeting time, then holds. Not
    collision-checked along the way, and its commands are clipped to the
    joint speed limits like anyone's (so it can arrive late when the dap is
    not reachable in time). An upper reference, not a planner."""

    def __init__(self, scenario):
        self.q_meet = np.asarray(scenario.q_ours_meet, float)
        self.t_meet = scenario.partner.t_meet

    def reset(self, model, q0):
        self.q0 = np.asarray(q0, float).copy()
        self.dt = model.dt

    def step(self, t, q, estimator):
        s = min_jerk((t + self.dt) / self.t_meet)
        return self.q0 + (self.q_meet - self.q0) * s


class HoldController:
    """Never moves. The floor: scenarios are sampled so the partner never
    touches a resting arm, so holding still always times out."""

    def reset(self, model, q0):
        self.q0 = np.asarray(q0, float).copy()

    def step(self, t, q, estimator):
        return self.q0.copy()


class DiffIKServo:
    """Reactive differential-IK controller (resolved-rate control with damped
    least squares) that drives our palm face toward a target pose every step.

    `partner_fn(t, estimator)` gives the partner palm face's predicted
    position, its velocity and its frame (columns: fingers, y, normal toward
    us). Our
    target is the antiparallel pose, R_ours = R_partner RotY(pi):
      - with `standoff`: first `standoff` m in front of the partner palm along
        its normal; once there (within `align_pos` m and `align_deg`), `close`
        m in front, i.e. behind its face, which drives the palms together
        along the normal. The closing phase latches (switching back whenever
        the alignment check fails makes the arm chatter at the boundary);
        its target still tracks the partner.
      - with `latch=False` (the ServoExpert, dap.teacher): no latch; it
        closes whenever our palm face is within `align_pos` of the partner's
        normal line in-plane, within `align_deg`, and no further out than
        `standoff + align_pos`. Advancing along the normal keeps that true,
        so it does not chatter, and the command is a function of the state
        alone (which imitation needs). Ours.
      - with `standoff=None`: exactly at the partner palm face (used by the
        blind baseline to sit at a fixed pose).
    The partner's velocity is fed forward. Joint velocities are scaled down
    together so no joint exceeds its limit (the harness clips anyway). No
    collision avoidance beyond approaching along the normal.
    """

    def __init__(self, partner_fn, standoff=0.05, close=-0.02, align_pos=0.02, align_deg=10.0,
                 gain=10.0, damping=0.05, latch=True):
        self.partner_fn = partner_fn
        self.latch = latch
        self.standoff, self.close = standoff, close
        self.align_pos, self.align_deg = align_pos, align_deg
        self.gain, self.damping = gain, damping

    def reset(self, model, q0):
        plant = model.plant
        if plant.num_positions() != plant.num_velocities():
            raise ValueError("expects q and v to share indices (revolute joints only)")
        self.m = model
        self.frame = plant.GetFrameByName("palm", plant.GetModelInstanceByName(f"{OURS}_hand"))
        self.q_full = plant.GetPositions(model.plant_context).copy()
        self.closing = False

    def step(self, t, q, estimator):
        p_p, v_p, R_p = self.partner_fn(t, estimator)
        m, plant, pc = self.m, self.m.plant, self.m.plant_context
        self.q_full[m.idx] = q
        plant.SetPositions(pc, self.q_full)
        X = plant.CalcRelativeTransform(pc, plant.world_frame(), self.frame)
        p_o, R_o = X @ P_PALM_FACE, X.rotation().matrix()

        aa = RotationMatrix(R_p @ ROT_Y_PI @ R_o.T).ToAngleAxis()  # our palm -> target
        n = R_p[:, 2]
        if self.standoff is None:
            p_t = p_p
        else:
            p_far = p_p + self.standoff * n
            if not self.latch:
                d = p_o - p_p
                self.closing = (np.linalg.norm(d - (d @ n) * n) < self.align_pos
                                and np.degrees(aa.angle()) < self.align_deg
                                and d @ n < self.standoff + self.align_pos)
            elif not self.closing:
                self.closing = (np.linalg.norm(p_far - p_o) < self.align_pos
                                and np.degrees(aa.angle()) < self.align_deg)
            p_t = p_p + self.close * n if self.closing else p_far

        V = np.concatenate([self.gain * aa.angle() * aa.axis(), self.gain * (p_t - p_o) + v_p])
        J = plant.CalcJacobianSpatialVelocity(pc, JacobianWrtVariable.kV, self.frame, P_PALM_FACE,
                                              plant.world_frame(), plant.world_frame())[:, m.idx]
        qd = J.T @ np.linalg.solve(J @ J.T + self.damping**2 * np.eye(6), V)
        qd /= max(1.0, float(np.max(np.abs(qd) / m.v_max)))
        return q + qd * m.dt


class DepthServo(DiffIKServo):
    """The servo on the partner palm perceived from head-camera depth
    (dap.perception.DepthPalmTracker), whose filter is updated at each
    frame's capture time. Holds still until the first palm measurement.
    Near the meeting our own hand hides the partner's palm from the head
    camera for a few hundred ms, and extrapolating the filter that long runs
    away, so the target is predicted at most `coast` s past the last
    measurement and then held (the partner's reach ends at rest)."""

    def __init__(self, tracker, coast=0.1, **kw):
        super().__init__(self._partner, **kw)
        self.tracker, self.coast = tracker, coast

    def _partner(self, t, _):
        e = self.tracker.estimator
        age = t - e.t
        p, v, _ = e.predict(min(age, self.coast) + LOOKAHEAD)
        return p, (v if age <= self.coast else np.zeros(3)), e.rotation()

    def observe(self, t_capture, depth, q_capture):
        self.tracker.observe(t_capture, depth, q_capture)

    def step(self, t, q, estimator):
        if self.tracker.estimator.x is None:
            return q.copy()
        return super().step(t, q, estimator)


class TruthAtCapture:
    """Stands in for dap.perception.DepthPalmTracker with perfect perception
    (reference only): at each delivered frame's capture time it feeds the
    partner palm's true pose into the same filter servo_depth uses, and
    ignores the image. So it has the camera's timing (30 fps, one frame
    late) and the filter's velocity estimate, but no detection errors and
    no occlusion. The teacher instead gets the true velocity at capture."""

    def __init__(self, motion):
        from dap.perception import NORMAL_ALPHA, POS_NOISE

        self.motion = motion
        self.estimator = PalmEstimator(pos_noise=POS_NOISE, normal_alpha=NORMAL_ALPHA)

    def observe(self, t, depth, q_ours):
        R = self.motion.rotation(t)
        self.estimator.update(t, self.motion.p(t), R[:, 2], R[:, 0])


# -- what each servo sees ----------------------------------------------------

LOOKAHEAD = 0.1  # s, aim where the partner palm will be one servo time constant ahead


def estimator_partner(t, estimator, lookahead=LOOKAHEAD):
    """From the stand-in sensor's filter (what a real controller may use;
    the harness updates it before every step)."""
    p, v, _ = estimator.predict(lookahead)
    return p, v, estimator.rotation()


def truth_partner(scenario, lookahead=LOOKAHEAD):
    """The same, from the true current state (reference only): current
    pose and velocity, extrapolated like the filter's prediction."""
    motion = scenario.partner

    def partner_fn(t, estimator):
        v = motion.v(t)
        return motion.p(t) + lookahead * v, v, motion.rotation(t)

    return partner_fn


# -- blind ---------------------------------------------------------------------

def approach_frame(layout):
    """World rotation with x horizontal toward the partner base and z up."""
    from dap.feasibility import approach_direction

    x = approach_direction(layout)
    z = np.array([0.0, 0.0, 1.0])
    return np.column_stack([x, np.cross(z, x), z])


@lru_cache(maxsize=None)
def blind_prior(preset, n=30):
    """The best constant guess of where our palm should meet the partner,
    fit on TRAINING scenarios (dap.experiment.train_rngs, disjoint from the
    evaluation seeds): the mean offset of our meeting palm face from the
    nominal meeting point and the chordal mean of its orientation, both in
    the approach frame. Returns (offset (3,), R (3, 3))."""
    from dap.experiment import train_rngs
    from dap.scenario import SCENARIO_PRESETS, ScenarioError, sample_scenario

    offs, Rs = [], []
    for seed in range(n):
        try:
            sc = sample_scenario(train_rngs(seed)["scenario"], SCENARIO_PRESETS[preset])
        except ScenarioError:
            continue
        A = approach_frame(sc.layout)
        offs.append(A.T @ (sc.witness.p_contact - sc.meet_prior))
        Rs.append(A.T @ sc.witness.X_W_ours.rotation().matrix())
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return np.mean(offs, axis=0), R


def blind_target(scenario, preset):
    """Fixed palm target from what the controller may know without looking:
    the nominal meeting point and the partner base's direction, plus the
    training-set prior."""
    off, R_rel = blind_prior(preset)
    A = approach_frame(scenario.layout)
    p, R = scenario.meet_prior + A @ off, A @ R_rel

    def partner_fn(t, estimator):  # a "partner" whose antiparallel pose is the target
        return p, np.zeros(3), R @ ROT_Y_PI

    return partner_fn


# -- factories (scripts/run_experiment.py --controller <name>) ---------------

def oracle_factory(cfg, cond, scenario):
    return OracleController(scenario)


def hold_factory(cfg, cond, scenario):
    return HoldController()


def blind_factory(cfg, cond, scenario):
    return DiffIKServo(blind_target(scenario, cfg.scenario), standoff=None)


def servo_factory(cfg, cond, scenario):
    return DiffIKServo(estimator_partner)


def servo_gt_factory(cfg, cond, scenario):
    return DiffIKServo(truth_partner(scenario))


# The student's camera: dap.envs.DapEnv renders the head camera at camera_scale=0.25.
SERVO_DEPTH_CAMERA = HeadCameraConfig().scaled(0.25)


def servo_depth_factory(cfg, cond, scenario):
    """Uses the scenario's layout (where the partner stands, as blind does)
    and the head camera's configuration, nothing else."""
    from dap.perception import DepthPalmTracker

    camera = scenario.scene.camera_cfg
    if camera is None:
        raise ValueError("servo_depth needs the head camera: run_trial(..., camera=...)")
    return DepthServo(DepthPalmTracker(scenario.layout, camera))


def servo_gt_cam_factory(cfg, cond, scenario):
    """servo_depth's servo on the true palm pose at the camera's capture
    times (reference only; reads the scenario)."""
    if scenario.scene.camera_cfg is None:
        raise ValueError("servo_gt_cam needs the head camera: run_trial(..., camera=...)")
    return DepthServo(TruthAtCapture(scenario.partner))


# Baselines that take the head camera's frames (run_trial(camera=SERVO_DEPTH_CAMERA)).
CAMERA_BASELINES = ("servo_depth", "servo_gt_cam")
