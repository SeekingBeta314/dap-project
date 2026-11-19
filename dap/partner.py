"""The partner arm: a randomized, human-like palm approach that the partner
iiwa actually executes, plus the (stand-in) noisy pose sensor.

Path model (sources in docs/setup.md, "Partner motion"):
  - Geometry: a cubic Bezier from the start palm position to the meeting
    point, with a random lateral bulge (GenH2R-style random key point), that
    arrives moving along the partner palm normal, then continues straight
    for a short overshoot. The overshoot means the hand is still moving at
    contact (people dap *through* the contact), which is what makes
    velocity matching meaningful.
  - Timing: minimum-jerk along arc length (Flash & Hogan 1985), scaled so
    the meeting point is reached at t_meet ~ U[0.8, 2.0] s (handovers take
    ~1.2 s; reaches 0.6-2 s; Chan et al. 2021, Nemlekar et al. 2019,
    Controzzi et al. 2018).
  - Optional mid-course target switch: a second min-jerk displacement is
    superimposed (Flash & Henis 1991), finishing before contact.
  - Orientation: slerp from a start orientation tilted up to 25 deg away
    from the meeting orientation, on the same min-jerk clock.
  - The partner iiwa tracks the palm path by IK at 20 Hz knots, each knot
    staying as close as possible to the previous one (minimum joint motion,
    like differential IK) and within the joint velocity limits.
    The resulting joint trajectory is the ground truth: palm poses, and the
    arm the head camera sees, come from its forward kinematics.

Only `NoisyPoseSensor` output may be used by our controller; `PartnerMotion`
is ground truth (for the partner arm, rendering, and scoring).
"""

from dataclasses import dataclass

import numpy as np
from pydrake.all import AngleAxis, PiecewisePolynomial, Quaternion, RigidTransform, RotationMatrix

from dap.ik import arm_ik
from dap.scene import OURS, P_PALM_FACE, PARTNER, Q_READY, Scene

# Stand-in sensor noise until the camera perception pipeline exists
# (docs/perception_pipeline.md). Scaled by ExperimentConfig noise_scale.
NOISE_POS = 0.005  # m
NOISE_ANG = np.deg2rad(2.0)  # rad


@dataclass(frozen=True)
class PathConfig:
    t_meet: tuple = (0.8, 2.0)  # s, start -> contact (divided by speed_scale)
    start_back: tuple = (0.1, 0.3)  # m, start offset away from us (toward partner base)
    start_lateral: float = 0.25  # m, +- sideways
    start_vertical: float = 0.2  # m, +- up/down
    bulge_sigma: tuple = (0.03, 0.08)  # m, std range of the Bezier bulge
    approach_frac: tuple = (0.25, 0.5)  # final Bezier tangent length / chord
    overshoot: tuple = (0.03, 0.10)  # m, travel past the contact point
    tilt_max_deg: float = 25.0  # start orientation vs. meeting orientation
    switch_prob: float = 0.0  # chance of a mid-course target switch
    switch_dist: tuple = (0.05, 0.15)  # m
    switch_start: tuple = (0.3, 0.6)  # fraction of t_meet
    switch_dur: tuple = (0.4, 0.8)  # s (capped to finish before contact)
    max_palm_speed: float = 1.0  # m/s peak (multiplied by speed_scale)
    knot_dt: float = 0.05  # s, IK tracking knots (20 Hz)
    dense_dt: float = 0.01  # s, ground-truth sampling
    track_pos_tol: float = 0.001  # m (per axis)
    track_angle_tol_deg: float = 1.0
    speed_margin: float = 0.95  # partner joint speeds <= margin * limits
    clearance: float = 0.005  # m, partner arm vs. itself / floor / pedestals
    start_tries: int = 5  # start poses to try before rejecting the sample


def min_jerk(tau):
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def min_jerk_dot(tau):
    """d/dtau of min_jerk."""
    tau = np.clip(tau, 0.0, 1.0)
    return 30 * tau**2 - 60 * tau**3 + 30 * tau**4


_TAU = np.linspace(0.0, 1.0, 2001)
_S = min_jerk(_TAU)


def min_jerk_inv(s):
    return float(np.interp(s, _S, _TAU))


def _quat_slerp(q0: Quaternion, q1: Quaternion, a):
    v0, v1 = q0.wxyz(), q1.wxyz()
    d = float(np.dot(v0, v1))
    if d < 0:
        v1, d = -v1, -d
    if d > 0.9995:
        v = v0 + a * (v1 - v0)
    else:
        th = np.arccos(d)
        v = (np.sin((1 - a) * th) * v0 + np.sin(a * th) * v1) / np.sin(th)
    return Quaternion(v / np.linalg.norm(v))


def _random_rotation(rng, max_deg):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return RotationMatrix(AngleAxis(np.deg2rad(rng.uniform(0, max_deg)), axis))


class PalmPathReference:
    """Reference pose of the partner palm FRAME over time (not the face)."""

    def __init__(self, p_start, X_meet: RigidTransform, R_start: RotationMatrix,
                 t_meet, bulge, approach_len, overshoot, switch=None):
        self.t_meet = t_meet
        self.X_meet = X_meet
        p_meet = X_meet.translation()
        n = X_meet.rotation().matrix()[:, 2]  # partner palm normal: points at us
        self.switch = switch  # (delta (3,), t_start, duration) or None
        delta = switch[0] if switch else np.zeros(3)
        p_aim = p_meet - delta  # where the unswitched path arrives

        P0, P3 = np.asarray(p_start, float), p_aim
        P1 = P0 + (P3 - P0) / 3 + bulge
        P2 = P3 - n * approach_len
        u = np.linspace(0, 1, 400)[:, None]
        bez = ((1 - u) ** 3 * P0 + 3 * (1 - u) ** 2 * u * P1
               + 3 * (1 - u) * u**2 * P2 + u**3 * P3)
        line = P3 + np.linspace(0, overshoot, 40)[1:, None] * n
        pts = np.vstack([bez, line])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.arc = np.concatenate([[0.0], np.cumsum(seg)])
        self.pts = pts
        self.L_meet = self.arc[len(bez) - 1]
        self.L_total = self.arc[-1]
        # Total min-jerk duration so that arc length L_meet is reached at t_meet.
        tau_meet = min_jerk_inv(self.L_meet / self.L_total)
        self.T_total = t_meet / tau_meet
        self.q_start = R_start.ToQuaternion()
        self.q_meet = X_meet.rotation().ToQuaternion()

    @property
    def t_end(self):
        return self.T_total

    def position(self, t):
        s = self.L_total * min_jerk(t / self.T_total)
        p = np.array([np.interp(s, self.arc, self.pts[:, i]) for i in range(3)])
        if self.switch is not None:
            delta, ts, dur = self.switch
            p = p + delta * min_jerk((t - ts) / dur)
        return p

    def rotation(self, t):
        a = min_jerk(t / self.t_meet)
        return RotationMatrix(_quat_slerp(self.q_start, self.q_meet, float(a)))

    def pose(self, t):
        return RigidTransform(self.rotation(t), self.position(t))

    def peak_speed(self, dt=0.005):
        ts = np.arange(0.0, self.T_total + dt, dt)
        P = np.array([self.position(t) for t in ts])
        return float(np.max(np.linalg.norm(np.diff(P, axis=0), axis=1)) / dt)


class PartnerMotion:
    """Ground-truth partner motion: a joint trajectory for the partner arm
    and its palm face pose sampled every `dense_dt`."""

    def __init__(self, scene: Scene, q_traj: PiecewisePolynomial, reference: PalmPathReference,
                 dense_dt):
        self.q_traj = q_traj
        self.reference = reference
        self.t_meet = reference.t_meet
        self.t_end = q_traj.end_time()
        self.times = np.arange(0.0, self.t_end + dense_dt / 2, dense_dt)
        q_saved = scene.q()
        P, N, R = [], [], []
        for t in self.times:
            scene.set_partner(self.q(t))
            X = scene.palm_pose(PARTNER)
            P.append(X @ P_PALM_FACE)
            Rm = X.rotation().matrix()
            R.append(Rm)
            N.append(Rm[:, 2])
        scene.set_q(q_saved)
        self.P = np.array(P)
        self.N = np.array(N)
        self.Rs = np.array(R)
        self.V = np.gradient(self.P, self.times, axis=0)
        self.p_meet = self.p(self.t_meet)

    def q(self, t):
        return self.q_traj.value(float(np.clip(t, 0.0, self.t_end))).ravel()

    def _interp(self, arr, t):
        t = float(np.clip(t, 0.0, self.t_end))
        return np.array([np.interp(t, self.times, arr[:, i]) for i in range(arr.shape[1])])

    def p(self, t):
        """Palm face center (world)."""
        return self._interp(self.P, t)

    def v(self, t):
        if t < 0 or t > self.t_end:
            return np.zeros(3)
        return self._interp(self.V, t)

    def normal(self, t):
        n = self._interp(self.N, t)
        return n / np.linalg.norm(n)

    def rotation(self, t):
        k = int(np.clip(round(t / (self.times[1] - self.times[0])), 0, len(self.times) - 1))
        return self.Rs[k]


def sample_start_noise(rng, kind, scale, size=7):
    if kind == "normal":
        return rng.normal(0.0, scale, size)
    if kind == "uniform":
        return rng.uniform(-scale, scale, size)
    raise ValueError(kind)


def make_partner_motion(scene: Scene, X_meet: RigidTransform, rng,
                        cfg: PathConfig = PathConfig(), speed_scale=1.0,
                        start_noise=("uniform", 0.3), margin_deg=5.0):
    """Sample a partner approach that reaches palm pose `X_meet` (partner
    palm frame) at t_meet. Returns
    (PartnerMotion, None) or (None, reason) when the sample is rejected."""
    layout = scene.layout
    p_meet = X_meet.translation()
    n = X_meet.rotation().matrix()[:, 2]

    # --- sample the reference path -------------------------------------
    base = np.array([layout.distance, layout.lateral, p_meet[2]])
    back = base - p_meet
    back[2] = 0.0
    back /= np.linalg.norm(back)
    lateral = np.cross([0, 0, 1.0], back)
    p_start = (p_meet + back * rng.uniform(*cfg.start_back)
               + lateral * rng.uniform(-cfg.start_lateral, cfg.start_lateral)
               + np.array([0, 0, 1.0]) * rng.uniform(-cfg.start_vertical, cfg.start_vertical))
    t_meet = rng.uniform(*cfg.t_meet) / speed_scale
    bulge = rng.normal(0.0, rng.uniform(*cfg.bulge_sigma), 3)
    chord = np.linalg.norm(p_meet - p_start)
    approach_len = rng.uniform(*cfg.approach_frac) * chord
    overshoot = rng.uniform(*cfg.overshoot)
    R_start = X_meet.rotation() @ _random_rotation(rng, cfg.tilt_max_deg)
    switch = None
    if rng.random() < cfg.switch_prob:
        d = rng.normal(size=3)
        d -= np.dot(d, n) * n  # move the meeting point within the palm plane
        d *= rng.uniform(*cfg.switch_dist) / np.linalg.norm(d)
        ts = rng.uniform(*cfg.switch_start) * t_meet
        dur = min(rng.uniform(*cfg.switch_dur), t_meet - ts)
        switch = (d, ts, dur)
    ref = PalmPathReference(p_start, X_meet, R_start, t_meet, bulge, approach_len,
                            overshoot, switch)
    if ref.peak_speed() > cfg.max_palm_speed * speed_scale:
        return None, "palm_too_fast"

    # --- partner start configuration ------------------------------------
    # The start palm pose is fixed by the path; the start joint configuration
    # is randomized through the IK's nominal posture (nullspace).
    lo = scene.q_lower[scene.idx[PARTNER]] + np.deg2rad(margin_deg)
    hi = scene.q_upper[scene.idx[PARTNER]] - np.deg2rad(margin_deg)
    q0 = None
    for _ in range(cfg.start_tries):
        q_nom = np.clip(Q_READY + sample_start_noise(rng, *start_noise), lo, hi)
        q0 = arm_ik(scene, PARTNER, ref.pose(0.0), q_nom, q_nom, cfg.track_pos_tol,
                         cfg.track_angle_tol_deg, bounds=(lo, hi))
        if q0 is not None:
            break
    if q0 is None:
        return None, "partner_start_ik"

    # --- track the path ---------------------------------------------------
    # Each knot may move every joint at most speed_margin * v_max * dt from
    # the previous knot, so IK can't flip elbow branches and a failure means
    # the partner arm genuinely can't follow the path at this speed.
    v_lim = cfg.speed_margin * scene.v_max[scene.idx[PARTNER]]
    grid = np.arange(0.0, ref.t_end, cfg.knot_dt)
    # Drop grid knots crowding t_meet / t_end; near-duplicate knots make the
    # spline ring.
    grid = grid[(np.abs(grid - ref.t_meet) > cfg.knot_dt / 2)
                & (np.abs(grid - ref.t_end) > cfg.knot_dt / 2)]
    times = np.unique(np.concatenate([grid, [ref.t_meet, ref.t_end]]))
    Q = [q0]
    q_prev = q0
    for t_prev, t in zip(times[:-1], times[1:]):
        step = v_lim * (t - t_prev)
        bounds = (np.maximum(lo, q_prev - step), np.minimum(hi, q_prev + step))
        q_k = arm_ik(scene, PARTNER, ref.pose(t), q_prev, q_prev, cfg.track_pos_tol,
                          cfg.track_angle_tol_deg, bounds=bounds)
        if q_k is None:
            return None, "partner_track_ik"
        Q.append(q_k)
        q_prev = q_k
    Q = np.array(Q)
    q_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
        times, Q.T, np.zeros(7), np.zeros(7))
    qd = q_traj.derivative(1)
    dense = np.arange(0.0, q_traj.end_time(), cfg.dense_dt)
    qdot = np.abs(np.array([qd.value(t).ravel() for t in dense]))
    if np.any(qdot > scene.v_max[scene.idx[PARTNER]]):
        return None, "partner_joint_speed"

    # --- partner arm must not hit itself / the floor / the pedestals ----
    # Checked on the dense trajectory (the one the camera renders), not just
    # at the IK knots.
    q_saved = scene.q()
    for t in dense:
        scene.set_partner(q_traj.value(t).ravel())
        if scene.clearance(ignore=OURS) < cfg.clearance:
            scene.set_q(q_saved)
            return None, "partner_collision"
    scene.set_q(q_saved)
    return PartnerMotion(scene, q_traj, ref, cfg.dense_dt), None


class NoisyPoseSensor:
    """Stand-in for the camera pipeline: noisy palm position and normal of
    the partner (ground truth + Gaussian noise)."""

    def __init__(self, motion: PartnerMotion, rng, noise_pos=NOISE_POS, noise_ang=NOISE_ANG):
        self.motion = motion
        self.rng = rng
        self.noise_pos = noise_pos
        self.noise_ang = noise_ang

    def measure(self, t):
        p = self.motion.p(t) + self.rng.normal(0, self.noise_pos, 3)
        n = self.motion.normal(t) + self.rng.normal(0, self.noise_ang, 3)
        return p, n / np.linalg.norm(n)
