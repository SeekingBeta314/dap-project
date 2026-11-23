"""The single definition of how a dap is judged. Used live (trial loop, RL
env) through `EventMonitor` and offline through `evaluate`, which share the
same per-sample rule, so the two can never disagree.

Everything here works from ground truth (palm poses from forward kinematics,
signed distances from the scene). Nothing a controller reports is used.

Geometry at one instant, with our palm frame R_o (columns: fingers x, y,
normal z), face center p_o, and the partner's R_p, p_p (n = partner normal,
which points toward us):
  gap       = (p_o - p_p) . n       our face center's signed distance in front
                                    of the partner face plane
  in_plane  = |offset - gap n|      center offset along the palm plane
  rot_err   = angle between R_o^T R_p and RotY(pi)   (0 = perfectly antiparallel:
                                    normals opposite AND fingers opposite)
  angle_err = angle between n_o and -n            (the "palms parallel" part)
  finger_err = angle between x_o and -x_p, both projected on the palm plane

Events at each sample:
  contact    the palm boxes touch: palm-palm signed distance <= touch_dist
             (2 mm), with the face centers overlapping (in_plane <= IN_PLANE),
             our face center no more than MAX_PENETRATION behind the partner's
             face plane, and on the previous sample our face in front of it
             (gap >= 0; the first sample uses its own gap), so the touch came
             from the front, not the back of the partner's hand.
  collision  (a) any geometry pair other than palm-palm has signed distance
             < 0 (hand-hand pairs included), or (b) the palm boxes overlap
             without a contact (edge-on, from behind, off to the side).
A contact takes precedence over a collision first seen on the SAME sample:
the touch came first within that step, and what penetrates afterwards (e.g.
the finger plates sliding into the other palm when a fast approach crosses
several millimetres in one 10 ms step) happens after the trial's end. A
collision on any earlier sample has already ended the trial.
The first event ends the trial. No event before the time limit = "timeout".

Success = the event is a contact and the palms are antiparallel
(rot_err <= tolerance). Everything else (position, timing, speed) is a
continuous score; `dap_loss` combines them. Tolerance and weights were
settled for Phase 2 (docs/setup.md section 1b) before any evaluation run.
"""

from dataclasses import dataclass

import numpy as np

# Palm-plane offset (m) up to which the two palm faces still overlap: the
# palms are 10 x 12 cm, so centers within half the short side overlap.
IN_PLANE = 0.05
# How far behind the partner face plane our face center may land in one
# sample and still count as the contact (m): more than a 10 ms step at any
# closing speed the arms can reach.
MAX_PENETRATION = 0.05

# Antiparallel tolerance on rot_err for success (deg). Ours: we found no
# source for a human dap's tolerance; settled for Phase 2 (docs/setup.md
# section 1b).
ROT_TOL_DEFAULT = 15.0

R_TARGET = np.array([[-1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])  # RotY(pi)


def _angle(c):
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def contact_geometry(p_o, R_o, p_p, R_p):
    """Vectorized: p (N, 3), R (N, 3, 3) -> dict of (N,) arrays (also works
    for a single sample with shapes (3,), (3, 3))."""
    single = np.ndim(p_o) == 1
    p_o, p_p = np.atleast_2d(p_o), np.atleast_2d(p_p)
    R_o = np.asarray(R_o).reshape(-1, 3, 3)
    R_p = np.asarray(R_p).reshape(-1, 3, 3)
    n = R_p[:, :, 2]
    off = p_o - p_p
    gap = np.einsum("ij,ij->i", off, n)
    in_plane = np.linalg.norm(off - gap[:, None] * n, axis=1)
    R_rel = np.einsum("nji,njk->nik", R_o, R_p)  # R_o^T R_p
    E = np.einsum("ji,njk->nik", R_TARGET, R_rel)  # R_target^T R_rel
    rot_err = _angle((np.trace(E, axis1=1, axis2=2) - 1.0) / 2.0)
    angle_err = _angle(-np.einsum("ij,ij->i", R_o[:, :, 2], n))
    a = R_o[:, :, 0] - np.einsum("ij,ij->i", R_o[:, :, 0], n)[:, None] * n
    b = -(R_p[:, :, 0] - np.einsum("ij,ij->i", R_p[:, :, 0], n)[:, None] * n)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        finger_err = _angle(np.einsum("ij,ij->i", a, b) / (na * nb))
    finger_err = np.where((na < 1e-9) | (nb < 1e-9), 90.0, finger_err)
    out = dict(gap=gap, in_plane=in_plane, rot_err=rot_err, angle_err=angle_err,
               finger_err=finger_err)
    if single:
        out = {k: float(v[0]) for k, v in out.items()}
    return out


def classify(gap, in_plane, clearance, palm_palm, prev_gap, touch_dist):
    """Per-sample event rule (vectorized). Returns an int array:
    0 = nothing, 1 = contact, 2 = collision."""
    gap, in_plane = np.asarray(gap, float), np.asarray(in_plane, float)
    clearance, palm_palm = np.asarray(clearance, float), np.asarray(palm_palm, float)
    prev_gap = np.asarray(prev_gap, float)
    contact = ((palm_palm <= touch_dist) & (gap >= -MAX_PENETRATION) & (in_plane <= IN_PLANE)
               & (prev_gap >= 0.0))
    collision = (clearance < 0.0) | (palm_palm < 0.0)
    out = np.zeros(gap.shape, dtype=int)
    out[collision] = 2
    out[contact] = 1  # the touch came first within the step
    return out


class EventMonitor:
    """Live version of the rule: call `step` once per sample; it returns
    None, "contact", or "collision" (and keeps the geometry of that sample
    in `self.last`)."""

    def __init__(self, touch_dist):
        self.touch_dist = touch_dist
        self.prev_gap = None  # first sample: judged by its own gap
        self.last = None

    def step(self, p_o, R_o, p_p, R_p, clearance, palm_palm):
        g = contact_geometry(p_o, R_o, p_p, R_p)
        prev = g["gap"] if self.prev_gap is None else self.prev_gap
        k = int(classify(g["gap"], g["in_plane"], clearance, palm_palm, prev,
                         self.touch_dist))
        self.prev_gap = g["gap"]
        self.last = g
        return {0: None, 1: "contact", 2: "collision"}[k]


@dataclass(frozen=True)
class Tolerances:
    touch_dist: float = 0.002  # m, palm-palm distance that counts as touching (Ours)
    rot_tol: float = ROT_TOL_DEFAULT  # deg, antiparallel tolerance on rot_err


def evaluate(a, t_meet_true, tol: Tolerances, v_max=None):
    """Score one logged trial. `a` is a dict of arrays with keys
    t (N,), q (N, 7), palm_p (N, 3), palm_R (N, 3, 3), partner_p, partner_R,
    partner_v (N, 3), clearance (N,), palm_palm (N,), arm_dist (N,).
    Returns a flat dict of metrics (NaN where undefined)."""
    t = np.asarray(a["t"], float)
    N = len(t)
    nan = float("nan")
    out = dict(outcome="timeout", success=False, contact=False, collision=False,
               t_event=nan, time_err=nan, gap=nan, pos_err=nan, rot_err=nan, angle_err=nan,
               finger_err=nan, rel_speed=nan, min_palm_dist=nan, closest_rot_err=nan,
               min_arm_dist=nan, vmax_frac=nan)
    if N == 0:
        return out
    g = contact_geometry(a["palm_p"], a["palm_R"], a["partner_p"], a["partner_R"])
    prev = np.concatenate([g["gap"][:1], g["gap"][:-1]])  # first sample: its own gap
    ev = classify(g["gap"], g["in_plane"], a["clearance"], a["palm_palm"], prev,
                  tol.touch_dist)
    hits = np.flatnonzero(ev)
    k = int(hits[0]) if hits.size else None
    last = N if k is None else k + 1

    d = np.linalg.norm(np.asarray(a["palm_p"])[:last] - np.asarray(a["partner_p"])[:last], axis=1)
    i = int(np.argmin(d))
    out["min_palm_dist"] = float(d[i])
    out["closest_rot_err"] = float(g["rot_err"][i])
    arm = np.asarray(a["arm_dist"], float)[:last]
    out["min_arm_dist"] = float(np.min(arm))
    if v_max is not None and last > 1:
        qdot = np.abs(np.diff(np.asarray(a["q"], float)[:last], axis=0)) / np.diff(t[:last])[:, None]
        out["vmax_frac"] = float(np.mean(np.any(qdot >= 0.95 * np.asarray(v_max), axis=1)))

    if k is None:
        return out
    out["t_event"] = float(t[k])
    if ev[k] == 2:
        out["outcome"], out["collision"] = "collision", True
        return out
    out["outcome"], out["contact"] = "contact", True
    out["time_err"] = float(t[k] - t_meet_true)
    for key in ("gap", "rot_err", "angle_err", "finger_err"):
        out[key] = float(g[key][k])
    out["pos_err"] = float(g["in_plane"][k])
    if k > 0:
        v_o = (np.asarray(a["palm_p"][k]) - np.asarray(a["palm_p"][k - 1])) / (t[k] - t[k - 1])
        out["rel_speed"] = float(np.linalg.norm(v_o - np.asarray(a["partner_v"][k])))
    out["success"] = bool(out["rot_err"] <= tol.rot_tol)
    return out


@dataclass(frozen=True)
class LossWeights:
    """Weights and scales (Ours) for `dap_loss`, settled for Phase 2
    (docs/setup.md section 1b). Timing is unweighted: success does not
    require it and no comparator chooses a meeting time, so `time_err` is
    reported as its own score instead.

    The loss is tiered, so outcomes keep a fixed order whatever the errors
    (every success < every failed contact < every miss < every collision):
      success         [0, tier)          e = sum of w * (error / scale)^2,
                                         mapped smoothly by tier * e / (e + 1)
      failed contact  [tier, 2 tier)     same e (rotation error beyond rot_tol)
      timeout         [2 tier, 3 tier]   grows with the closest face-center
                                         distance, saturating at miss_dist_scale
      collision       collision_penalty  (> 3 tier)
    """

    pos_scale: float = 0.03  # m
    rot_scale: float = 15.0  # deg
    time_scale: float = 0.1  # s
    rel_speed_scale: float = 0.5  # m/s
    w_pos: float = 1.0
    w_rot: float = 1.0
    w_time: float = 0.0  # timing: reported, unweighted (see above)
    w_speed: float = 0.0  # relative speed at contact: reported, unweighted for now
    tier: float = 5.0  # width of each outcome tier
    miss_dist_scale: float = 0.3  # m, closest-approach distance at which a miss saturates
    collision_penalty: float = 20.0


def dap_loss(r, w: LossWeights = LossWeights()):
    """Vectorized loss over trial results: `r` is a dict of arrays or a
    pandas DataFrame with the columns `evaluate` produces (crashed trials,
    outcome "error", get NaN). Lower is better; see LossWeights."""
    if not 3 * w.tier < w.collision_penalty:
        raise ValueError("collision_penalty must exceed every other tier")

    def col(k):
        return np.asarray(r[k], float)

    outcome = np.asarray(r["outcome"])
    success = np.asarray(r["success"], dtype=bool)
    with np.errstate(invalid="ignore"):
        e = (w.w_pos * (col("pos_err") / w.pos_scale) ** 2
             + w.w_rot * (col("rot_err") / w.rot_scale) ** 2
             + w.w_time * (col("time_err") / w.time_scale) ** 2)
        if w.w_speed:
            e = e + w.w_speed * (np.nan_to_num(col("rel_speed")) / w.rel_speed_scale) ** 2
        within = w.tier * e / (e + 1.0)  # in [0, tier), increasing in e
        lc = np.where(success, within, w.tier + within)
        d = np.clip(col("min_palm_dist") / w.miss_dist_scale, 0.0, 1.0)
        lm = 2 * w.tier + w.tier * d**2
    return np.select(
        [outcome == "contact", outcome == "timeout", outcome == "collision"],
        [lc, lm, np.full(lc.shape, w.collision_penalty)],
        default=np.nan,
    )
