"""Per-trial results (proposal section 5)."""

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class TrialResult:
    seed: int
    planner: str  # "once" | "replan"
    speed_scale: float
    noise_scale: float
    n_checkpoints: int
    velocity_match: bool = True

    success: bool = False
    contact: bool = False  # palms touched at all (success also needs alignment + clasp)
    clasp_held: bool = False
    t_contact: float = np.nan  # s, sim time of first contact
    t_meet_pred: float = np.nan  # s, controller's last predicted meeting time
    pos_err: float = np.nan  # m, palm-center distance at contact
    time_err: float = np.nan  # s, actual minus predicted contact time
    angle_err: float = np.nan  # deg, angle between palm normals (anti-parallel = 0)
    checkpoint_errs: list = field(default_factory=list)  # m, per checkpoint
    min_arm_dist: float = np.nan  # m, closest our-arm/hand vs. partner-arm/hand distance
    collision: bool = False  # any unfiltered geometry pair penetrated (hand-hand excluded)
    vel_mismatch: float = np.nan  # m/s, mean over the last 0.2 s before contact
    plan_times_ms: list = field(default_factory=list)
    n_replans: int = 0
    n_infeasible: int = 0  # plans that failed even after loosening checkpoints
    t_meet_true: float = np.nan  # s, when the partner palm reaches its meeting pose
    scenario_attempts: int = 0  # scenario samples drawn before one passed every check
    wall_time_s: float = np.nan
    error: str = ""  # exception text if the trial crashed

    def to_dict(self):
        d = asdict(self)
        # NaN and inf are not valid JSON (e.g. min_arm_dist is inf when the
        # arms never came within the distance query's range).
        return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in d.items()}


# Palm-plane offset (m) up to which the two palm faces still overlap: the
# palms are 10 x 12 cm, so centers within half the short side overlap.
CONTACT_IN_PLANE = 0.05
# Our face may be at most this far behind the partner face plane (m) and
# still count as contact (it passed through between two samples).
CONTACT_MAX_PENETRATION = 0.05


def palm_contact(p_ours, p_partner, n_partner, contact_dist):
    """Palm-to-palm contact from face centers and the partner palm normal
    (which points toward us).

    Returns (contact, normal_gap, in_plane): `normal_gap` is our face center's
    signed distance in front of the partner face plane (negative = passed
    through), `in_plane` the center offset along the plane. Contact means the
    gap closed to `contact_dist` or below (penetration included, so a fast
    approach cannot step over it between samples) while the faces overlap.
    Hand-hand collisions are filtered in the scene, so this is the only
    contact signal."""
    offset = np.asarray(p_ours) - np.asarray(p_partner)
    n = np.asarray(n_partner)
    gap = float(np.dot(offset, n))
    in_plane = float(np.linalg.norm(offset - gap * n))
    contact = (-CONTACT_MAX_PENETRATION <= gap <= contact_dist) and in_plane <= CONTACT_IN_PLANE
    return contact, gap, in_plane


def palm_angle_error_deg(n_ours, n_partner):
    """0 when the palms face each other exactly."""
    c = np.clip(np.dot(n_ours, -n_partner), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))
