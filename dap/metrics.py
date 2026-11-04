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
    min_arm_dist: float = np.nan  # m
    vel_mismatch: float = np.nan  # m/s, mean over the last 0.2 s before contact
    plan_times_ms: list = field(default_factory=list)
    n_replans: int = 0
    n_infeasible: int = 0  # plans that failed even after loosening checkpoints
    wall_time_s: float = np.nan
    error: str = ""  # exception text if the trial crashed

    def to_dict(self):
        d = asdict(self)
        # NaN is not valid JSON.
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def palm_angle_error_deg(n_ours, n_partner):
    """0 when the palms face each other exactly."""
    c = np.clip(np.dot(n_ours, -n_partner), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))
