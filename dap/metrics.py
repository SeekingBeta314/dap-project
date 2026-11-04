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

    success: bool = False
    pos_err: float = np.nan  # m, palm-center distance at contact
    time_err: float = np.nan  # s, actual minus predicted contact time
    angle_err: float = np.nan  # deg, angle between palm normals (anti-parallel = 0)
    checkpoint_errs: list = field(default_factory=list)  # m, per checkpoint
    min_arm_dist: float = np.nan  # m
    vel_mismatch: float = np.nan  # m/s, mean over the last 0.2 s before contact
    plan_times_ms: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def palm_angle_error_deg(n_ours, n_partner):
    """0 when the palms face each other exactly."""
    c = np.clip(np.dot(n_ours, -n_partner), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))
