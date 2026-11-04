"""Pick a meeting time and our palm pose (proposal section 4.2)."""

from dataclasses import dataclass

import numpy as np
from pydrake.all import RigidTransform


@dataclass
class MeetingPoint:
    t_meet: float  # absolute time of contact
    X_WPalm: RigidTransform  # our palm pose at contact
    v_partner: np.ndarray  # predicted partner palm velocity at contact
    q_meet: np.ndarray  # IK solution for our arm
    slack: float  # spare time (s) beyond the minimum needed to get there


def target_palm_pose(p_partner, n_partner, standoff=0.02):
    """Our palm faces the partner palm, `standoff` m in front of it."""
    # TODO: pick the rotation about the palm normal (which way the fingers point).
    raise NotImplementedError


def select_meeting_point(estimator, plant, plant_context, t_now, candidates):
    """Sample candidate times, solve IK with a min-distance constraint against
    the partner arm, keep the feasible candidate with the most slack."""
    # TODO: InverseKinematics + AddMinimumDistanceLowerBoundConstraint;
    # reachability check from joint velocity limits.
    raise NotImplementedError
