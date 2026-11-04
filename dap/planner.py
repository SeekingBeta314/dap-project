"""Kinematic trajectory optimization + replanning loop (proposal section 4.3)."""


def plan_to_meeting(plant, plant_context, q_now, qdot_now, meeting, checkpoints,
                    t_now, checkpoint_tol=0.02):
    """Return a joint-space trajectory that passes `checkpoints` in order and
    arrives at `meeting.X_WPalm` at `meeting.t_meet` with palm velocity
    matching `meeting.v_partner`.

    Constraints (KinematicTrajectoryOptimization):
      - PositionConstraint per checkpoint at a free normalized time, ordered
      - MinimumDistanceLowerBoundConstraint vs. the predicted partner arm
      - joint position / velocity limits
      - start at (q_now, qdot_now); end at meeting.q_meet with matched velocity
    On failure: loosen checkpoint_tol, then let the caller try another meeting point.
    """
    # TODO
    raise NotImplementedError


class ReplanningController:
    """Runs estimate -> meeting point -> plan at a fixed rate; hands off to
    the clasp controller once the palms are within `handoff_dist`."""

    def __init__(self, replan_hz=10.0, handoff_dist=0.04, replan=True):
        self.replan_hz = replan_hz
        self.handoff_dist = handoff_dist
        self.replan = replan  # False = plan-once baseline
        # TODO
