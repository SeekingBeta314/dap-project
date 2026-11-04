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
    the clasp controller once the palms are within `handoff_dist`.

    This is the contract `dap.trial.run_trial` drives. The trial loop is
    kinematic for now: whatever (q, qdot) `step` returns is where our arm is.

    Attributes the trial loop reads:
      phase          "approach" | "clasp" | "done"; "done" ends the trial
      hinge          commanded finger-hinge angle (rad), 0 = open
      clasp_held     set by the clasp phase once the plate has closed on the partner hand
      t_meet_pred    latest predicted meeting time (s)
      plan_times_ms  wall time of each plan_to_meeting call
      n_replans, n_infeasible
    """

    def __init__(self, replan_hz=10.0, handoff_dist=0.04, replan=True,
                 velocity_match=True):
        self.replan_hz = replan_hz
        self.handoff_dist = handoff_dist
        self.replan = replan  # False = plan-once baseline
        self.velocity_match = velocity_match  # False = arrive with zero velocity

        self.phase = "approach"
        self.hinge = 0.0
        self.clasp_held = False
        self.t_meet_pred = float("nan")
        self.plan_times_ms = []
        self.n_replans = 0
        self.n_infeasible = 0

    def reset(self, plant, plant_context, q0, checkpoints, t0=0.0):
        """Start of trial. `checkpoints` is (n, 3), to be hit in order."""
        # TODO
        raise NotImplementedError

    def step(self, t, estimator):
        """Advance to time t; return our arm's commanded (q, qdot), each (7,).

        Replans when `replan` is set and 1/replan_hz has passed since the last
        plan (plan once otherwise), switches to the clasp controller when the
        estimated palm gap drops below handoff_dist.
        """
        # TODO
        raise NotImplementedError
