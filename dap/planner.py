"""Our controller: kinematic trajectory optimization + replanning (proposal
section 4.3), and the contract `dap.trial.run_trial` drives.

Contract (also implemented by the scripted references in dap/baselines.py):

    controller.reset(model: ControllerModel, q0)
    q_cmd = controller.step(t, q, estimator)    # every dt

`step` sees the measured joint positions `q` of our arm at time t and the
partner estimate, and returns the joint positions it wants at t + dt. The
harness enforces the arm's limits: each joint moves at most v_max * dt per
step and stays within its position limits, so a command that asks for more
is clipped (no teleporting). A non-finite or wrongly shaped command holds the
arm still and is counted. The harness ends the trial at the first contact or
collision, or at the time limit, and scores it from ground truth; nothing the
controller reports is scored. Its compute time is measured by the harness.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ControllerModel:
    """What the controller may use: the robot model, its OWN diagram context
    (separate from the one the harness measures with, so planning can pose
    the arms freely), and our arm's limits. The partner in this context sits
    at its default configuration; where the partner is must come from the
    estimator."""

    diagram: object
    plant: object
    scene_graph: object
    context: object  # root diagram context, for scene_graph queries
    plant_context: object
    ours: object  # our arm's ModelInstanceIndex
    idx: np.ndarray  # our joints' indices in the plant's position vector
    q_lower: np.ndarray
    q_upper: np.ndarray
    v_max: np.ndarray
    dt: float


def controller_model(scene, dt):
    from dap.scene import OURS

    ctx, pc = scene.new_context()
    idx = scene.idx[OURS]
    return ControllerModel(
        diagram=scene.diagram, plant=scene.plant, scene_graph=scene.scene_graph,
        context=ctx, plant_context=pc, ours=scene.models[OURS], idx=idx,
        q_lower=scene.q_lower[idx].copy(), q_upper=scene.q_upper[idx].copy(),
        v_max=scene.v_max[idx].copy(), dt=dt,
    )


def plan_to_meeting(model: ControllerModel, q_now, qdot_now, meeting, t_now):
    """Return a joint-space trajectory that arrives at `meeting.X_WPalm` at
    `meeting.t_meet`. The arrival velocity is a planner design choice (e.g.
    at rest, or moving with the predicted partner velocity); success does
    not score it, the loss may (dap.scoring.LossWeights.w_speed).

    Constraints (KinematicTrajectoryOptimization):
      - MinimumDistanceLowerBoundConstraint vs. the predicted partner arm
      - joint position / velocity limits
      - start at (q_now, qdot_now); end at meeting.q_meet
    On failure: let the caller try another meeting point.
    """
    # TODO
    raise NotImplementedError


class ReplanningController:
    """Runs estimate -> meeting point -> plan at a fixed rate."""

    def __init__(self, replan_hz=10.0, replan=True):
        self.replan_hz = replan_hz
        self.replan = replan  # False = plan-once baseline

    def reset(self, model: ControllerModel, q0):
        """Start of a trial."""
        # TODO
        raise NotImplementedError

    def step(self, t, q, estimator):
        """Return the commanded joint positions (7,) for t + dt.

        Replans when `replan` is set and 1/replan_hz has passed since the last
        plan (plans once otherwise).
        """
        # TODO
        raise NotImplementedError
