"""The controller contract `dap.trial.run_trial` drives (implemented by the
baselines and references in dap/baselines.py):

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

