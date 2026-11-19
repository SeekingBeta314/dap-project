"""Single-arm inverse kinematics helper shared by the partner path tracker
and the scenario sampler."""

import numpy as np
from pydrake.all import InverseKinematics, RotationMatrix, Solve

from dap.scene import Scene


def arm_ik(scene: Scene, name, X_target, q_nominal, q_seed, pos_tol, angle_tol_deg,
           bounds=None, min_distance=None, q_full=None, cost_weight=1.0):
    """IK for one arm's palm frame (`name` = OURS or PARTNER). Every other
    joint is held at `q_full` (default: the scene's current positions).

    Minimizes ||q_arm - q_nominal||^2 subject to the palm frame being within
    `pos_tol` m / `angle_tol_deg` of `X_target`, optional per-joint `bounds`
    (lo, hi), and an optional minimum distance over all unfiltered collision
    pairs. Returns the arm's (7,) configuration or None.
    """
    plant = scene.plant
    ctx = scene.diagram.CreateDefaultContext()
    pc = plant.GetMyMutableContextFromRoot(ctx)
    ik = InverseKinematics(plant, pc)
    q = ik.q()
    prog = ik.get_mutable_prog()
    q_now = scene.q() if q_full is None else np.asarray(q_full, float)
    arm = scene.idx[name]
    arm_set = set(int(i) for i in arm)
    for i in range(len(q_now)):
        if i not in arm_set:
            prog.AddBoundingBoxConstraint(q_now[i], q_now[i], q[i])
    F = scene.palm_frames[name]
    p = X_target.translation()
    ik.AddPositionConstraint(F, np.zeros(3), plant.world_frame(), p - pos_tol, p + pos_tol)
    ik.AddOrientationConstraint(F, RotationMatrix(), plant.world_frame(), X_target.rotation(),
                                np.deg2rad(angle_tol_deg))
    if min_distance is not None:
        ik.AddMinimumDistanceLowerBoundConstraint(min_distance)
    qa = q[arm]
    if bounds is not None:
        prog.AddBoundingBoxConstraint(bounds[0], bounds[1], qa)
    prog.AddQuadraticErrorCost(cost_weight * np.eye(7), np.asarray(q_nominal, float), qa)
    guess = q_now.copy()
    guess[arm] = q_seed
    prog.SetInitialGuess(q, guess)
    result = Solve(prog)
    if not result.is_success():
        return None
    return result.GetSolution(qa)
