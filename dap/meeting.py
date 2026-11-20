"""Estimate a reachable interception using the noisy palm face and normal."""

from dataclasses import dataclass

import numpy as np
from pydrake.all import InverseKinematics, RigidTransform, RotationMatrix, Solve

from dap.scene import P_PALM_FACE


@dataclass
class MeetingPoint:
    t_meet: float
    X_WPalm: RigidTransform
    v_partner: np.ndarray
    q_meet: np.ndarray
    slack: float
    q_partner: np.ndarray  # estimated by IK, never read from the trial


def target_palm_pose(p_partner, n_partner, standoff=0.0, fingers=(0.0, 0.0, -1.0)):
    """Target palm *frame* given the partner's observed *face center*.

    Roll is unobservable from a normal. Project the assumed finger direction
    onto the palm plane. Use the least aligned world axis at a singularity.
    """
    z = -np.asarray(n_partner, float)
    z /= np.linalg.norm(z)
    x = np.asarray(fingers, float) - np.dot(fingers, z) * z
    if np.linalg.norm(x) < 1e-6:
        axis = np.eye(3)[np.argmin(abs(z))]
        x = axis - axis.dot(z) * z
    x /= np.linalg.norm(x)
    R = RotationMatrix(np.column_stack((x, np.cross(z, x), z)))
    return RigidTransform(R, np.asarray(p_partner) - standoff * z - R @ P_PALM_FACE)


def predict_face(estimator, prior, horizon):
    """Short-horizon extrapolation blended with the public meeting-point prior.

    A long constant-acceleration extrapolation of a min-jerk reach is
    unreliable. The model trusts the midpoint prior when the hand is still
    far on the partner's side, then trusts measured position near contact.
    """
    horizon = min(horizon, 0.12)
    pos, vel, _ = estimator.predict(0.0)
    delta = vel * horizon
    # Bound extrapolation by the scenario's 0.9 m/s design speed.
    delta *= min(1.0, 0.9 * horizon / max(np.linalg.norm(delta), 1e-9))
    future = pos + delta
    remaining = (prior - future) @ estimator.normal
    blend = np.clip((remaining - 0.025) / 0.12, 0.0, 1.0)
    return (1 - blend) * future + blend * prior, vel, blend


def select_meeting_point(estimator, model, q_now, t_now, candidates):
    """Return the earliest candidate allowing a speed-limited approach.

    The partner's elbow configuration and wrist roll are unobserved. Fit a
    plausible pair of arm postures using the public geometry and estimated
    palm. Minimum distance excludes the intentional palm-palm contact only.
    """
    plant = model.plant
    world = plant.world_frame()
    p, velocity, blend = predict_face(estimator, model.meet_prior, max(candidates))
    normal = (1 - blend) * estimator.normal + blend * model.normal_prior
    normal /= max(np.linalg.norm(normal), 1e-10)
    X = target_palm_pose(p, normal, fingers=model.fingers)
    ik = InverseKinematics(plant, model.plant_context)
    prog = ik.get_mutable_prog()
    q = ik.q()
    for frame in (model.palm, model.partner_palm):
        ik.AddPositionConstraint(frame, P_PALM_FACE, world, p - 0.002, p + 0.002)
    ik.AddOrientationConstraint(model.palm, RotationMatrix(), world, X.rotation(),
                                np.deg2rad(3))
    ik.AddAngleBetweenVectorsConstraint(model.partner_palm, [0, 0, 1], world,
                                        normal, 0, np.deg2rad(4))
    ik.AddMinimumDistanceLowerBoundConstraint(0.001)
    seed = model.q_guess.copy()
    seed[model.idx] = q_now
    prog.AddQuadraticErrorCost(np.eye(len(seed)), seed, q)
    prog.SetInitialGuess(q, seed)
    result = Solve(prog)
    if not result.is_success():
        return None
    both = result.GetSolution(q)
    q_goal = both[model.idx]
    # Exact velocity feasibility is checked later on the spline.
    travel = 1.5 * np.max(abs(q_goal - q_now) / model.v_max)
    for horizon in candidates:
        if travel <= horizon:
            return MeetingPoint(t_now + horizon, X, velocity, q_goal, horizon - travel,
                                both[model.partner_idx])
    return None
