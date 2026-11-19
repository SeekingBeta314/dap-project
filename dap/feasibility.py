"""Lightweight check that a dap is possible in a given scene.

The meeting type is fixed for now: palms parallel and facing each other, with
the fingers pointing in opposite directions (two flat hands pressed together
with one rotated 180 deg about the shared palm-plane axis). In palm frames
(+z = palm normal, +x = toward the fingers) that is

    R_{ours, partner} = RotY(pi):   x_partner = -x_ours, z_partner = -z_ours.

`find_meeting` solves ONE inverse-kinematics problem over both arms (14
joints) for a pair of configurations where the two palms meet like that,
nothing else collides, and our palm faces roughly toward the partner. It does
not have to pick a meeting pose first: the palm-to-palm constraint is
expressed between the two palm frames. A target point can optionally pin
where the meeting happens.

Approach after Vahrenkamp et al. (IROS 2009) dual-arm IK for re-grasping
and Bauml et al. (2010) catch-configuration NLPs; timings in docs/setup.md
(~3 ms median with SNOPT on Drake 1.58). Even the loose tier is within the
contact definition and the success tolerance (dap.scoring), so "feasible"
means a successful dap exists. Hand-hand pairs are filtered in the IK's own
context; each solution is then re-checked in the full scene, where only the
palm-palm pair may touch.
"""

import time
from dataclasses import dataclass

import numpy as np
from pydrake.all import InverseKinematics, RigidTransform, RotationMatrix, Solve

from dap.scene import OURS, P_PALM_FACE, PARTNER, Q_READY, Scene

# Relative rotation of the partner palm frame in our palm frame at the meeting.
R_OURS_PARTNER = RotationMatrix.MakeYRotation(np.pi)

# Min-jerk peak speed is 1.875 x average speed (Flash & Hogan 1985).
MIN_JERK_PEAK = 1.875


@dataclass(frozen=True)
class DapTolerance:
    in_plane: float = 0.02  # m, palm-center offset along the palm plane (per axis)
    gap: float = 0.01  # m, max gap between the palm faces (0 = touching)
    angle_deg: float = 10.0  # max rotation error vs. R_OURS_PARTNER
    clearance: float = 0.01  # m, min distance for every other geometry pair
    facing_cone_deg: float = 45.0  # our palm normal vs. direction to partner base


# A clean dap, tried first; its witness is what the partner aims for.
TIGHT = DapTolerance(in_plane=0.005, gap=0.002, angle_deg=2.0)
# The "low-quality" dap that still counts as possible (fallback).
LOOSE = DapTolerance()


@dataclass
class MeetingWitness:
    """One pair of configurations that achieves the dap."""

    q: np.ndarray  # full plant positions
    q_ours: np.ndarray  # (7,)
    q_partner: np.ndarray  # (7,)
    X_W_ours: RigidTransform  # our palm frame at the meeting
    X_W_partner: RigidTransform  # partner palm frame at the meeting
    p_contact: np.ndarray  # our palm face center (world)
    n_ours: np.ndarray  # our palm normal (world)
    solve_ms: float
    attempts: int
    quality: str = ""  # "tight" or "loose" (see check_feasible)


def base_distance_ok(layout, max_distance=1.85):
    """Cheap analytic gate. Two iiwa7s with these hands can meet palm to palm
    up to ~1.88 m base separation (measured, docs/setup.md)."""
    return float(np.hypot(layout.distance, layout.lateral)) <= max_distance


def approach_direction(layout):
    """Horizontal unit vector from our base toward the partner base."""
    d = np.array([layout.distance, layout.lateral, 0.0])
    return d / np.linalg.norm(d)


def find_meeting(scene: Scene, rng, target=None, target_box=0.03,
                 tol: DapTolerance = DapTolerance(), restarts=5, q_seed=None):
    """Return a `MeetingWitness`, or None if no attempt succeeded.

    `target` (3,) pins our palm face center to within `target_box` m (per
    axis) of a world point. Attempt 0 starts from `q_seed` (default: both
    arms at Q_READY); later attempts start from uniform random configurations.
    """
    plant = scene.plant
    # Hand-hand pairs are excluded here: the meeting puts the hands adjacent
    # by construction; the clearance constraint is for everything else.
    _, pc = scene.scratch_context(filter_hands=True)
    F_ours, F_partner = scene.palm_frames[OURS], scene.palm_frames[PARTNER]

    ik = InverseKinematics(plant, pc)
    q = ik.q()
    prog = ik.get_mutable_prog()

    # Partner palm origin in our palm frame: faces touch when it sits at
    # z = 2 * face offset (each palm face is 2 cm in front of its frame).
    face = P_PALM_FACE[2]
    lo = np.array([-tol.in_plane, -tol.in_plane, 2 * face])
    hi = np.array([tol.in_plane, tol.in_plane, 2 * face + tol.gap])
    ik.AddPositionConstraint(F_partner, np.zeros(3), F_ours, lo, hi)
    ik.AddOrientationConstraint(F_ours, R_OURS_PARTNER, F_partner, RotationMatrix(),
                                np.deg2rad(tol.angle_deg))
    ik.AddAngleBetweenVectorsConstraint(
        plant.world_frame(), approach_direction(scene.layout), F_ours, [0, 0, 1],
        0.0, np.deg2rad(tol.facing_cone_deg),
    )
    if target is not None:
        t = np.asarray(target, float)
        ik.AddPositionConstraint(F_ours, P_PALM_FACE, plant.world_frame(),
                                 t - target_box, t + target_box)
    ik.AddMinimumDistanceLowerBoundConstraint(tol.clearance)

    if q_seed is None:
        q_seed = scene.q()
        q_seed[scene.idx[OURS]] = Q_READY
        q_seed[scene.idx[PARTNER]] = Q_READY
    t0 = time.perf_counter()
    for k in range(restarts):
        guess = q_seed if k == 0 else rng.uniform(scene.q_lower, scene.q_upper)
        guess = np.array(guess, float)
        prog.SetInitialGuess(q, guess)
        result = Solve(prog)
        if result.is_success():
            qs = result.GetSolution(q)
            if not _hands_clear(scene, qs):
                continue  # the IK ignored hand-hand pairs; this meeting penetrates
            plant.SetPositions(pc, qs)
            X_o = plant.CalcRelativeTransform(pc, plant.world_frame(), F_ours)
            X_p = plant.CalcRelativeTransform(pc, plant.world_frame(), F_partner)
            return MeetingWitness(
                q=qs, q_ours=qs[scene.idx[OURS]].copy(),
                q_partner=qs[scene.idx[PARTNER]].copy(),
                X_W_ours=X_o, X_W_partner=X_p,
                p_contact=X_o @ P_PALM_FACE, n_ours=X_o.rotation().matrix()[:, 2],
                solve_ms=(time.perf_counter() - t0) * 1e3, attempts=k + 1,
            )
    return None


def _hands_clear(scene: Scene, q):
    """True if, at full positions `q`, no geometry pair penetrates in the
    scene itself (hand-hand pairs included; only the palm-palm contact pair
    is left to the scorer). The IK above excludes hand-hand pairs, so its
    solutions are re-checked here."""
    q_saved = scene.q()
    scene.set_q(q)
    try:
        return scene.clearance() >= 0.0
    finally:
        scene.set_q(q_saved)


def min_jerk_time(q_from, q_to, v_max):
    """Shortest duration (s) of a joint-space min-jerk move from q_from to
    q_to that keeps every joint under its velocity limit (Bauml et al. 2010
    style reachability-in-time check)."""
    dq = np.abs(np.asarray(q_to) - np.asarray(q_from))
    return float(np.max(MIN_JERK_PEAK * dq / np.asarray(v_max)))


def reachable_in_time(q_from, q_to, v_max, t_available, speed_margin=0.9):
    """True if a min-jerk move fits in `t_available` at `speed_margin` x the
    velocity limits."""
    return min_jerk_time(q_from, q_to, speed_margin * np.asarray(v_max)) <= t_available


def check_feasible(scene: Scene, rng, target=None, target_box=0.03, restarts=5,
                   tiers=(("tight", TIGHT), ("loose", LOOSE))):
    """Try a clean dap first, then a low-quality one. Returns the first
    witness found (with `quality` set to the tier name) or None."""
    total_ms, total_attempts = 0.0, 0
    for name, tol in tiers:
        w = find_meeting(scene, rng, target, target_box, tol, restarts)
        if w is not None:
            w.quality = name
            w.solve_ms += total_ms
            w.attempts += total_attempts
            return w
        total_attempts += restarts
    return None
