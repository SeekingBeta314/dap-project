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

from dataclasses import dataclass, replace

import numpy as np
from pydrake.all import (CollisionFilterDeclaration, GeometrySet,
                         KinematicTrajectoryOptimization,
                         MinimumDistanceLowerBoundConstraint, Solve)

from dap.meeting import select_meeting_point
from dap.scene import OURS, PARTNER, Scene
from dap.feasibility import check_feasible


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
    palm: object
    partner_palm: object
    partner_idx: np.ndarray
    meet_prior: np.ndarray
    q_guess: np.ndarray
    layout: object
    fingers: np.ndarray
    normal_prior: np.ndarray


def controller_model(scene, dt):
    from dap.scenario import nominal_meeting_point

    ctx, pc = scene.new_context()
    sg = scene.scene_graph.GetMyMutableContextFromRoot(ctx)
    scene.scene_graph.collision_filter_manager(sg).Apply(
        CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(scene.palm_geoms[OURS]), GeometrySet(scene.palm_geoms[PARTNER])))
    idx = scene.idx[OURS]
    return ControllerModel(
        diagram=scene.diagram, plant=scene.plant, scene_graph=scene.scene_graph,
        context=ctx, plant_context=pc, ours=scene.models[OURS], idx=idx,
        q_lower=scene.q_lower[idx].copy(), q_upper=scene.q_upper[idx].copy(),
        v_max=scene.v_max[idx].copy(), dt=dt, palm=scene.palm_frames[OURS],
        partner_palm=scene.palm_frames[PARTNER], partner_idx=scene.idx[PARTNER],
        meet_prior=nominal_meeting_point(scene.layout, 0.6),
        q_guess=scene.plant.GetPositions(pc).copy(),
        layout=scene.layout, fingers=np.array([0., 1., 0.]),
        normal_prior=np.array([-1., 0., 0.]),
    )


def plan_to_meeting(model, q_now, qdot_now, meeting, t_now):
    """Fit a cubic joint spline under joint limits and sampled clearance.

    The partner holds its estimated meeting configuration for collision
    checks. First solve the convex smooth-path problem. If that path fails
    the sampled distance check, add constraints at the offending knots,
    solve again, and verify the resulting path at every sample.
    """
    duration = meeting.t_meet - t_now
    plant = model.plant
    start = model.q_guess.copy()
    start[model.idx], start[model.partner_idx] = q_now, meeting.q_partner
    end = start.copy()
    end[model.idx] = meeting.q_meet
    lo, hi = plant.GetPositionLowerLimits().copy(), plant.GetPositionUpperLimits().copy()
    lo[model.partner_idx] = hi[model.partner_idx] = meeting.q_partner
    speed = plant.GetVelocityUpperLimits()
    opt = KinematicTrajectoryOptimization(len(start), 6, duration=duration)
    opt.AddDurationConstraint(duration, duration)
    opt.AddPositionBounds(lo, hi)
    opt.AddVelocityBounds(-speed, speed)
    opt.AddPathPositionConstraint(start, start, 0.0)
    opt.AddPathPositionConstraint(end, end, 1.0)
    v0 = np.zeros_like(start)
    v0[model.idx] = qdot_now * duration
    opt.AddPathVelocityConstraint(v0, v0, 0.0)
    zero = np.zeros_like(start)
    opt.AddPathVelocityConstraint(zero, zero, 1.0)
    opt.AddPathEnergyCost()

    sg = model.scene_graph.GetMyContextFromRoot(model.context)
    qs = model.scene_graph.get_query_output_port()
    phases = np.linspace(0, 1, 11)
    for attempt in range(2):
        result = Solve(opt.prog())
        if not result.is_success():
            return None
        path = opt.ReconstructTrajectory(result)
        bad = []
        for s in phases:
            plant.SetPositions(model.plant_context, path.value(s * duration).ravel())
            if qs.Eval(sg).ComputeSignedDistancePairwiseClosestPoints(0.0):
                bad.append(s)
        if not bad:
            return path
        if attempt == 0:
            dist = MinimumDistanceLowerBoundConstraint(plant, 0.001, model.plant_context)
            for s in bad:
                opt.AddPathPositionConstraint(dist, s)
            opt.SetInitialGuess(path)
    return None


class ReplanningController:
    """Choose a bounded motion from current observations and follow it."""

    def __init__(self, replan_hz=10.0, replan=True):
        if replan_hz <= 0:
            raise ValueError("replan_hz must be positive")
        self.replan_hz = replan_hz
        self.replan = replan

    def reset(self, model: ControllerModel, q0):
        # The pose sensor measures only the normal. Infer a roll prior by
        # finding one opposed-palm posture at the midpoint of the *known*
        # base layout. This solve uses a clean model and no trial witness.
        nominal = check_feasible(Scene(model.layout), np.random.default_rng(0),
                                 model.meet_prior, restarts=1)
        if nominal is not None:
            model.fingers = nominal.X_W_ours.rotation().matrix()[:, 0].copy()
            model.normal_prior = nominal.X_W_partner.rotation().matrix()[:, 2].copy()
            model.q_guess[model.partner_idx] = nominal.q_partner
        self.model = model
        self.path = None
        self.next_plan = 0.0
        self.last_q = np.array(q0, copy=True)

    def step(self, t, q, estimator):
        model = self.model
        if t >= self.next_plan - 1e-9:
            self.next_plan = t + 1 / self.replan_hz if self.replan else np.inf
            velocity = np.clip((q - self.last_q) / model.dt, -model.v_max, model.v_max)
            horizons = (0.18, 0.40, 0.75, 1.25, 2.25, 3.5)
            meeting = select_meeting_point(estimator, model, q, t, horizons)
            if meeting is not None:
                for horizon in horizons:
                    arrival = t + horizon
                    if arrival < meeting.t_meet:
                        continue
                    candidate = replace(meeting, t_meet=arrival,
                                        slack=meeting.slack + arrival - meeting.t_meet)
                    path = plan_to_meeting(model, q, velocity, candidate, t)
                    if path is not None:
                        self.path, self.plan_time = path, t
                        model.q_guess[model.partner_idx] = meeting.q_partner
                        break
        self.last_q = q.copy()
        if self.path is None:
            return q
        elapsed = min(t + model.dt - self.plan_time, self.path.end_time())
        return self.path.value(elapsed).ravel()[model.idx]
