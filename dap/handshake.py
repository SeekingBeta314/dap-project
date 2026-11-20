"""Kinematic slap, clasp, pull, release and knuckle bump by two right hands.

The wrists follow authored Cartesian key poses, solved with the iiwa models.
The fingers flex against signed-distance geometry at the clasp. This scene has
its own arm/hand models; the first-contact study has a different state space.
"""

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    Cylinder,
    DiagramBuilder,
    InverseKinematics,
    JacobianWrtVariable,
    MakeRenderEngineVtk,
    PiecewisePolynomial,
    RigidTransform,
    RotationMatrix,
    Solve,
)

from dap.camera import add_head_camera
from dap.hand_timing import reach_progress, release_progress
from dap.scene import (
    IIWA_URL,
    OURS,
    PARTNER,
    Q_READY,
    SceneLayout,
    _add_static_geometry,
    make_parser,
)

DIGITS = ("index", "middle", "ring", "little", "thumb")
SIDES = (OURS, PARTNER)
MEETING = np.array([0.67, 0.0, 0.58])
LAYOUT = SceneLayout(distance=1.34)


def _digit_joints(digit):
    suffixes = ("cmc", "mcp", "ip") if digit == "thumb" else ("mcp", "pip", "dip")
    return [digit + "_" + suffix for suffix in suffixes]


class HandshakeScene:
    """Two articulated hands replaying a joint-limit-checked Cartesian choreography."""

    def __init__(self, camera=None, renderer=None):
        builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=0.001
        )
        p = self.plant
        parser = make_parser(p)
        self.models, self.hands, self.palm_frames = {}, {}, {}
        self.owner = {}
        for side in SIDES:
            arm = parser.AddModels(url=IIWA_URL)[0]
            p.RenameModelInstance(arm, side)
            hand = parser.AddModels(url="package://dap/articulated_hand.sdf")[0]
            p.RenameModelInstance(hand, side + "_hand")
            X_base = RigidTransform() if side == OURS else LAYOUT.partner_base()
            p.WeldFrames(p.world_frame(), p.GetFrameByName("iiwa_link_0", arm), X_base)
            flange = p.GetBodyByName("iiwa_link_7", arm)
            # The hand adapter fills the gap between the arm's tool flange and wrist.
            adapter_pose = RigidTransform([0.0, 0.0, 0.053])
            adapter = Cylinder(0.022, 0.022)
            p.RegisterVisualGeometry(
                flange, adapter_pose, adapter, "wrist_adapter", [0.19, 0.21, 0.23, 1.0]
            )
            p.RegisterCollisionGeometry(
                flange,
                adapter_pose,
                adapter,
                "wrist_adapter",
                CoulombFriction(0.8, 0.6),
            )
            p.WeldFrames(
                p.GetFrameByName("iiwa_link_7", arm),
                p.GetFrameByName("wrist", hand),
                RigidTransform([0, 0, 0.061]),
            )
            self.models[side], self.hands[side] = arm, hand
            self.palm_frames[side] = p.GetFrameByName("palm", hand)
            for model in (arm, hand):
                for body_index in p.GetBodyIndices(model):
                    body = p.get_body(body_index)
                    for g in p.GetCollisionGeometriesForBody(body):
                        self.owner[g] = (side, body.name())
        _add_static_geometry(p, LAYOUT)
        p.Finalize()
        if renderer is not None:
            self.scene_graph.AddRenderer("dap_studio", MakeRenderEngineVtk(renderer))
        sensor = (
            add_head_camera(builder, p, self.scene_graph, camera) if camera else None
        )
        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = p.GetMyMutableContextFromRoot(self.context)
        self.sg_context = self.scene_graph.GetMyContextFromRoot(self.context)
        self.camera = sensor
        self.camera_context = (
            sensor.GetMyContextFromRoot(self.context) if sensor else None
        )
        self.arm_idx, self.digit_idx = {}, {}
        for side in SIDES:
            self.arm_idx[side] = np.array(
                [
                    p.GetJointByName(
                        f"iiwa_joint_{i}", self.models[side]
                    ).position_start()
                    for i in range(1, 8)
                ]
            )
            self.digit_idx[side] = {
                d: np.array(
                    [
                        p.GetJointByName(n, self.hands[side]).position_start()
                        for n in _digit_joints(d)
                    ]
                )
                for d in DIGITS
            }
        self._prepare_motion()
        self.set_time(0.0)

    def query_object(self):
        return self.scene_graph.get_query_output_port().Eval(self.sg_context)

    def palm_centers(self):
        return np.array(
            [
                frame.CalcPoseInWorld(self.plant_context).translation()
                for frame in self.palm_frames.values()
            ]
        )

    def render_rgb(self):
        img = self.camera.color_image_output_port().Eval(self.camera_context)
        return np.array(img.data[:, :, :3], copy=True)

    def render_depth(self):
        img = self.camera.depth_image_32F_output_port().Eval(self.camera_context)
        return np.array(img.data[:, :, 0], copy=True)

    def _first_pose(self, targets):
        p = self.plant
        ik = InverseKinematics(p, self.plant_context)
        q = ik.q()
        prog = ik.get_mutable_prog()
        nominal = p.GetPositions(self.plant_context).copy()
        for side, target in zip(SIDES, targets):
            nominal[self.arm_idx[side]] = Q_READY
            world, palm = p.world_frame(), self.palm_frames[side]
            pos = target.translation()
            ik.AddPositionConstraint(palm, np.zeros(3), world, pos - 1e-5, pos + 1e-5)
            ik.AddOrientationConstraint(
                palm, RotationMatrix(), world, target.rotation(), 0.001
            )
        prog.AddQuadraticErrorCost(np.eye(len(nominal)), nominal, q)
        rng = np.random.default_rng(27)
        for attempt in range(16):
            guess = nominal.copy()
            for side in SIDES:
                if attempt:
                    guess[self.arm_idx[side]] += rng.normal(0, 0.7, 7)
            prog.SetInitialGuess(q, guess)
            result = Solve(prog)
            if result.is_success():
                return result.GetSolution(q)
        raise RuntimeError("The initial dap wrist poses are unreachable")

    def _track(self, q, targets):
        p, ctx = self.plant, self.plant_context
        q = q.copy()
        world = p.world_frame()
        # Each arm is independent. Continuation from the previous pose keeps the
        # redundant elbow on its IK branch without solving a full program per sample.
        for side, target in zip(SIDES, targets):
            frame, idx = self.palm_frames[side], self.arm_idx[side]
            for iteration in range(30):
                p.SetPositions(ctx, q)
                pose = frame.CalcPoseInWorld(ctx)
                diff = (target.rotation() @ pose.rotation().inverse()).ToAngleAxis()
                error = np.r_[
                    diff.axis() * diff.angle(),
                    target.translation() - pose.translation(),
                ]
                if np.linalg.norm(error) < 1e-7:
                    break
                J = p.CalcJacobianSpatialVelocity(
                    ctx, JacobianWrtVariable.kQDot, frame, np.zeros(3), world, world
                )[:, idx]
                pinv = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(6), np.eye(6))
                travel = np.linalg.norm(error[:3]) + 5.0 * np.linalg.norm(error[3:])
                lower = p.GetPositionLowerLimits()[idx] + 0.45
                upper = p.GetPositionUpperLimits()[idx] - 0.45
                cushion = np.clip(q[idx], lower, upper) - q[idx]
                bias = (
                    min(0.08, travel) * (Q_READY - q[idx] + 20.0 * cushion)
                    if iteration == 0
                    else np.zeros(7)
                )
                q[idx] += pinv @ error + (np.eye(7) - pinv @ J) @ bias
            else:
                raise RuntimeError(
                    f"Cartesian tracking failed for {side}, error={error}"
                )
        return q

    @staticmethod
    def _targets(values):
        # Common grasp frame motion plus antisymmetric opening offsets. Positive
        # pitch slopes both sets of fingertips down as in a thumb-up greeting.
        length, side_gap, height, pitch, pull, side_shift, split_height = values
        targets = []
        for sign in (1, -1):
            a = np.deg2rad(pitch)
            x = np.array([sign * np.cos(a), 0, -np.sin(a)])
            z = np.array([0.0, sign, 0.0])
            R = RotationMatrix(np.column_stack([x, np.cross(z, x), z]))
            pos = MEETING + [
                pull - sign * length,
                side_shift - sign * side_gap,
                height + sign * split_height,
            ]
            targets.append(RigidTransform(R, pos))
        return targets

    def _prepare_motion(self):
        # metres and degrees. First contact is followed by a shared pull/lift,
        # a short settle, finger extension, and distinct retreat trajectories.
        times = np.array(
            [
                0.0,
                0.92,
                1.12,
                1.42,
                1.72,
                1.94,
                2.16,
                2.45,
                2.83,
                3.23,
                3.31,
                3.74,
                3.99,
            ]
        )
        # Each hand's half-spacing at tangency of the two sculpted palm hulls.
        touch = 0.011977
        vals = np.array(
            [
                [0.145, 0.085, -0.067, 22.0, 0.0, 0.0, 0.0],
                [0.022, touch, 0.0, 40.0, 0.0, 0.0, -0.007],
                [0.022, touch, 0.0, 40.0, 0.0, 0.0, -0.007],
                [0.022, touch, 0.047, 40.0, -0.038, 0.008, -0.007],
                [0.022, touch, 0.029, 40.0, -0.019, 0.003, -0.007],
                [0.022, touch, 0.029, 40.0, -0.019, 0.003, -0.007],
                [0.049, 0.044, 0.045, 32.0, -0.012, 0.0, -0.006],
                [0.112, 0.079, 0.005, 24.0, 0.0, 0.0, -0.002],
                [0.125, 0.045, -0.023, 12.0, 0.0, 0.0, 0.0],
                [0.05519, 0.010, 0.016, 10.0, 0.0, 0.0, 0.0],
                [0.05519, 0.010, 0.016, 10.0, 0.0, 0.0, 0.0],
                [0.135, 0.061, -0.037, 16.0, 0.0, 0.0, 0.0],
                [0.145, 0.085, -0.067, 22.0, 0.0, 0.0, 0.0],
            ]
        )
        xyz = PiecewisePolynomial.CubicShapePreserving(
            times, vals.T, zero_end_point_derivatives=True
        )
        t = np.unique(
            np.round(np.r_[times, 0.22, 0.58, np.arange(0.0, times[-1], 0.016)], 8)
        )
        q = self._first_pose(self._targets(vals[0]))
        path = []
        for ti in t:
            if ti < 0.92:
                u = reach_progress(ti / 0.92)
                pose = (1 - u) * vals[0] + u * vals[1]
                pose[2] += 0.040 * np.sin(np.pi * u)
            elif 1.94 < ti < 2.83:
                u = release_progress((ti - 1.94) / 0.89)
                pose = xyz.value(1.94 + 0.89 * u).ravel()
            else:
                pose = xyz.value(ti).ravel()
            q = self._track(q, self._targets(pose))
            path.append(q)
        path = np.array(path)
        self.plant.SetPositions(self.plant_context, path[np.searchsorted(t, 1.12)])
        grip = self._fit_fingers()
        # Finger curl lags initial contact. Both hands extend before separating.
        column = t[:, None]
        clasp = self._smooth((column - 0.92) / 0.28) * (
            1.0 - self._smooth((column - 1.75) / 0.19)
        )
        bump = self._smooth((column - 2.30) / 0.49) * (
            1.0 - self._smooth((column - 3.40) / 0.53)
        )
        rest = 1.0 - self._smooth(column / 0.86) + self._smooth((column - 3.54) / 0.45)
        for side in SIDES:
            for digit, idx in self.digit_idx[side].items():
                thumb = digit == "thumb"
                opened = np.array([-0.60, 0.0, 0.0]) if thumb else np.zeros(3)
                fist = np.array([0.15, 1.20, 1.02] if thumb else [1.25, 1.48, 0.91])
                relaxed = (
                    np.zeros(3)
                    if thumb
                    else (1 + DIGITS.index(digit)) * np.array([0.025, 0.039, 0.027])
                )
                path[:, idx] = (
                    opened
                    + rest * relaxed
                    + clasp * (grip[idx] - opened)
                    + bump * (fist - opened)
                )
        lo, hi = (
            self.plant.GetPositionLowerLimits(),
            self.plant.GetPositionUpperLimits(),
        )
        if np.any(path < lo - 1e-5) or np.any(path > hi + 1e-5):
            raise RuntimeError(
                "The dap trajectory exceeded the model's joint position limits"
            )
        vmax = self.plant.GetVelocityUpperLimits()
        v = np.diff(path, axis=0) / np.diff(t)[:, None]
        ratio = np.maximum(1.0, 1.015 * np.max(np.abs(v) / vmax, axis=1))
        # Slow only an interval that needs time to meet a joint's speed limit.
        # Expand that interval by one sample on either side for gentler transitions.
        ratio = np.maximum.reduce(
            [ratio, np.r_[ratio[:1], ratio[:-1]], np.r_[ratio[1:], ratio[-1:]]]
        )
        self._phase_times = t
        self._times = np.r_[0.0, np.cumsum(np.diff(t) * ratio)]
        self._positions = path
        self.duration = self._times[-1]

    @staticmethod
    def _smooth(x):
        x = np.clip(x, 0, 1)
        return x**3 * (10 + x * (-15 + 6 * x))

    def _fit_fingers(self):
        """Close until a finger meets the other hand, then flex its remaining joints."""
        q = self.plant.GetPositions(self.plant_context).copy()
        targets = {
            "index": [1.0, 1.45, 0.9],
            "middle": [1.05, 1.48, 1.0],
            "ring": [1.12, 1.5, 1.02],
            "little": [1.17, 1.54, 1.03],
            "thumb": [0.68, 0.82, 0.64],
        }
        for side in SIDES:
            q[self.digit_idx[side]["thumb"][0]] = -0.60
        for side in SIDES:
            for digit, idx in self.digit_idx[side].items():
                opened = q[idx].copy()
                for frac in np.linspace(0.02, 1.0, 50):
                    trial = q.copy()
                    trial[idx] = opened + (np.array(targets[digit]) - opened) * frac
                    self.plant.SetPositions(self.plant_context, trial)
                    if self._digit_clearance(side, digit) < 0.00035:
                        break
                    q = trial
        # A proximal phalanx can touch first. Continue wrapping the free PIP
        # and DIP joints so the two fingertips can reach around the grasp.
        for side in SIDES:
            for digit, idx in self.digit_idx[side].items():
                middle = "proximal" if digit == "thumb" else "middle"
                cap = [1.17, 1.12] if digit == "thumb" else [1.62, 1.25]
                for joint, parts, limit in [
                    (1, (middle, "distal"), cap[0]),
                    (2, ("distal",), cap[1]),
                ]:
                    for angle in np.linspace(q[idx[joint]], limit, 51)[1:]:
                        trial = q.copy()
                        trial[idx[joint]] = angle
                        self.plant.SetPositions(self.plant_context, trial)
                        if self._digit_clearance(side, digit, parts) < 0.00035:
                            break
                        q = trial
        self.plant.SetPositions(self.plant_context, q)
        return q

    def _digit_clearance(self, side, digit, parts=None):
        query = self.query_object()
        clearance = np.inf
        for pair in query.ComputeSignedDistancePairwiseClosestPoints(0.005):
            a, b = self.owner.get(pair.id_A), self.owner.get(pair.id_B)
            if a and b and a[0] != b[0]:
                body = a[1] if a[0] == side else b[1]
                if body.startswith(digit + "_") and (
                    parts is None or body.split("_")[1] in parts
                ):
                    clearance = min(clearance, pair.distance)
        return clearance

    def q_at(self, t):
        t = np.clip(t, 0.0, self.duration)
        hi = min(np.searchsorted(self._times, t, side="right"), len(self._times) - 1)
        lo = hi - 1
        f = (t - self._times[lo]) / (self._times[hi] - self._times[lo])
        return (1 - f) * self._positions[lo] + f * self._positions[hi]

    def set_positions(self, q):
        self.plant.SetPositions(self.plant_context, q)

    def set_time(self, t):
        self.set_positions(self.q_at(t))

    def time_at_phase(self, authored_time):
        """Convert a choreography time to playback seconds after joint-speed limiting."""
        return float(np.interp(authored_time, self._phase_times, self._times))

    def phase(self, t):
        authored = np.interp(t, self._times, self._phase_times)
        boundaries = [
            (0.92, "Reach"),
            (1.02, "Palm touch"),
            (1.20, "Clasp"),
            (1.75, "Pull"),
            (1.94, "Open"),
            (2.30, "Release"),
            (2.83, "Make a fist"),
            (3.23, "Approach for bump"),
            (3.31, "Knuckle bump"),
        ]
        return next((name for end, name in boundaries if authored < end), "Recoil")
