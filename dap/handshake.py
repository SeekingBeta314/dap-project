"""A single rising palm slap between two hands held in a fixed cupped pose."""

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    Cylinder,
    DiagramBuilder,
    InverseKinematics,
    JacobianWrtVariable,
    MakeRenderEngineVtk,
    RigidTransform,
    RotationMatrix,
    Solve,
)

from dap.camera import add_head_camera
from dap.hand_timing import dap_swing_progress
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
MEETING = np.array([0.67, 0.0, 0.59])
# Joint radians from the base to the tip. The ulnar digits curve more deeply,
# leaving a hollow between the palm, the gently bent fingers, and the open thumb.
CUP_ANGLES = {
    "index": (0.19, 0.36, 0.13),
    "middle": (0.23, 0.42, 0.16),
    "ring": (0.28, 0.45, 0.19),
    "little": (0.31, 0.49, 0.20),
    "thumb": (-0.42, 0.21, 0.12),
}
LAYOUT = SceneLayout(distance=1.34)


def _digit_joints(digit):
    suffixes = ("cmc", "mcp", "ip") if digit == "thumb" else ("mcp", "pip", "dip")
    return [digit + "_" + suffix for suffix in suffixes]


class HandshakeScene:
    """Two iiwa wrists swing cupped right hands together and rebound."""

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
        for hand in self.hands.values():
            for joint in self.plant.GetJointIndices(hand):
                self.plant.get_joint(joint).Lock(self.plant_context)

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
    def _targets(pose):
        # The finger axis rises outward from each wrist. A small backward cant
        # puts the opposing palmar rims together with room for both sets of fingers.
        half_x, half_y, height, rake, cant, split_z, lead = pose
        targets = []
        for sign in (1, -1):
            a = np.deg2rad(rake)
            x = np.array([sign * np.cos(a), 0.0, np.sin(a)])
            z = np.array([0.0, sign, 0.0])
            facing = RotationMatrix(np.column_stack([x, np.cross(z, x), z]))
            # Roll the radial and ulnar pads onto the contact plane together.
            rotation = (
                facing
                @ RotationMatrix.MakeYRotation(np.deg2rad(cant))
                @ RotationMatrix.MakeXRotation(np.deg2rad(-2.96))
            )
            # Carry the striking wrist through a longer sweep than the receiving wrist.
            center = MEETING + lead * np.array([-1.0, -1.0, -0.4])
            position = center + [
                -sign * half_x,
                -sign * half_y,
                height + sign * split_z,
            ]
            targets.append(RigidTransform(rotation, position))
        return targets

    def _prepare_motion(self):
        # Relative wrist offsets in metres, then rake and cant in degrees, and
        # a vertical stagger and asymmetry offset in metres.
        ready = np.array([0.073, 0.145, -0.055, 32.0, 8.0, 0.007, 0.043])
        strike = np.array([0.002, 0.019, 0.0, 37.0, 4.0, 0.0, 0.0])
        finish = np.array([0.080, 0.118, -0.065, 32.0, 8.5, -0.004, 0.025])
        q = self._first_pose(self._targets(ready))
        for side in SIDES:
            for digit, idx in self.digit_idx[side].items():
                q[idx] = CUP_ANGLES[digit]

        # Solve the normal separation from the actual pad surfaces. This also
        # accommodates a palm described by multiple contact meshes.
        pads = [
            self.plant.GetCollisionGeometriesForBody(
                self.plant.GetBodyByName("palm", self.hands[side])
            )
            for side in SIDES
        ]
        contact_q = q
        for _ in range(8):
            contact_q = self._track(contact_q, self._targets(strike))
            self.set_positions(contact_q)
            query = self.query_object()
            gap = min(
                query.ComputeSignedDistancePairClosestPoints(a, b).distance
                for a in pads[0]
                for b in pads[1]
            )
            if abs(gap - 0.00001) < 0.000001:
                break
            strike[1] += (0.00001 - gap) / 2.0
        else:
            raise RuntimeError("Could not resolve cupped palm contact")

        onset, impact, release, end = 0.18, 0.59, 0.61, 1.12
        times = np.unique(
            np.round(np.r_[np.arange(0.0, end, 0.008), onset, impact, release, end], 8)
        )
        positions = []
        for t in times:
            if t < onset:
                # The striker draws back while the receiving arm presents its palm.
                u = t / onset
                windup = u * u * (3.0 - 2.0 * u)
                pose = finish + windup * (ready - finish)
            elif t <= impact:
                u = np.clip((t - onset) / (impact - onset), 0.0, 1.0)
                # Reach from rest with nonzero approach speed at the clap.
                swing = dap_swing_progress(u)
                pose = ready + swing * (strike - ready)
                arc = np.sin(np.pi * swing)
                pose[2] += 0.030 * arc
                pose[5] += 0.016 * arc
            elif t < release:
                pose = strike
            else:
                u = np.clip((t - release) / (end - release), 0.0, 1.0)
                # Immediate rebound, followed by a soft arc back down.
                rebound = 1.0 - (1.0 - u) ** 2
                pose = strike + rebound * (finish - strike)
                pose[2] += 0.030 * np.sin(np.pi * rebound)
            q = self._track(q, self._targets(pose))
            positions.append(q)
        self._positions = np.array(positions)
        lo, hi = (
            self.plant.GetPositionLowerLimits(),
            self.plant.GetPositionUpperLimits(),
        )
        if np.any(self._positions < lo - 1e-5) or np.any(self._positions > hi + 1e-5):
            raise RuntimeError(
                "The dap trajectory exceeded the model's joint position limits"
            )
        # Stretch only the intervals that need more time under the arm speed limits.
        dt = np.diff(times)
        velocities = np.diff(self._positions, axis=0) / dt[:, None]
        speed_fraction = np.max(
            abs(velocities) / self.plant.GetVelocityUpperLimits(), axis=1
        )
        stretch = np.maximum(1.0, 1.015 * speed_fraction)
        stretch = np.maximum.reduce(
            [stretch, np.r_[stretch[0], stretch[:-1]], np.r_[stretch[1:], stretch[-1]]]
        )
        self._times = np.r_[0.0, np.cumsum(dt * stretch)]
        self.swing_time, self.impact_time, self.release_time = np.interp(
            [onset, impact, release], times, self._times
        )
        self.duration = self._times[-1]

    def q_at(self, t):
        t = np.clip(t, 0.0, self.duration)
        hi = min(np.searchsorted(self._times, t, side="right"), len(self._times) - 1)
        lo = hi - 1
        f = (t - self._times[lo]) / (self._times[hi] - self._times[lo])
        return self._positions[lo] + f * (self._positions[hi] - self._positions[lo])

    def set_positions(self, q):
        self.plant.SetPositions(self.plant_context, q)

    def set_time(self, t):
        self.set_positions(self.q_at(t))

    def phase(self, t):
        if t < self.swing_time:
            return "Wind-up"
        if t < self.impact_time:
            return "Swing"
        if t <= self.release_time:
            return "Palm contact"
        return "Rebound"
