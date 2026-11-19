"""Two iiwa arms facing each other, each ending in a dap hand.

Our arm ("iiwa") sits at the world origin facing +x. The partner arm
("partner") sits across from it; where exactly is a per-trial `SceneLayout`
(distance, lateral offset, yaw, height). Both bases stand on pedestals over a
floor so the head camera sees a realistic background.

Ranges for the layout are in dap/scenario.py and docs/setup.md.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    CollisionFilterDeclaration,
    CoulombFriction,
    DiagramBuilder,
    GeometrySet,
    LoadModelDirectivesFromString,
    Parser,
    ProcessModelDirectives,
    RigidTransform,
    RollPitchYaw,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Nominal distance between the two arm bases (m). Drake's iiwa7 reaches about
# 0.8 m; ManiSkill3 / robosuite put facing arms 1.5-1.6 m apart.
ARM_SEPARATION = 1.4

# Default pose: palm about 0.58 m out and 0.6 m up, palm normal pointing at
# the other arm (found by a grid search over joints 2, 4, 6).
Q_READY = np.array([0.0, 0.2, 0.0, -1.7, 0.0, -0.3, 0.0])

IIWA_URL = "package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf"

# Floor height (m) relative to our base. Both bases stand on pedestals; 0.75 m
# is a typical table height for tabletop iiwa setups.
FLOOR_Z = -0.75
PEDESTAL_SIZE = 0.3  # m, square pedestal footprint

# Palm face center in the palm frame (the palm box is 2 cm thick; see dap_hand.sdf).
P_PALM_FACE = np.array([0.0, 0.0, 0.02])

OURS = "iiwa"
PARTNER = "partner"


@dataclass(frozen=True)
class SceneLayout:
    """Where the partner base sits relative to ours (our base frame = world)."""

    distance: float = ARM_SEPARATION  # m, along +x
    lateral: float = 0.0  # m, along y
    yaw_deg: float = 0.0  # extra yaw on top of facing us (180 deg)
    height: float = 0.0  # m, partner base z relative to ours

    def partner_base(self):
        return RigidTransform(
            RollPitchYaw(0, 0, np.deg2rad(180.0 + self.yaw_deg)),
            [self.distance, self.lateral, self.height],
        )


NOMINAL_LAYOUT = SceneLayout()


def _arm_directives(name, xyz, yaw_deg):
    # Joint names are iiwa_joint_1..7 inside every iiwa model instance.
    joints = ", ".join(f"iiwa_joint_{i + 1}: [{v}]" for i, v in enumerate(Q_READY))
    x, y, z = xyz
    return f"""
- add_model:
    name: {name}
    file: {IIWA_URL}
    default_joint_positions:
      {{{joints}}}
- add_weld:
    parent: world
    child: {name}::iiwa_link_0
    X_PC:
      translation: [{x}, {y}, {z}]
      rotation: !Rpy {{ deg: [0, 0, {yaw_deg}] }}
- add_model:
    name: {name}_hand
    file: package://dap/dap_hand.sdf
- add_weld:
    parent: {name}::iiwa_link_7
    child: {name}_hand::palm
    X_PC:
      translation: [0, 0, 0.045]
"""


def scene_directives(layout: SceneLayout = NOMINAL_LAYOUT):
    """Model directives (YAML) for the two arms + hands."""
    return "directives:" + _arm_directives(OURS, (0, 0, 0), 0) + _arm_directives(
        PARTNER,
        (layout.distance, layout.lateral, layout.height),
        180.0 + layout.yaw_deg,
    )


def make_parser(plant):
    parser = Parser(plant)
    parser.package_map().AddPackageXml(str(MODELS_DIR / "package.xml"))
    return parser


def _add_static_geometry(plant, layout):
    """Floor + one pedestal under each base, welded to the world. Geometry on
    the world body is not checked against the world-welded iiwa_link_0."""
    W = plant.world_body()
    grey = np.array([0.55, 0.55, 0.55, 1.0])
    floor = Box(6.0, 6.0, 0.02)
    X_floor = RigidTransform([layout.distance / 2, 0, FLOOR_Z - 0.01])
    plant.RegisterVisualGeometry(W, X_floor, floor, "floor_visual", [0.8, 0.8, 0.78, 1.0])
    plant.RegisterCollisionGeometry(W, X_floor, floor, "floor_collision", CoulombFriction(1, 1))
    for name, (x, y, z) in [("ours", (0, 0, 0)),
                            ("partner", (layout.distance, layout.lateral, layout.height))]:
        h = z - FLOOR_Z
        box = Box(PEDESTAL_SIZE, PEDESTAL_SIZE, h)
        X = RigidTransform([x, y, FLOOR_Z + h / 2])
        plant.RegisterVisualGeometry(W, X, box, f"pedestal_{name}_visual", grey)
        plant.RegisterCollisionGeometry(W, X, box, f"pedestal_{name}_collision", CoulombFriction(1, 1))


def collision_geometries(plant, model_instances):
    ids = []
    for mi in model_instances:
        for b in plant.GetBodyIndices(mi):
            ids += plant.GetCollisionGeometriesForBody(plant.get_body(b))
    return GeometrySet(ids)


def add_scene(builder, time_step=1e-3, layout: SceneLayout = NOMINAL_LAYOUT, static=True):
    """Add the plant + scene graph to `builder`. Returns (plant, scene_graph).

    The two hands are excluded from colliding with each other: palm-to-palm
    contact is the goal, so it must not count as a collision.
    """
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = make_parser(plant)
    ProcessModelDirectives(LoadModelDirectivesFromString(scene_directives(layout)), plant, parser)
    if static:
        _add_static_geometry(plant, layout)
    plant.Finalize()
    hands = [plant.GetModelInstanceByName(f"{n}_hand") for n in (OURS, PARTNER)]
    scene_graph.collision_filter_manager().Apply(
        CollisionFilterDeclaration().ExcludeBetween(
            collision_geometries(plant, hands[:1]), collision_geometries(plant, hands[1:])
        )
    )
    return plant, scene_graph


def build_scene(meshcat=None, time_step=1e-3, layout: SceneLayout = NOMINAL_LAYOUT,
                camera=None, static=True):
    """Build a diagram with the scene, plus Meshcat and/or the head camera.

    `camera` is a `dap.camera.HeadCameraConfig`; the sensor is the subsystem
    named `camera.name`.
    """
    builder = DiagramBuilder()
    plant, scene_graph = add_scene(builder, time_step, layout, static)
    if camera is not None:
        from dap.camera import add_head_camera

        add_head_camera(builder, plant, scene_graph, camera)
    if meshcat is not None:
        from pydrake.all import AddDefaultVisualization

        AddDefaultVisualization(builder, meshcat)
    diagram = builder.Build()
    return diagram, plant, scene_graph


class Scene:
    """A built scene plus a context, with helpers to pose both arms and query
    palms, distances, and the camera. Rebuilt whenever the layout changes."""

    def __init__(self, layout: SceneLayout = NOMINAL_LAYOUT, camera=None, meshcat=None,
                 static=True):
        self.layout = layout
        self.camera_cfg = camera
        self.diagram, self.plant, self.scene_graph = build_scene(
            meshcat=meshcat, layout=layout, camera=camera, static=static
        )
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)
        self.sg_context = self.scene_graph.GetMyContextFromRoot(self.context)
        p = self.plant
        self.models = {n: p.GetModelInstanceByName(n) for n in (OURS, PARTNER)}
        self.hands = {n: p.GetModelInstanceByName(f"{n}_hand") for n in (OURS, PARTNER)}
        self.palm_frames = {n: p.GetFrameByName("palm", self.hands[n]) for n in (OURS, PARTNER)}
        # Index of each arm's 7 joints / each hinge in the full position vector.
        self.idx = {
            n: np.array([p.GetJointByName(f"iiwa_joint_{i}", self.models[n]).position_start()
                         for i in range(1, 8)])
            for n in (OURS, PARTNER)
        }
        self.hinge_idx = {
            n: p.GetJointByName("finger_hinge", self.hands[n]).position_start()
            for n in (OURS, PARTNER)
        }
        self.q_lower = p.GetPositionLowerLimits()
        self.q_upper = p.GetPositionUpperLimits()
        self.v_max = p.GetVelocityUpperLimits()  # same indexing as q (revolute only)
        inspector = self.scene_graph.model_inspector()
        self._owner = {}
        for name in (OURS, PARTNER):
            for mi in (self.models[name], self.hands[name]):
                for b in p.GetBodyIndices(mi):
                    for g in p.GetCollisionGeometriesForBody(p.get_body(b)):
                        self._owner[g] = name
        self._inspector = inspector
        self.camera = None
        if camera is not None:
            self.camera = self.diagram.GetSubsystemByName(camera.name)
            self.camera_context = self.camera.GetMyContextFromRoot(self.context)

    # -- posing ---------------------------------------------------------
    def q(self):
        return self.plant.GetPositions(self.plant_context).copy()

    def set_q(self, q):
        self.plant.SetPositions(self.plant_context, q)

    def set_arm(self, name, q_arm, hinge=None):
        q = self.q()
        q[self.idx[name]] = q_arm
        if hinge is not None:
            q[self.hinge_idx[name]] = hinge
        self.set_q(q)

    def set_ours(self, q_arm, hinge=0.0):
        self.set_arm(OURS, q_arm, hinge)

    def set_partner(self, q_arm, hinge=0.0):
        self.set_arm(PARTNER, q_arm, hinge)

    def arm_q(self, name, q_full=None):
        q = self.q() if q_full is None else q_full
        return q[self.idx[name]].copy()

    # -- queries ----------------------------------------------------------
    def palm_pose(self, name=OURS):
        return self.plant.CalcRelativeTransform(
            self.plant_context, self.plant.world_frame(), self.palm_frames[name]
        )

    def palm(self, name=OURS):
        """(palm face center, palm normal) in the world frame."""
        X = self.palm_pose(name)
        return X @ P_PALM_FACE, X.rotation().matrix()[:, 2]

    def _query(self):
        return self.scene_graph.get_query_output_port().Eval(self.sg_context)

    def min_distance(self, max_distance=0.1):
        """Smallest signed distance over all unfiltered collision pairs within
        `max_distance` (hand-hand pairs are filtered), and the pair's names.
        Returns (inf, None) if nothing is that close."""
        pairs = self._query().ComputeSignedDistancePairwiseClosestPoints(max_distance)
        if not pairs:
            return np.inf, None
        p = min(pairs, key=lambda r: r.distance)
        names = (self._inspector.GetName(p.id_A), self._inspector.GetName(p.id_B))
        return p.distance, names

    def clearance(self, ignore=None, max_distance=0.1):
        """Smallest signed distance over unfiltered pairs, skipping any pair
        that involves geometry of the arm/hand named `ignore` (OURS or
        PARTNER). Returns inf if nothing is within `max_distance`."""
        pairs = self._query().ComputeSignedDistancePairwiseClosestPoints(max_distance)
        best = np.inf
        for p in pairs:
            if ignore and ignore in (self._owner.get(p.id_A), self._owner.get(p.id_B)):
                continue
            best = min(best, p.distance)
        return best

    def min_arm_distance(self, max_distance=0.5):
        """Smallest signed distance between any geometry of our arm/hand and
        any geometry of the partner arm/hand (hand-hand pairs excluded)."""
        pairs = self._query().ComputeSignedDistancePairwiseClosestPoints(max_distance)
        best = np.inf
        for p in pairs:
            a, b = self._owner.get(p.id_A), self._owner.get(p.id_B)
            if a and b and a != b:
                best = min(best, p.distance)
        return best

    # -- camera -----------------------------------------------------------
    def render_depth(self):
        """Clean depth image (H, W) float32, metres (inf = beyond range)."""
        img = self.camera.depth_image_32F_output_port().Eval(self.camera_context)
        return np.array(img.data[:, :, 0], copy=True)

    def render_label(self):
        img = self.camera.label_image_output_port().Eval(self.camera_context)
        return np.array(img.data[:, :, 0], copy=True)

    def render_rgb(self):
        img = self.camera.color_image_output_port().Eval(self.camera_context)
        return np.array(img.data[:, :, :3], copy=True)

    def palm_label(self, name=PARTNER):
        """Render label (int) of a hand's palm body: Drake labels each body by
        its index."""
        return int(self.plant.GetBodyByName("palm", self.hands[name]).index())
