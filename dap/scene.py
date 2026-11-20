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

    Every geometry pair is checked, including hand-hand pairs, so hands
    cannot pass through each other unnoticed. The one special pair is our
    palm vs. the partner palm: touching it is the goal, so `Scene.clearance`
    leaves it out and the scorer (dap/scoring.py) decides whether a palm-palm
    overlap is the dap (approached from the front) or a collision.
    """
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = make_parser(plant)
    ProcessModelDirectives(LoadModelDirectivesFromString(scene_directives(layout)), plant, parser)
    if static:
        _add_static_geometry(plant, layout)
    plant.Finalize()
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
        # Index of each arm's 7 joints in the full position vector (the hands
        # are rigid, so the arms' joints are all there is).
        self.idx = {
            n: np.array([p.GetJointByName(f"iiwa_joint_{i}", self.models[n]).position_start()
                         for i in range(1, 8)])
            for n in (OURS, PARTNER)
        }
        self.q_lower = p.GetPositionLowerLimits()
        self.q_upper = p.GetPositionUpperLimits()
        self.v_max = p.GetVelocityUpperLimits()  # same indexing as q (revolute only)
        self._scratch = {}  # scratch_context's cache
        inspector = self.scene_graph.model_inspector()
        self._owner = {}
        for name in (OURS, PARTNER):
            for mi in (self.models[name], self.hands[name]):
                for b in p.GetBodyIndices(mi):
                    for g in p.GetCollisionGeometriesForBody(p.get_body(b)):
                        self._owner[g] = name
        self._inspector = inspector
        # The palm-palm pair (the dap contact itself).
        self.palm_geoms = {
            n: p.GetCollisionGeometriesForBody(p.GetBodyByName("palm", self.hands[n]))[0]
            for n in (OURS, PARTNER)
        }
        self._palm_pair = frozenset(self.palm_geoms.values())
        self.camera = None
        if camera is not None:
            self.camera = self.diagram.GetSubsystemByName(camera.name)
            self.camera_context = self.camera.GetMyContextFromRoot(self.context)

    # -- posing ---------------------------------------------------------
    def q(self):
        return self.plant.GetPositions(self.plant_context).copy()

    def set_q(self, q):
        self.plant.SetPositions(self.plant_context, q)

    def set_arm(self, name, q_arm):
        q = self.q()
        q[self.idx[name]] = q_arm
        self.set_q(q)

    def set_ours(self, q_arm):
        self.set_arm(OURS, q_arm)

    def set_partner(self, q_arm):
        self.set_arm(PARTNER, q_arm)

    def arm_q(self, name, q_full=None):
        q = self.q() if q_full is None else q_full
        return q[self.idx[name]].copy()

    def new_context(self, filter_hands=False):
        """A fresh diagram context (and its plant context) for planning or
        IK, separate from the scene's own context so nothing done there can
        move the arms the scene is measuring. With `filter_hands`, hand-hand
        collision pairs are excluded in that context only (used by IK that
        constructs palm-to-palm meetings, where the hands are adjacent by
        design)."""
        ctx = self.diagram.CreateDefaultContext()
        if filter_hands:
            sg_ctx = self.scene_graph.GetMyMutableContextFromRoot(ctx)
            hands = [self.hands[OURS]], [self.hands[PARTNER]]
            self.scene_graph.collision_filter_manager(sg_ctx).Apply(
                CollisionFilterDeclaration().ExcludeBetween(
                    collision_geometries(self.plant, hands[0]),
                    collision_geometries(self.plant, hands[1]),
                )
            )
        return ctx, self.plant.GetMyMutableContextFromRoot(ctx)

    def scratch_context(self, filter_hands=False):
        """Like `new_context`, but one cached context per `filter_hands` value,
        reused across calls. For IK and feasibility solves only: each solve
        overwrites every position in it, so reuse does not change results,
        and it saves building a diagram context (a few ms, most of a reset)
        per solve. Never the context the scene measures with."""
        if filter_hands not in self._scratch:
            self._scratch[filter_hands] = self.new_context(filter_hands)
        return self._scratch[filter_hands]

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

    def _is_palm_pair(self, p):
        return frozenset((p.id_A, p.id_B)) == self._palm_pair

    def min_distance(self, max_distance=0.1):
        """Smallest signed distance over every pair except the palm-palm pair,
        within `max_distance`, and that pair's geometry names. Returns
        (inf, None) if nothing is that close."""
        pairs = [p for p in self._query().ComputeSignedDistancePairwiseClosestPoints(max_distance)
                 if not self._is_palm_pair(p)]
        if not pairs:
            return np.inf, None
        p = min(pairs, key=lambda r: r.distance)
        names = (self._inspector.GetName(p.id_A), self._inspector.GetName(p.id_B))
        return p.distance, names

    def clearance(self, ignore=None, max_distance=0.1):
        """Smallest signed distance over every pair except the palm-palm pair
        (hand-hand pairs included), skipping pairs that involve the arm/hand
        named `ignore` (OURS or PARTNER). inf if nothing is within
        `max_distance`. Negative = penetration = collision."""
        best = np.inf
        for p in self._query().ComputeSignedDistancePairwiseClosestPoints(max_distance):
            if self._is_palm_pair(p):
                continue
            if ignore and ignore in (self._owner.get(p.id_A), self._owner.get(p.id_B)):
                continue
            best = min(best, p.distance)
        return best

    def palm_palm_distance(self):
        """Signed distance between our palm box and the partner palm box
        (negative = overlapping)."""
        r = self._query().ComputeSignedDistancePairClosestPoints(
            self.palm_geoms[OURS], self.palm_geoms[PARTNER])
        return float(r.distance)

    def measure(self, max_distance=0.5):
        """Ground-truth snapshot for scoring (dap.scoring): both palm faces
        and frames, and the signed distances, from one distance query.
        `clearance` and `arm_dist` are as in `clearance()` and
        `min_arm_distance()` (inf when nothing is within `max_distance`)."""
        clearance, arm = np.inf, np.inf
        for p in self._query().ComputeSignedDistancePairwiseClosestPoints(max_distance):
            if self._is_palm_pair(p):
                continue
            clearance = min(clearance, p.distance)
            a, b = self._owner.get(p.id_A), self._owner.get(p.id_B)
            if a and b and a != b:
                arm = min(arm, p.distance)
        X_o, X_p = self.palm_pose(OURS), self.palm_pose(PARTNER)
        return dict(
            palm_p=X_o @ P_PALM_FACE, palm_R=X_o.rotation().matrix(),
            partner_p=X_p @ P_PALM_FACE, partner_R=X_p.rotation().matrix(),
            clearance=float(clearance), palm_palm=self.palm_palm_distance(),
            arm_dist=float(arm),
        )

    def min_arm_distance(self, max_distance=0.5):
        """Smallest signed distance between any geometry of our arm/hand and
        any geometry of the partner arm/hand, except the palm-palm pair."""
        best = np.inf
        for p in self._query().ComputeSignedDistancePairwiseClosestPoints(max_distance):
            if self._is_palm_pair(p):
                continue
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
