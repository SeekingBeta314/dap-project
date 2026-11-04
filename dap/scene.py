"""Two iiwa arms facing each other, each ending in a dap hand.

Our arm ("iiwa") sits at the world origin facing +x. The partner arm
("partner") sits ARM_SEPARATION meters down +x, yawed 180 degrees to face us.
"""

from pathlib import Path

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    LoadModelDirectivesFromString,
    Parser,
    ProcessModelDirectives,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Distance between the two arm bases (m). An iiwa7 reaches about 0.8 m, so the
# midpoint is comfortably reachable by both arms.
ARM_SEPARATION = 1.4

# Both arms' default pose: palm about 0.58 m out and 0.6 m up, palm normal
# pointing at the other arm (found by a grid search over joints 2, 4, 6).
Q_READY = np.array([0.0, 0.2, 0.0, -1.7, 0.0, -0.3, 0.0])

IIWA_URL = "package://drake_models/iiwa_description/sdf/iiwa7_with_box_collision.sdf"


def _arm_directives(name, x, yaw_deg):
    # Joint names are iiwa_joint_1..7 inside every iiwa model instance.
    joints = ", ".join(f"iiwa_joint_{i + 1}: [{v}]" for i, v in enumerate(Q_READY))
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
      translation: [{x}, 0, 0]
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


def scene_directives():
    """Model directives (YAML) for the full two-arm scene."""
    return "directives:" + _arm_directives("iiwa", 0, 0) + _arm_directives(
        "partner", ARM_SEPARATION, 180
    )


def make_parser(plant):
    parser = Parser(plant)
    parser.package_map().AddPackageXml(str(MODELS_DIR / "package.xml"))
    return parser


def add_scene(builder, time_step=1e-3):
    """Add the plant + scene graph to `builder`. Returns (plant, scene_graph)."""
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = make_parser(plant)
    directives = LoadModelDirectivesFromString(scene_directives())
    ProcessModelDirectives(directives, plant, parser)
    plant.Finalize()
    return plant, scene_graph


def build_scene(meshcat=None, time_step=1e-3):
    """Build a diagram with just the scene (and Meshcat, if given)."""
    builder = DiagramBuilder()
    plant, scene_graph = add_scene(builder, time_step)
    if meshcat is not None:
        from pydrake.all import AddDefaultVisualization

        AddDefaultVisualization(builder, meshcat)
    diagram = builder.Build()
    return diagram, plant, scene_graph
