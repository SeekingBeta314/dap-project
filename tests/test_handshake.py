"""Geometry-level checks for the slap, thumb-side clasp and separate knuckle bump."""

import numpy as np
import pytest

from dap.handshake import OURS, PARTNER, HandshakeScene
from dap.handshake_audit import audit_handshake


@pytest.fixture(scope="module")
def replay():
    scene = HandshakeScene()
    return scene, audit_handshake(scene)


def test_slap_clasp_pull_and_knuckle_bump(replay):
    scene, report = replay
    assert report["first_cross_robot_within_0.2_mm"]["bodies"] == [
        "iiwa/palm",
        "partner/palm",
    ]
    assert 0 <= report["palm_touch"]["gap_mm"] < 0.08
    for side in (OURS, PARTNER):
        tips = report["grasp_fingertip_to_opposite_palm_mm"][side]
        assert 0 < tips["thumb"] < 1.0
        assert 0 < tips["index"] < 1.0
    assert report["pull_displacement_xyz_mm"][0] < -30
    assert report["pull_displacement_xyz_mm"][2] > 40
    assert report["separated_hand_clearance_mm"] > 100
    assert report["knuckle_bump"]["bodies"] == [
        "iiwa/middle_proximal",
        "partner/middle_proximal",
    ]
    assert 0 <= report["knuckle_bump"]["gap_mm"] < 0.08
    assert report["final_hand_clearance_mm"] > 100
    assert scene.phase(0) == "Reach"
    assert scene.phase(scene.duration) == "Recoil"


def test_swept_clearance_and_limits(replay):
    scene, report = replay
    assert scene.plant.num_positions() == 2 * (7 + 15)
    assert report["minimum_cross_robot"]["gap_mm"] >= 0
    assert report["minimum_cross_robot_involving_arm"]["gap_mm"] > 0.5
    assert report["minimum_self_or_environment"]["gap_mm"] > 0
    assert report["peak_joint_speed_fraction_of_limit"] <= 1.0
    assert np.all(scene._positions >= scene.plant.GetPositionLowerLimits() - 1e-7)
    assert np.all(scene._positions <= scene.plant.GetPositionUpperLimits() + 1e-7)
