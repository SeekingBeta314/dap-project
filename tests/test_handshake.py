"""Check the rigid cup, single palm contact and recoil on the delivered motion."""

import numpy as np
import pytest

from dap.handshake import SIDES, HandshakeScene
from dap.handshake_audit import audit_handshake


@pytest.fixture(scope="module")
def replay():
    scene = HandshakeScene()
    return scene, audit_handshake(scene)


def test_both_hands_keep_a_cup_throughout_playback(replay):
    scene, report = replay
    assert scene.plant.num_positions() == 2 * (7 + 15)
    assert report["max_finger_or_thumb_joint_excursion_deg"] < 1e-9
    pocket = report["pocket_at_impact"]
    assert pocket["angle_between_palm_normals_deg"] > 155
    for side in SIDES:
        rise = report["fingertip_rise_above_knuckle_mm"][side]
        assert min(rise.values()) > 10
        # The center of the encounter lies over both palms and inside an air pocket.
        center_xy = pocket["center_in_palm_xy_mm"][side]
        assert np.linalg.norm(center_xy) < 15
        assert pocket["clearance_to_each_robot_mm"][side] > 5


def test_single_cupped_palm_dap(replay):
    scene, report = replay
    (episode,) = report["contact_episodes"]
    for measured in (episode["first"], report["at_impact"]):
        for side, surface in zip(SIDES, measured["surfaces"]):
            assert surface.startswith(f"{side}/palm/palm_")
        assert 0 <= measured["gap_mm"] <= report["contact_threshold_mm"]
    patches = report["palm_patches_at_impact"]
    assert patches["max_span_mm"] > 35
    assert len(patches["outer_contacts"]) == 2
    for patch in patches["outer_contacts"]:
        assert 0 <= patch["gap_mm"] <= report["contact_threshold_mm"]
        for side, surface in zip(SIDES, patch["surfaces"]):
            assert surface.startswith(f"{side}/palm/palm_")
            assert "mid_center" not in surface
    assert episode["first"]["time_s"] <= scene.impact_time <= episode["last_sample_s"]
    assert episode["duration_upper_bound_s"] < 0.1
    assert report["end_gap_mm"] > 50
    assert report["mean_closing_speed_m_s"] > 0.1
    assert report["mean_rebound_speed_m_s"] > 0.05


def test_long_curved_swing(replay):
    scene, report = replay
    assert report["initial_hand_clearance"]["gap_mm"] > 500
    striker, receiver = (report["palm_routes"][side] for side in SIDES)
    # Measure the inward stroke after the wind-up so drawing straight back
    # and returning along the same line cannot produce a passing arc.
    assert striker["inward_swing"]["max_bow_mm"] > 70
    assert receiver["inward_swing"]["max_bow_mm"] > 25
    assert (
        striker["inward_swing"]["path_length_mm"]
        > receiver["inward_swing"]["path_length_mm"]
    )
    assert striker["recovery"]["max_bow_mm"] > 35
    # Both palms carry through contact together, then ease into the rebound.
    assert report["contact_midpoint_displacement_mm"] > 2
    assert max(report["palm_velocity_changes"]["change_m_s"]["release"].values()) < 0.25
    # An identical wrist pose can hide an elbow jump when the animation loops.
    np.testing.assert_allclose(scene.q_at(0), scene.q_at(scene.duration), atol=1e-7, rtol=0)


def test_swept_clearance_and_joint_limits(replay):
    scene, report = replay
    assert report["closest_cross_robot"]["gap_mm"] >= 0
    clearance = report["clearance_lower_bounds_mm"]
    assert clearance["between_hands_involving_a_digit"] > report["contact_threshold_mm"]
    assert clearance["between_robots_involving_an_arm"] > 0.5
    assert clearance["self_or_environment"] > 0
    assert report["peak_joint_speed_fraction_of_limit"] <= 1.0
    assert np.all(scene._positions >= scene.plant.GetPositionLowerLimits() - 1e-7)
    assert np.all(scene._positions <= scene.plant.GetPositionUpperLimits() + 1e-7)
