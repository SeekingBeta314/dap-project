"""Measure a cupped-hand dap from Drake's playback and collision geometry."""

import argparse
import json
from itertools import combinations

import numpy as np

from dap.handshake import DIGITS, SIDES, HandshakeScene

# The rigid model has no flesh compliance. A pair within 0.2 mm counts as touching.
CONTACT_GAP_M = 0.0002
SAMPLE_PERIOD_S = 0.001


def _geometry_labels(scene):
    inspector = scene.query_object().inspector()
    return {
        gid: (side, body, inspector.GetName(gid).split("::")[-1])
        for gid, (side, body) in scene.owner.items()
    }


def _pairs(scene, labels, radius):
    world = ("world", "static", "geometry")
    for pair in scene.query_object().ComputeSignedDistancePairwiseClosestPoints(radius):
        a, b = labels.get(pair.id_A, world), labels.get(pair.id_B, world)
        if b[0] == SIDES[0]:
            a, b = b, a
        yield (pair.distance, a, b)


def _nearest_cross_pair(scene, time, labels):
    scene.set_time(time)
    return min(
        row for row in _pairs(scene, labels, np.inf) if (row[1][0], row[2][0]) == SIDES
    )


def _contact_record(time, pair):
    return {
        "time_s": time,
        "gap_mm": 1000 * pair[0],
        "surfaces": ["/".join(surface) for surface in pair[1:]],
    }


def _impact_patches(scene, labels):
    """Locate rim contacts in each palm's frame, merging duplicate witnesses at seams."""
    query = scene.query_object()
    inspector = query.inspector()
    contacts = []
    for pair in sorted(
        query.ComputeSignedDistancePairwiseClosestPoints(CONTACT_GAP_M),
        key=lambda p: p.distance,
    ):
        witnesses = []
        for gid, point in ((pair.id_A, pair.p_ACa), (pair.id_B, pair.p_BCb)):
            if gid in labels and labels[gid][1] == "palm":
                witnesses.append((labels[gid], inspector.GetPoseInFrame(gid) @ point))
        if len(witnesses) != 2 or witnesses[0][0][0] == witnesses[1][0][0]:
            continue
        witnesses.sort(key=lambda x: SIDES.index(x[0][0]))
        points_mm = 1000 * np.array([point for _, point in witnesses])
        # Adjacent convex mesh sections can each report the same physical point.
        if any(
            np.linalg.norm(points_mm - points, axis=1).max() < 1.0
            for points, _ in contacts
        ):
            continue
        contacts.append(
            (
                points_mm,
                {
                    "surfaces": ["/".join(surface) for surface, _ in witnesses],
                    "witness_in_each_palm_mm": points_mm.tolist(),
                    "gap_mm": 1000 * pair.distance,
                },
            )
        )
    # The smaller of the two spans requires the sites to be distinct on both palms.
    span, outer = 0.0, contacts
    for a, b in combinations(contacts, 2):
        separation = np.linalg.norm(a[0] - b[0], axis=1).min()
        if separation > span:
            span, outer = separation, (a, b)
    return {"max_span_mm": span, "outer_contacts": [record for _, record in outer]}


def _hand_shape(scene, side):
    """Height of each fingertip above its finger base along the palm's facing normal."""
    p, ctx = scene.plant, scene.plant_context
    frame = scene.palm_frames[side]
    model = scene.hands[side]
    inspector = scene.query_object().inspector()
    heights = {}
    for digit in DIGITS:
        if digit == "thumb":
            continue
        base = p.GetBodyByName(f"{digit}_proximal", model)
        tip = p.GetBodyByName(f"{digit}_distal", model)
        (gid,) = p.GetCollisionGeometriesForBody(tip)
        capsule = inspector.GetShape(gid)
        X_PC = p.CalcRelativeTransform(
            ctx, frame, tip.body_frame()
        ) @ inspector.GetPoseInFrame(gid)
        fingertip = X_PC @ [0, 0, capsule.length() / 2 + capsule.radius()]
        base_z = p.CalcRelativeTransform(ctx, frame, base.body_frame()).translation()[2]
        heights[digit] = 1000 * (fingertip[2] - base_z)
    return heights


def _pocket_at_impact(scene):
    """Distances from the midpoint of the two palm frame origins to each hand."""
    center = scene.palm_centers().mean(axis=0)
    query = scene.query_object()
    distances = query.ComputeSignedDistanceToPoint(center)
    clearances, palm_xy = {}, {}
    normals = []
    for side, palm in scene.palm_frames.items():
        pose = palm.CalcPoseInWorld(scene.plant_context)
        normals.append(pose.rotation().matrix()[:, 2])
        local_center = pose.inverse() @ center
        palm_xy[side] = (1000 * local_center[:2]).tolist()
        clearances[side] = 1000 * min(
            row.distance
            for row in distances
            if row.id_G in scene.owner and scene.owner[row.id_G][0] == side
        )
    facing_angle = np.arccos(np.clip(normals[0] @ normals[1], -1.0, 1.0))
    return {
        "angle_between_palm_normals_deg": np.rad2deg(facing_angle),
        "center_in_palm_xy_mm": palm_xy,
        "clearance_to_each_robot_mm": clearances,
    }


def audit_handshake(scene, step=SAMPLE_PERIOD_S):
    """Audit all trajectory knots and a uniform time grid, including the rebound."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be a positive finite number of seconds")
    times = np.unique(
        np.r_[np.arange(0.0, scene.duration, step), scene._times, scene.impact_time]
    )
    labels = _geometry_labels(scene)
    hand_indices = np.concatenate(
        [idx for side in SIDES for idx in scene.digit_idx[side].values()]
    )
    hand_positions, near = [], []
    closest, closest_time = None, None
    # A radius-limited distance query produces a conservative clearance floor for
    # classes of geometry that never come close, and exact gaps near contact.
    radius = 0.025
    arm_gap = digit_gap = other_gap = radius
    for t in times:
        scene.set_time(t)
        hand_positions.append(
            scene.plant.GetPositions(scene.plant_context)[hand_indices]
        )
        frame_nearest = None
        for row in _pairs(scene, labels, radius):
            d, a, b = row
            if (a[0], b[0]) != SIDES:
                other_gap = min(d, other_gap)
                continue
            if frame_nearest is None or row < frame_nearest:
                frame_nearest = row
            if a[1].startswith("iiwa_link_") or b[1].startswith("iiwa_link_"):
                arm_gap = min(d, arm_gap)
            elif (a[1], b[1]) != ("palm", "palm"):
                digit_gap = min(d, digit_gap)
        touch = frame_nearest is not None and frame_nearest[0] <= CONTACT_GAP_M
        near.append(frame_nearest if touch else None)
        if frame_nearest is not None and (closest is None or frame_nearest < closest):
            closest, closest_time = frame_nearest, t

    # Use the samples just outside each encounter to bound its duration. An
    # impact at a single knot can be shorter than the sampling interval.
    hit = np.array([row is not None for row in near])
    starts = np.flatnonzero(hit & ~np.r_[False, hit[:-1]])
    ends = np.flatnonzero(hit & ~np.r_[hit[1:], False])
    episodes = []
    for i, j in zip(starts, ends):
        episodes.append(
            {
                "first": _contact_record(times[i], near[i]),
                "last_sample_s": times[j],
                "duration_upper_bound_s": times[min(j + 1, len(times) - 1)]
                - times[max(i - 1, 0)],
            }
        )
    # Measure approach and rebound from relative changes of signed distance.
    # Instantaneous velocity at a rigid impact can be discontinuous.
    impact = scene.impact_time
    window = min(0.040, impact, scene.duration - impact)
    before = _nearest_cross_pair(scene, impact - window, labels)[0]
    after = _nearest_cross_pair(scene, impact + window, labels)[0]
    start = _nearest_cross_pair(scene, 0.0, labels)[0]
    end = _nearest_cross_pair(scene, scene.duration, labels)[0]
    meeting = _nearest_cross_pair(scene, impact, labels)
    speeds = np.diff(scene._positions, axis=0) / np.diff(scene._times)[:, None]
    return {
        "duration_s": scene.duration,
        "samples": len(times),
        "sample_period_s": step,
        "contact_threshold_mm": 1000 * CONTACT_GAP_M,
        "contact_episodes": episodes,
        "at_impact": _contact_record(impact, meeting),
        "palm_patches_at_impact": _impact_patches(scene, labels),
        "closest_cross_robot": _contact_record(closest_time, closest)
        if closest
        else None,
        "start_gap_mm": 1000 * start,
        "end_gap_mm": 1000 * end,
        "approach_rebound_window_s": window,
        "mean_closing_speed_m_s": (before - meeting[0]) / window,
        "mean_rebound_speed_m_s": (after - meeting[0]) / window,
        "pocket_at_impact": _pocket_at_impact(scene),
        "fingertip_rise_above_knuckle_mm": {
            side: _hand_shape(scene, side) for side in SIDES
        },
        "max_finger_or_thumb_joint_excursion_deg": float(
            np.rad2deg(np.ptp(hand_positions, axis=0)).max()
        ),
        "clearance_lower_bounds_mm": {
            "between_robots_involving_an_arm": 1000 * arm_gap,
            "between_hands_involving_a_digit": 1000 * digit_gap,
            "self_or_environment": 1000 * other_gap,
        },
        "peak_joint_speed_fraction_of_limit": float(
            np.max(np.abs(speeds) / scene.plant.GetVelocityUpperLimits())
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        type=float,
        default=SAMPLE_PERIOD_S,
        help="sampling interval in seconds",
    )
    args = parser.parse_args()
    print(json.dumps(audit_handshake(HandshakeScene(), args.step), indent=2))


if __name__ == "__main__":
    main()
