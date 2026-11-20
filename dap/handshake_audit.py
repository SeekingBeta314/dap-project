"""Reproduce signed clearances, contact order and joint speeds for the articulated dap."""

import argparse
import json

import numpy as np

from dap.handshake import DIGITS, OURS, PARTNER, HandshakeScene


def _distances(scene, maximum):
    """Drake candidate pairs, with the left robot first in a cross-robot pair."""
    environment = ("world", "static geometry")
    for pair in scene.query_object().ComputeSignedDistancePairwiseClosestPoints(
        maximum
    ):
        a = scene.owner.get(pair.id_A, environment)
        b = scene.owner.get(pair.id_B, environment)
        if b[0] == OURS:
            a, b = b, a
        yield pair.distance, a, b


def _cross_pairs(scene, time):
    scene.set_time(time)
    return [
        (d, a, b)
        for d, a, b in _distances(scene, 0.5)
        if a[0] == OURS and b[0] == PARTNER
    ]


def _record(time, row):
    distance, a, b = row
    return {
        "time_s": time,
        "gap_mm": 1000 * distance,
        "bodies": ["/".join(a), "/".join(b)],
    }


def audit_handshake(scene, step=0.005):
    """Sample the interpolated playback and all trajectory knots, reporting mm and s."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be a positive finite number of seconds")
    events = {
        name: scene.time_at_phase(authored)
        for name, authored in [
            ("palm_touch", 0.92),
            ("fingers_closed", 1.20),
            ("pull_apex", 1.42),
            ("opening_begins", 1.75),
            ("hands_open", 1.94),
            ("separated", 2.83),
            ("knuckle_bump", 3.23),
            ("recoil_ends", 3.99),
        ]
    }
    min_cross, min_arm, min_other = None, None, None
    first = None
    times = np.unique(np.r_[np.arange(0.0, scene.duration, step), scene._times])
    for t in times:
        scene.set_time(t)
        for row in _distances(scene, 0.025):
            distance, a, b = row
            if a[0] == OURS and b[0] == PARTNER:
                if min_cross is None or distance < min_cross[1][0]:
                    min_cross = (t, row)
                if first is None and distance <= 0.0002:
                    first = (t, row)
                arm_pair = a[1].startswith("iiwa_link_") or b[1].startswith(
                    "iiwa_link_"
                )
                if arm_pair and (min_arm is None or distance < min_arm[1][0]):
                    min_arm = (t, row)
            elif min_other is None or distance < min_other[1][0]:
                min_other = (t, row)
    touch = _cross_pairs(scene, events["palm_touch"])
    touch_center = scene.palm_centers().mean(axis=0)
    grasp = _cross_pairs(scene, events["pull_apex"])
    lift = scene.palm_centers().mean(axis=0) - touch_center
    # Gap from each individual fingertip to the opposing palm, including the thumb.
    wrap = {}
    for idx, side in enumerate((OURS, PARTNER), 1):
        wrap[side] = {
            digit: 1000
            * min(
                row[0]
                for row in grasp
                if row[idx][1] == digit + "_distal" and row[3 - idx][1] == "palm"
            )
            for digit in DIGITS
        }
    speeds = np.diff(scene._positions, axis=0) / np.diff(scene._times)[:, None]

    def measured(record):
        return _record(*record) if record else None

    return {
        "duration_s": scene.duration,
        "sample_period_s": step,
        "samples": len(times),
        "phase_times_s": events,
        "first_cross_robot_within_0.2_mm": measured(first),
        "palm_touch": _record(
            events["palm_touch"],
            min(row for row in touch if row[1][1] == row[2][1] == "palm"),
        ),
        "grasp_fingertip_to_opposite_palm_mm": wrap,
        "pull_displacement_xyz_mm": (lift * 1000).tolist(),
        "separated_hand_clearance_mm": 1000
        * min(_cross_pairs(scene, events["separated"]))[0],
        "knuckle_bump": _record(
            events["knuckle_bump"], min(_cross_pairs(scene, events["knuckle_bump"]))
        ),
        "final_hand_clearance_mm": 1000 * min(_cross_pairs(scene, scene.duration))[0],
        "minimum_cross_robot": measured(min_cross),
        "minimum_cross_robot_involving_arm": measured(min_arm),
        "minimum_self_or_environment": measured(min_other),
        "peak_joint_speed_fraction_of_limit": float(
            np.max(np.abs(speeds) / scene.plant.GetVelocityUpperLimits())
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step", type=float, default=0.005, help="sampling interval in seconds"
    )
    args = parser.parse_args()
    print(json.dumps(audit_handshake(HandshakeScene(), args.step), indent=2))


if __name__ == "__main__":
    main()
