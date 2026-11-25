"""The scoring rule on synthetic logs: geometry, events, success, loss, and
that the live monitor and the offline evaluator agree."""

import numpy as np
import pytest

from dap.scoring import (
    EventMonitor,
    LossWeights,
    Tolerances,
    contact_geometry,
    dap_loss,
    evaluate,
)

TOL = Tolerances()

# Partner palm at x = 0.7 facing -x (toward us), fingers pointing down (-z).
# Frame columns: x = fingers, y, z = normal.
R_PARTNER = np.array([[0, 0, -1.0], [0, -1.0, 0], [-1.0, 0, 0]]).T
P_PARTNER = np.array([0.7, 0.0, 0.6])


def rot_about(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    th = np.radians(deg)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


# A perfect dap: ours = partner rotated by pi about its y axis (fingers up,
# normal +x).
R_OURS = R_PARTNER @ rot_about([0, 1, 0], 180)


def _log(xs, R_o=R_OURS, offset=(0, 0, 0), clearance=0.05, palm_palm=None, dt=0.01):
    """Our palm moving along x toward the partner's. By default the palm-box
    distance is the face gap while the faces overlap in the plane (aligned
    boxes), and large otherwise."""
    n = len(xs)
    p = np.array([[x, 0, 0.6] for x in xs]) + np.asarray(offset, float)
    if palm_palm is None:
        gap = P_PARTNER[0] - p[:, 0]
        overlap = np.linalg.norm(p[:, 1:] - P_PARTNER[1:], axis=1) < 0.10
        palm_palm = np.where(overlap, gap, 0.05)
    return dict(
        t=np.arange(n) * dt, q=np.zeros((n, 7)), palm_p=p,
        palm_R=np.repeat(np.asarray(R_o)[None], n, 0),
        partner_p=np.repeat(P_PARTNER[None], n, 0),
        partner_R=np.repeat(R_PARTNER[None], n, 0),
        partner_v=np.zeros((n, 3)),
        clearance=np.full(n, clearance, float),
        palm_palm=np.broadcast_to(np.asarray(palm_palm, float), (n,)).copy(),
        arm_dist=np.full(n, 0.2),
    )


def _approach(n=40, x_end=0.7):
    return list(np.linspace(0.5, x_end, n))


def test_geometry_of_a_perfect_dap_is_zero():
    g = contact_geometry(P_PARTNER, R_OURS, P_PARTNER, R_PARTNER)
    assert abs(g["gap"]) < 1e-12 and g["in_plane"] < 1e-12
    for k in ("rot_err", "angle_err", "finger_err"):
        assert g[k] < 1e-5, k


def test_geometry_decomposes_rotation_errors():
    # Spin 30 deg about the palm normal: palms still parallel, fingers off.
    g = contact_geometry(P_PARTNER, R_OURS @ rot_about([0, 0, 1], 30), P_PARTNER, R_PARTNER)
    assert g["angle_err"] < 1e-5 and abs(g["finger_err"] - 30) < 1e-5
    assert abs(g["rot_err"] - 30) < 1e-5
    # Tilt 20 deg about the palm's y axis: palms not parallel.
    g = contact_geometry(P_PARTNER, R_OURS @ rot_about([0, 1, 0], 20), P_PARTNER, R_PARTNER)
    assert abs(g["angle_err"] - 20) < 1e-5 and abs(g["rot_err"] - 20) < 1e-5
    # Same hand orientation as the partner (not antiparallel): 180 deg off.
    g = contact_geometry(P_PARTNER, R_PARTNER, P_PARTNER, R_PARTNER)
    assert abs(g["rot_err"] - 180) < 1e-4


def test_geometry_is_vectorized():
    Rs = np.stack([R_OURS, R_OURS @ rot_about([0, 0, 1], 30)])
    P = np.stack([P_PARTNER, P_PARTNER + [-0.1, 0.02, 0]])
    g = contact_geometry(P, Rs, np.stack([P_PARTNER] * 2), np.stack([R_PARTNER] * 2))
    assert g["rot_err"].shape == (2,)
    assert abs(g["gap"][1] - 0.1) < 1e-12 and abs(g["in_plane"][1] - 0.02) < 1e-12


def test_clean_dap_succeeds():
    r = evaluate(_log(_approach()), t_meet_true=0.39, tol=TOL)
    assert r["outcome"] == "contact" and r["success"]
    assert r["rot_err"] < 1e-5 and r["pos_err"] < 1e-9
    assert -0.01 <= r["gap"] <= TOL.touch_dist
    assert r["rel_speed"] > 0  # we were still moving at contact


def test_misaligned_touch_is_a_contact_but_not_a_success():
    r = evaluate(_log(_approach(), R_o=R_OURS @ rot_about([0, 0, 1], 40)), 0.39, TOL)
    assert r["outcome"] == "contact" and not r["success"]
    assert abs(r["finger_err"] - 40) < 1e-4


def test_position_and_timing_are_scores_not_success_criteria():
    # 4 cm off-center (still overlapping) and early: a success with errors.
    r = evaluate(_log(_approach(), offset=(0, 0.04, 0)), t_meet_true=1.5, tol=TOL)
    assert r["success"] and abs(r["pos_err"] - 0.04) < 1e-9 and r["time_err"] < -1.0


def test_hovering_close_is_not_a_touch():
    # Stops 5 mm in front of the partner's palm and stays there.
    r = evaluate(_log(list(np.linspace(0.5, 0.695, 20)) + [0.695] * 20), 0.2, TOL)
    assert r["outcome"] == "timeout" and abs(r["min_palm_dist"] - 0.005) < 1e-9


def test_no_overlap_is_no_contact():
    r = evaluate(_log(_approach(), offset=(0, 0.08, 0)), 0.39, TOL)
    assert r["outcome"] == "timeout" and not r["contact"]
    assert abs(r["min_palm_dist"] - 0.08) < 1e-9


def test_fast_step_through_the_plane_still_counts():
    # 2 cm in front, then 1.5 cm behind in one step; in the real scene the
    # finger plates would penetrate the other palm on that same sample.
    log = _log([0.5, 0.68, 0.715, 0.72])
    log["clearance"][2:] = -0.004
    r = evaluate(log, 0.02, TOL)
    assert r["contact"] and abs(r["t_event"] - 0.02) < 1e-12


def test_contact_from_behind_is_not_a_dap():
    # Our palm starts behind the partner's palm plane and moves away from it
    # through the contact band: never "from the front".
    r = evaluate(_log([0.72, 0.705, 0.69]), 0.0, TOL)
    assert not r["contact"]


def test_collision_ends_the_trial_and_fails():
    log = _log(_approach())
    log["clearance"][10:] = -0.001
    r = evaluate(log, 0.39, TOL)
    assert r["outcome"] == "collision" and not r["success"]
    assert abs(r["t_event"] - 0.10) < 1e-12
    # Palm boxes overlapping without a valid contact is also a collision.
    log = _log(_approach(), offset=(0, 0.08, 0))
    log["palm_palm"][20:] = -0.002
    assert evaluate(log, 0.39, TOL)["outcome"] == "collision"
    # A collision after the contact never happens: the trial already ended.
    log = _log(_approach())
    log["clearance"][-1] = -0.05
    assert evaluate(log, 0.39, TOL)["success"]


def test_live_monitor_matches_offline_evaluation():
    rng = np.random.default_rng(0)
    outcomes = []
    for _ in range(300):
        # Start in front or (sometimes) behind the plane; mostly clear arms.
        xs = np.cumsum(rng.uniform(-0.01, 0.03, 30)) + rng.uniform(0.15, 0.72)
        log = _log(list(xs), offset=(0, rng.uniform(-0.07, 0.07), 0),
                   R_o=R_OURS @ rot_about(rng.normal(size=3), rng.uniform(0, 60)))
        log["clearance"] = np.where(rng.random(30) < 0.03, -0.002, 0.05)
        log["palm_palm"] = log["palm_palm"] + rng.normal(0, 0.002, 30)
        mon = EventMonitor(TOL.touch_dist)
        live = None
        for k in range(30):
            live = mon.step(log["palm_p"][k], log["palm_R"][k], log["partner_p"][k],
                            log["partner_R"][k], log["clearance"][k], log["palm_palm"][k])
            if live is not None:
                break
        r = evaluate(log, 0.2, TOL)
        assert r["outcome"] == (live or "timeout")
        outcomes.append(r["outcome"])
        if live is not None:
            assert abs(r["t_event"] - log["t"][k]) < 1e-12
    # Every outcome is well represented, so the agreement means something.
    for o in ("contact", "collision", "timeout"):
        assert outcomes.count(o) >= 30, (o, outcomes.count(o))


def test_loss_is_vectorized_and_orders_outcomes():
    ok = evaluate(_log(_approach()), 0.39, TOL)
    late = evaluate(_log(_approach()), 0.0, TOL)  # success, 0.39 s early-vs-late
    bad = evaluate(_log(_approach(), R_o=R_OURS @ rot_about([0, 0, 1], 40)), 0.39, TOL)
    miss = evaluate(_log(_approach(), offset=(0, 0.08, 0)), 0.39, TOL)
    log = _log(_approach())
    log["clearance"][5:] = -0.01
    hit = evaluate(log, 0.39, TOL)
    rows = [ok, late, bad, miss, hit]
    assert [r["outcome"] for r in rows] == ["contact"] * 3 + ["timeout", "collision"]
    assert ok["success"] and late["success"] and not bad["success"]
    table = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    # Timing is unweighted by default; weighted, it must still respect the tiers.
    assert dap_loss(table)[0] == dap_loss(table)[1]
    w = LossWeights(w_time=1.0)
    loss = dap_loss(table, w)
    assert loss.shape == (5,) and np.all(np.isfinite(loss))
    # Tiers: every success < every failed contact < every miss < collision,
    # even a badly timed success vs. a well-timed failure.
    assert loss[0] < loss[1] < w.tier <= loss[2] < 2 * w.tier <= loss[3] <= 3 * w.tier
    assert loss[4] == w.collision_penalty


def test_loss_tiers_hold_for_extreme_values():
    w = LossWeights()
    table = {
        "outcome": np.array(["contact", "contact", "timeout", "timeout", "collision", "error"]),
        "success": np.array([True, False, False, False, False, False]),
        "pos_err": np.array([0.05, 0.0, np.nan, np.nan, np.nan, np.nan]),
        "rot_err": np.array([14.0, 16.0, np.nan, np.nan, np.nan, np.nan]),
        "time_err": np.array([3.0, 0.0, np.nan, np.nan, np.nan, np.nan]),
        "rel_speed": np.full(6, np.nan),
        "min_palm_dist": np.array([np.nan, np.nan, 0.0, 5.0, np.nan, np.nan]),
    }
    loss = dap_loss(table, w)
    assert loss[0] < w.tier <= loss[1]  # terrible success still beats a near-success failure
    assert loss[2] == 2 * w.tier and loss[3] == 3 * w.tier < loss[4]
    assert np.isnan(loss[5])  # crashed trials are not scored here
    with pytest.raises(ValueError):
        dap_loss(table, LossWeights(collision_penalty=12.0))


def test_pandas_table_works():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame([evaluate(_log(_approach()), 0.39, TOL)] * 3)
    assert np.allclose(dap_loss(df), dap_loss(df.iloc[:1])[0])
