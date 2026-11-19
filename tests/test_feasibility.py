import numpy as np

from dap.feasibility import (
    R_OURS_PARTNER,
    base_distance_ok,
    check_feasible,
    min_jerk_time,
    reachable_in_time,
)
from dap.scene import OURS, PARTNER, Scene, SceneLayout


def _rel_angle_deg(w):
    R_rel = w.X_W_ours.rotation().inverse() @ w.X_W_partner.rotation()
    E = R_rel.matrix().T @ R_OURS_PARTNER.matrix()
    return np.degrees(np.arccos(np.clip((np.trace(E) - 1) / 2, -1, 1)))


def test_nominal_layout_has_a_clean_dap():
    scene = Scene()
    w = check_feasible(scene, np.random.default_rng(0), target=[0.7, 0.0, 0.6])
    assert w is not None and w.quality == "tight"
    scene.set_q(w.q)
    p_o, n_o = scene.palm(OURS)
    p_p, n_p = scene.palm(PARTNER)
    assert np.linalg.norm(p_o - p_p) < 0.01  # faces touch
    assert np.dot(n_o, n_p) < -0.999  # palms anti-parallel
    assert _rel_angle_deg(w) < 2.5  # fingers point opposite ways
    assert scene.clearance() > 0.009  # nothing else collides
    assert np.all(np.abs(np.asarray(p_o) - [0.7, 0, 0.6]) < 0.031)


def test_far_layout_is_infeasible():
    layout = SceneLayout(distance=2.4)
    assert not base_distance_ok(layout)
    assert check_feasible(Scene(layout), np.random.default_rng(0), restarts=2) is None


def test_reach_time_check():
    v = np.ones(7)
    q0, q1 = np.zeros(7), np.r_[1.0, np.zeros(6)]
    assert abs(min_jerk_time(q0, q1, v) - 1.875) < 1e-9
    assert reachable_in_time(q0, q1, v, 2.1, speed_margin=1.0)
    assert not reachable_in_time(q0, q1, v, 1.8, speed_margin=1.0)
