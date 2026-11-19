import numpy as np

from dap.feasibility import min_jerk_time
from dap.partner import MIN_JERK_PEAK, PathConfig
from dap.scenario import SCENARIO_PRESETS, _sample, sample_scenario
from dap.scene import OURS, P_PALM_FACE, PARTNER


def test_partner_reaches_the_certified_meeting(scenario):
    m, w = scenario.partner, scenario.witness
    # The partner palm face is at the witness's partner palm face at t_meet...
    assert np.linalg.norm(m.p(m.t_meet) - w.X_W_partner @ P_PALM_FACE) < 0.005
    assert np.dot(m.normal(m.t_meet), w.X_W_partner.rotation().matrix()[:, 2]) > 0.999
    # ...and still moving toward us (overshoot), along its palm normal.
    v = m.v(m.t_meet)
    assert np.linalg.norm(v) > 0.02
    assert np.dot(v, m.normal(m.t_meet)) > 0
    # It is at rest at the start and at the end, and holds after the end.
    assert np.linalg.norm(m.v(0.0)) < 0.05
    assert np.allclose(m.q(m.t_end), m.q(m.t_end + 1.0))


def test_partner_is_a_real_iiwa(scenario):
    """Joint positions and velocities stay within the iiwa's limits along
    the whole trajectory (sampled finer than the ground-truth grid)."""
    scene, m = scenario.scene, scenario.partner
    idx = scene.idx[PARTNER]
    v_max, lo, hi = scene.v_max[idx], scene.q_lower[idx], scene.q_upper[idx]
    qd = m.q_traj.derivative(1)
    for t in np.arange(0, m.t_end, 0.002):
        assert np.all(np.abs(qd.value(t).ravel()) <= v_max + 1e-9)
        q = m.q(t)
        assert np.all(q >= lo) and np.all(q <= hi)
    assert scenario.partner_joint_speed_ratio <= 1.0


def test_partner_speed_is_the_drawn_speed():
    rng_speeds, realized = [], []
    for seed in range(3):
        sc = sample_scenario(np.random.default_rng([seed, 0]), SCENARIO_PRESETS["nominal"])
        lo, hi = PathConfig().peak_speed
        assert lo <= sc.target_peak_speed <= hi
        rng_speeds.append(sc.target_peak_speed)
        realized.append(sc.partner.peak_speed)
    # The partner tracks the min-jerk reference closely, so the realized
    # palm peak speed is the drawn one within about 5% (slightly higher:
    # tracking error and the 10 ms finite differences).
    assert np.allclose(realized, rng_speeds, rtol=0.07)


def test_duration_follows_from_speed_and_path_length():
    cfg = SCENARIO_PRESETS["nominal"]
    slow = _sample(np.random.default_rng(5), cfg, 0.2, None).partner.reference
    fast = _sample(np.random.default_rng(5), cfg, 0.8, None).partner.reference
    for ref, v in ((slow, 0.2), (fast, 0.8)):
        assert abs(ref.T_total - MIN_JERK_PEAK * ref.L_total / v) < 1e-9
        assert abs(ref.peak_speed() / v - 1) < 0.01
    assert slow.t_meet > fast.t_meet


def test_our_start_is_valid(scenario):
    scene = scenario.scene
    q_saved = scene.q()
    scene.set_ours(scenario.q_ours0)
    scene.set_partner(scenario.partner.q(0.0))
    assert scene.clearance() >= 0.01
    assert scene.palm_palm_distance() >= 0.01
    scene.set_q(q_saved)
    # Reachability is recorded, not filtered on.
    v = scene.v_max[scene.idx[OURS]]
    reach = min_jerk_time(scenario.q_ours0, scenario.q_ours_meet, 0.9 * v) <= scenario.t_meet
    assert reach == scenario.reachable_in_time


def test_presets_sample_varied_scenarios():
    for name in ("fixed", "wide"):
        a = sample_scenario(np.random.default_rng(1), SCENARIO_PRESETS[name])
        b = sample_scenario(np.random.default_rng(2), SCENARIO_PRESETS[name])
        assert not np.allclose(a.partner.P[0], b.partner.P[0])
        assert not np.allclose(a.q_ours0, b.q_ours0)
        assert a.target_peak_speed != b.target_peak_speed
        if name == "wide":
            assert a.layout != b.layout


def test_seed_fixes_the_scenario():
    cfg = SCENARIO_PRESETS["nominal"]
    a = sample_scenario(np.random.default_rng([7, 0]), cfg)
    b = sample_scenario(np.random.default_rng([7, 0]), cfg)
    assert a.layout == b.layout and a.target_peak_speed == b.target_peak_speed
    assert np.allclose(a.q_ours0, b.q_ours0)
    assert np.allclose(a.partner.p(0.5), b.partner.p(0.5))


def test_partner_never_touches_our_resting_arm(scenario):
    scene, m = scenario.scene, scenario.partner
    q_saved = scene.q()
    scene.set_ours(scenario.q_ours0)
    for t in m.times:
        scene.set_partner(m.q(t))
        assert scene.min_arm_distance(max_distance=0.05) >= 0.0
        assert scene.palm_palm_distance() >= 0.0
    scene.set_q(q_saved)
