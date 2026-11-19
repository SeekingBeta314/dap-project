import numpy as np

from dap.feasibility import min_jerk_time
from dap.scenario import SCENARIO_PRESETS, sample_scenario
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
    # It is at rest at the start.
    assert np.linalg.norm(m.v(0.0)) < 0.05


def test_partner_respects_joint_velocity_limits(scenario):
    scene, m = scenario.scene, scenario.partner
    v_max = scene.v_max[scene.idx[PARTNER]]
    qd = m.q_traj.derivative(1)
    for t in np.arange(0, m.t_end, 0.01):
        assert np.all(np.abs(qd.value(t).ravel()) <= v_max + 1e-9)


def test_our_start_is_valid(scenario):
    scene = scenario.scene
    q_saved = scene.q()
    scene.set_ours(scenario.q_ours0)
    scene.set_partner(scenario.partner.q(0.0))
    assert scene.clearance() >= 0.01
    scene.set_q(q_saved)
    v = scene.v_max[scene.idx[OURS]]
    assert min_jerk_time(scenario.q_ours0, scenario.q_ours_meet, 0.9 * v) <= scenario.t_meet


def test_presets_sample_varied_scenarios():
    for name in ("fixed", "wide"):
        a = sample_scenario(np.random.default_rng(1), SCENARIO_PRESETS[name])
        b = sample_scenario(np.random.default_rng(2), SCENARIO_PRESETS[name])
        assert not np.allclose(a.partner.P[0], b.partner.P[0])
        assert not np.allclose(a.q_ours0, b.q_ours0)
        if name == "wide":
            assert a.layout != b.layout


def test_speed_scale_shortens_the_approach():
    cfg = SCENARIO_PRESETS["nominal"]
    fast = sample_scenario(np.random.default_rng(4), cfg, speed_scale=1.5)
    assert fast.t_meet <= cfg.path.t_meet[1] / 1.5 + 1e-9
