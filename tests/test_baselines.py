"""End-to-end checks of the trial loop with scripted controllers: the task
is solvable (the oracle daps), not trivial (holding still never does), and
the harness, not the controller, decides what the arm does."""

import numpy as np

from dap.baselines import hold_factory, oracle_factory
from dap.experiment import Condition, ExperimentConfig
from dap.trial import limit_command, run_trial

CFG = ExperimentConfig(name="t")
SEEDS = range(4)
COND = Condition("oracle", 1.0)


def _run(factory, seeds=SEEDS, **kw):
    return [run_trial(CFG, COND, s, controller_factory=factory, **kw) for s in seeds]


def test_oracle_daps():
    # Seeds 0-3 include scenarios flagged not reachable_in_time (the flag is
    # conservative); the oracle daps in all of them.
    rs = _run(oracle_factory)
    assert any(not r.reachable_in_time for r in rs)
    for r in rs:
        assert r.error == "" and r.success, r
        assert r.rot_err < 5 and r.pos_err < 0.01
        # The palms touch at the partner's true meeting time, give or take a step.
        assert abs(r.time_err) <= 0.01


def test_servo_daps_with_the_stand_in_sensor_and_with_ground_truth():
    from dap.baselines import servo_factory, servo_gt_factory

    for factory in (servo_factory, servo_gt_factory):
        for r in _run(factory, seeds=range(3)):
            assert r.error == "" and r.success, r
            assert r.rot_err < 5


def test_blind_ignores_the_estimator_and_holds_a_fixed_target():
    from dap.baselines import blind_factory
    from dap.scenario import SCENARIO_PRESETS, sample_scenario
    from dap.experiment import trial_rngs

    sc = sample_scenario(trial_rngs(0)["scenario"], SCENARIO_PRESETS[CFG.scenario])
    blind = blind_factory(CFG, COND, sc)
    a = blind.partner_fn(0.0, None)  # never touches the estimator
    b = blind.partner_fn(1.0, None)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    r = run_trial(CFG, COND, 0, controller_factory=blind_factory)
    assert r.error == ""


def test_holding_still_never_daps():
    for r in _run(hold_factory):
        assert r.outcome == "timeout" and not r.success and not r.collision


def test_step_over_in_the_real_scene_is_a_contact():
    """A fast approach that lands 1.5 cm through the partner's palm in one
    step still scores as the touch, although on that sample the finger
    plates penetrate the other palm."""
    from pydrake.all import RigidTransform

    from dap.feasibility import check_feasible
    from dap.ik import arm_ik
    from dap.scene import OURS, Scene
    from dap.scoring import EventMonitor

    scene = Scene()
    w = check_feasible(scene, np.random.default_rng(0), target=[0.7, 0.0, 0.6])
    scene.set_q(w.q)
    X = scene.palm_pose(OURS)
    n = X.rotation().matrix()[:, 2]
    mon = EventMonitor(CFG.tolerances.touch_dist)
    for shift, expect in [(-0.01, None), (0.015, "contact")]:
        q = arm_ik(scene, OURS, RigidTransform(X.rotation(), X.translation() + shift * n),
                   w.q_ours, w.q_ours, 0.001, 1.0)
        scene.set_ours(q)
        m = scene.measure()
        if shift > 0:
            assert m["clearance"] < 0  # something besides the palms penetrates
        assert mon.step(m["palm_p"], m["palm_R"], m["partner_p"], m["partner_R"],
                        m["clearance"], m["palm_palm"]) == expect


class _ConstVel:
    """Commands a constant fraction of each joint's speed limit."""

    def __init__(self, a):
        self.a = np.asarray(a, float)

    def reset(self, model, q0):
        self.step_size = self.a * model.v_max * model.dt

    def step(self, t, q, estimator):
        return q + self.step_size


def test_env_ends_exactly_where_the_trial_ends():
    import gymnasium as gym

    import dap.envs  # noqa: F401

    env = gym.make("dap/PalmMeet-v0").unwrapped
    for seed, a in [(0, np.zeros(7)), (1, np.array([0.3, 0.6, 0, -0.5, 0, 0.4, 0])),
                    (2, np.array([-0.2, 0.8, 0.1, 0.6, 0, -0.3, 0]))]:
        r, log = run_trial(CFG, COND, seed, controller_factory=lambda *_: _ConstVel(a),
                           return_log=True)
        env.reset(options={"trial_seed": seed})
        assert np.allclose(env.q, log.q[0])
        done = False
        while not done:
            _, _, term, trunc, info = env.step(a.astype(np.float32))
            done = term or trunc
        assert (info["event"] or "timeout") == r.outcome, seed
        assert abs(env.t - log.t[-1]) < 1e-9, seed
        assert np.allclose(env.q, log.q[-1]), seed


def test_env_replays_an_oracle_dap_to_the_same_contact():
    """The oracle's logged motion, replayed as 100 Hz velocity actions,
    ends the episode with the same contact at the same time."""
    import gymnasium as gym

    import dap.envs  # noqa: F401

    r, log = run_trial(CFG, COND, 0, controller_factory=oracle_factory, return_log=True)
    assert r.outcome == "contact"
    env = gym.make("dap/PalmMeet-v0", control_hz=100.0).unwrapped
    env.reset(options={"trial_seed": 0})
    q = np.asarray(log.q)
    for k in range(len(q) - 1):
        a = (q[k + 1] - q[k]) / (env.v_max * env.sim_dt)
        _, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    assert term and info["event"] == "contact" and info["success"]
    assert k == len(q) - 2 and abs(env.t - r.t_event) < 1e-9
    assert abs(info["rot_err_deg"] - r.rot_err) < 1e-6


class _Teleport:
    """Asks for the far end of the joint range at once, then garbage."""

    def reset(self, model, q0):
        self.hi = model.q_upper
        self.k = 0

    def step(self, t, q, estimator):
        self.k += 1
        if self.k % 3 == 0:
            return np.full(7, np.nan)
        if self.k % 3 == 1:
            return np.zeros(3)  # wrong shape
        return self.hi


def test_arm_cannot_teleport_and_garbage_holds_still():
    r, log = run_trial(CFG, COND, 1, controller_factory=lambda *a: _Teleport(),
                       return_log=True)
    q, t = np.asarray(log.q), np.asarray(log.t)
    from dap.scene import OURS, Scene

    s = Scene()
    idx = s.idx[OURS]
    v_max, lo, hi = s.v_max[idx], s.q_lower[idx], s.q_upper[idx]
    dq = np.abs(np.diff(q, axis=0))
    assert np.all(dq <= v_max * np.diff(t)[:, None] + 1e-12)
    assert np.all(q >= lo - 1e-12) and np.all(q <= hi + 1e-12)
    assert r.n_bad_commands >= 2 * (r.n_steps // 3) - 1
    # Steps after a bad command did not move the arm.
    assert np.allclose(q[1], q[0]) and np.allclose(q[3], q[2])


def test_limit_command():
    q = np.zeros(7)
    v = np.ones(7)
    lo, hi = -np.ones(7), np.ones(7)
    out, ok = limit_command(np.full(7, 5.0), q, v, 0.01, lo, hi)
    assert ok and np.allclose(out, 0.01)
    out, ok = limit_command(np.full(7, np.inf), q, v, 0.01, lo, hi)
    assert not ok and np.allclose(out, q)
    out, ok = limit_command(None, q, v, 0.01, lo, hi)
    assert not ok
    out, ok = limit_command(np.full(7, 1.5), np.full(7, 0.995), v, 0.01, lo, hi)
    assert ok and np.allclose(out, 1.0)


def test_servo_converges_to_a_fixed_pose(scenario):
    """The resolved-rate step itself: with a still "partner" whose
    antiparallel pose is a reachable palm pose, the palm face gets there."""
    from dap.baselines import ROT_Y_PI, DiffIKServo
    from dap.planner import controller_model
    from dap.scene import OURS, P_PALM_FACE

    scene, dt = scenario.scene, 0.01
    saved = scene.q()
    idx = scene.idx[OURS]
    v_max, lo, hi = scene.v_max[idx], scene.q_lower[idx], scene.q_upper[idx]
    q = scenario.q_ours0.copy()
    scene.set_ours(np.clip(q + 0.3, lo, hi))
    X = scene.palm_pose(OURS)
    p_goal, R_goal = X @ P_PALM_FACE, X.rotation().matrix()
    servo = DiffIKServo(lambda t, est: (p_goal, np.zeros(3), R_goal @ ROT_Y_PI), standoff=None)
    servo.reset(controller_model(scene, dt), q)
    for k in range(300):
        q, ok = limit_command(servo.step(k * dt, q, None), q, v_max, dt, lo, hi)
        assert ok
    scene.set_ours(q)
    X = scene.palm_pose(OURS)
    scene.set_q(saved)
    assert np.linalg.norm(X @ P_PALM_FACE - p_goal) < 1e-3
    cos = (np.trace(X.rotation().matrix().T @ R_goal) - 1) / 2
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))) < 0.5
