"""Policies are scored exactly like trials: the same scenarios, the same end
of the episode, the same scores and covariates."""

import math

import numpy as np
import pytest

from dap.experiment import Condition, ExperimentConfig
from dap.policy_eval import make_eval_env, run_policy_trial
from dap.trial import run_trial

CFG = ExperimentConfig(name="t")
COND = Condition("student", 1.0)


class _ConstVelController:
    def __init__(self, a):
        self.a = np.asarray(a, float)

    def reset(self, model, q0):
        self.step_size = self.a * model.v_max * model.dt

    def step(self, t, q, estimator):
        return q + self.step_size


class _ConstVelPolicy:
    obs_mode = "pixels"

    def __init__(self, a):
        self.a = np.asarray(a, float)  # float32 would round the motion

    def __call__(self, obs):
        assert set(obs) == {"proprio", "depth"}
        return self.a


SCORES = ("outcome", "success", "contact", "collision", "t_event", "time_err", "gap", "pos_err",
          "rot_err", "min_palm_dist", "closest_rot_err", "min_arm_dist", "vmax_frac")
COVARIATES = ("target_peak_speed", "partner_peak_speed", "partner_speed_at_meet", "t_meet_true",
              "reachable_in_time", "partner_joint_speed_ratio", "layout_distance",
              "layout_lateral", "layout_yaw_deg", "layout_height", "scenario_attempts")


def _eq(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b or (isinstance(a, float) and abs(a - b) < 1e-9)


@pytest.mark.parametrize("seed,a", [
    (0, np.zeros(7)),
    (1, np.array([0.3, 0.6, 0, -0.5, 0, 0.4, 0])),
    (2, np.array([-0.2, 0.8, 0.1, 0.6, 0, -0.3, 0])),
])
def test_policy_trial_matches_run_trial(seed, a):
    """The same motion scored through the env and through run_trial gives
    the same row (constant velocity is the same motion at 20 and 100 Hz)."""
    r = run_trial(CFG, COND, seed, controller_factory=lambda *_: _ConstVelController(a))
    p = run_policy_trial(CFG, COND, seed, _ConstVelPolicy(a), env_kwargs={"camera_scale": 0.125})
    for k in SCORES + COVARIATES:
        assert _eq(getattr(p, k), getattr(r, k)), (k, getattr(p, k), getattr(r, k))
    assert p.planner == "student" and not p.error


class _OraclePolicy:
    """Privileged reference: velocity actions that track the oracle's
    joint-space min-jerk path (it reads the scenario, so it is not a method)."""

    obs_mode = "privileged"

    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        from dap.partner import min_jerk

        env = self.env
        sc = env.scenario
        dt = env.n_sub * env.sim_dt
        s = min_jerk((env.t + dt) / sc.partner.t_meet)
        q_next = sc.q_ours0 + (sc.q_ours_meet - sc.q_ours0) * s
        return np.clip((q_next - env.q) / (env.v_max * dt), -1, 1)


def test_privileged_reference_daps_and_must_be_asked_for():
    env = make_eval_env(CFG, COND, "privileged")
    policy = _OraclePolicy(env)
    with pytest.raises(ValueError, match="privileged"):
        run_policy_trial(CFG, COND, 0, policy, env=env)
    r = run_policy_trial(CFG, COND, 0, policy, env=env, allow_privileged=True)
    assert r.outcome == "contact" and r.success


class _Crashes:
    obs_mode = "pixels"

    def __call__(self, obs):
        raise RuntimeError("boom")


def test_policy_crash_is_a_trial_outcome_with_covariates():
    r = run_policy_trial(CFG, COND, 3, _Crashes(), env_kwargs={"camera_scale": 0.125})
    assert r.outcome == "error" and r.error_kind == "controller" and "boom" in r.error
    assert not math.isnan(r.target_peak_speed) and not math.isnan(r.layout_distance)


def test_eval_env_must_match_the_experiment():
    env = make_eval_env(CFG, COND, "pixels", camera_scale=0.125)
    env.t_extra = 0.5
    with pytest.raises(ValueError, match="t_extra"):
        run_policy_trial(CFG, COND, 0, _ConstVelPolicy(np.zeros(7)), env=env)
    other = make_eval_env(CFG, COND, "privileged")
    with pytest.raises(ValueError, match="obs_mode"):
        run_policy_trial(CFG, COND, 0, _ConstVelPolicy(np.zeros(7)), env=other)
