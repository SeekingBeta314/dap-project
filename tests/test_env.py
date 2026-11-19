import warnings

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import dap.envs  # noqa: F401  (registers dap/PalmMeet-v0)
from dap.envs import ENV_ID, STATE_DIM


def _check(**kw):
    env = gym.make(ENV_ID, **kw)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_env(env.unwrapped, skip_render_check=True)
    assert not caught, [str(w.message) for w in caught]
    return env


def test_state_env_passes_checker_and_steps():
    env = _check(obs_mode="state")
    obs, info = env.reset(seed=0)
    assert obs["state"].shape == (STATE_DIM,)
    assert env.observation_space.contains(obs)
    # Zero action: our arm holds still, the partner keeps coming.
    q0 = obs["state"][:7].copy()
    gaps = []
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(np.zeros(7, np.float32))
        gaps.append(info["palm_dist"])
        assert reward == 0.0  # no reward until one is designed
        if terminated or truncated:
            break
    assert info["t"] > 0
    assert len(set(np.round(gaps, 6))) > 1
    assert np.allclose(obs["state"][:7], q0)


def test_env_runs_to_the_end_with_the_scorers_rule():
    env = gym.make(ENV_ID).unwrapped
    env.reset(seed=3)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.zeros(7, np.float32))
    # Holding still: the partner never touches a resting arm.
    assert truncated and info["event"] is None and not info["success"]
    assert info["t"] >= info["t_meet"]


def test_bad_actions_are_ignored_and_counted():
    env = gym.make(ENV_ID).unwrapped
    obs, _ = env.reset(seed=4)
    obs2, _, _, _, info = env.step(np.full(7, np.nan, np.float32))
    assert np.allclose(obs["state"][:7], obs2["state"][:7])
    assert info["n_bad_actions"] == 1
    # Full-speed action moves each joint at most v_max * dt per inner step.
    q_before = env.q.copy()
    env.step(np.ones(7, np.float32))
    dt_total = env.n_sub * env.sim_dt
    assert np.all(np.abs(env.q - q_before) <= env.v_max * dt_total + 1e-9)


def test_depth_env_passes_checker():
    env = _check(obs_mode="state+depth", camera_scale=0.125)
    obs, _ = env.reset(seed=1)
    for _ in range(3):
        obs, *_ = env.step(env.action_space.sample())
    d = obs["depth"]
    assert d.shape == env.observation_space["depth"].shape
    assert d.dtype == np.float32 and (d > 0).any()


def test_reward_hook():
    env = gym.make(ENV_ID, reward_fn=lambda info: -info["palm_dist"])
    env.reset(seed=2)
    _, reward, *_ = env.step(np.zeros(7, np.float32))
    assert reward < 0
