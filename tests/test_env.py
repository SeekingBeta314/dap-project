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
    gaps = []
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(np.zeros(8, np.float32))
        gaps.append(info["palm_gap"])
        assert reward == 0.0  # no reward until one is designed
        if terminated or truncated:
            break
    assert info["t"] > 0
    assert len(set(np.round(gaps, 6))) > 1


def test_depth_env_passes_checker():
    env = _check(obs_mode="state+depth", camera_scale=0.125)
    obs, _ = env.reset(seed=1)
    for _ in range(3):
        obs, *_ = env.step(env.action_space.sample())
    d = obs["depth"]
    assert d.shape == env.observation_space["depth"].shape
    assert d.dtype == np.float32 and (d > 0).any()


def test_reward_hook():
    env = gym.make(ENV_ID, reward_fn=lambda info: -info["palm_gap"])
    env.reset(seed=2)
    _, reward, *_ = env.step(np.zeros(8, np.float32))
    assert reward < 0
