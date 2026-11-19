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


# -- pixels / privileged modes -------------------------------------------------

from dap.envs import (  # noqa: E402
    FRAME_STACK,
    PRIVILEGED_DIM,
    PRIVILEGED_LAYOUT,
    PROPRIO_DIM,
    PROPRIO_LAYOUT,
)


def test_pixel_obs_has_no_partner_information():
    env = _check(obs_mode="pixels", camera_scale=0.125)
    assert set(env.observation_space.spaces) == {"proprio", "depth"}
    assert not any(name.startswith("partner") for name, _ in PROPRIO_LAYOUT)
    obs, info = env.reset(seed=0)
    assert obs["proprio"].shape == (PROPRIO_DIM,)
    assert obs["depth"].shape == (FRAME_STACK, *env.observation_space["depth"].shape[1:])
    # The proprio vector is the privileged vector minus the partner part.
    assert np.array_equal(obs["proprio"], info["privileged"][:PROPRIO_DIM])


def test_pixel_frame_stack_holds_the_newest_delivered_frames():
    env = gym.make(ENV_ID, obs_mode="pixels", camera_scale=0.125).unwrapped
    obs, _ = env.reset(seed=1)
    assert not obs["depth"].any()  # one frame of latency: nothing delivered at t = 0
    captures = []
    for _ in range(8):
        obs, *_ = env.step(np.zeros(7, np.float32))
        got = env.frames.latest(env.t)
        if got is not None and (not captures or captures[-1] != got[0]):
            captures.append(got[0])
    assert len(set(captures)) >= FRAME_STACK
    newest = env.frames.latest(env.t)[1]
    assert np.array_equal(obs["depth"][-1], newest)
    assert all(obs["depth"][i].any() for i in range(FRAME_STACK))
    # Consecutive frames are different captures (the partner is moving).
    assert not np.array_equal(obs["depth"][0], obs["depth"][-1])


def test_privileged_obs_is_proprio_plus_partner_truth():
    env = _check(obs_mode="privileged")
    obs, info = env.reset(seed=2)
    obs, _, _, _, info = env.step(np.zeros(7, np.float32))
    s = obs["state"]
    assert s.shape == (PRIVILEGED_DIM,)
    assert np.array_equal(s, info["privileged"])
    names = [n for n, _ in PRIVILEGED_LAYOUT]
    off = sum(n for _, n in PRIVILEGED_LAYOUT[:names.index("partner_p")])
    assert np.allclose(s[off:off + 3], info["partner_p"], atol=1e-6)


def test_trial_seed_replays_the_depth_noise_too():
    env = gym.make(ENV_ID, obs_mode="pixels", camera_scale=0.125).unwrapped
    runs = []
    for _ in range(2):
        env.reset(seed=123 + len(runs), options={"trial_seed": 4})
        for _ in range(4):
            obs, *_ = env.step(np.zeros(7, np.float32))
        runs.append(obs["depth"].copy())
    assert np.array_equal(runs[0], runs[1])


def test_bank_resets_rebuild_banked_scenarios(tmp_path):
    from dap.bank import build_bank, save_bank

    path = tmp_path / "bank.pkl"
    save_bank(build_bank(2, "fixed", first_seed=10), path)
    env = gym.make(ENV_ID, obs_mode="privileged", bank=str(path)).unwrapped
    assert env.scenario_name == "fixed"
    seen = set()
    for s in range(6):
        _, info = env.reset(seed=s)
        assert info["scenario_id"][0] == "bank"
        seen.add(info["scenario_id"][1])
    assert seen <= {10, 11}
    # Evaluation seeds bypass the bank.
    _, info = env.reset(options={"trial_seed": 0})
    assert info["scenario_id"] == ("trial", 0)
    import pytest

    with pytest.raises(ValueError, match="preset"):
        gym.make(ENV_ID, obs_mode="privileged", bank=str(path), scenario="wide")
