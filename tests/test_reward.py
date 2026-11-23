"""The training reward (dap/reward.py) ranks outcomes like the evaluation
loss, and its shaping is potential-based."""

import gymnasium as gym
import numpy as np
import pytest

import dap.envs  # noqa: F401
from dap.baselines import oracle_factory
from dap.experiment import Condition, ExperimentConfig
from dap.reward import DapReward, RewardConfig, outcome_reward, potential
from dap.scoring import dap_loss
from dap.trial import run_trial

CFG = ExperimentConfig(name="t")
RC = RewardConfig()


def _info(event, success=False, pos=0.0, rot=0.0, t=1.0, t_meet=1.0, gap=0.1):
    return dict(event=event, success=success, in_plane_offset=pos, rot_err_deg=rot,
                t=t, t_meet=t_meet, normal_gap=gap)


def test_outcome_reward_keeps_the_loss_tiers():
    worst_success = outcome_reward(_info("contact", True, pos=0.05, rot=15.0), RC)
    best_success = outcome_reward(_info("contact", True), RC)
    best_failed = outcome_reward(_info("contact", False, rot=15.01), RC)
    worst_failed = outcome_reward(_info("contact", False, pos=0.05, rot=180.0), RC)
    collision = outcome_reward(_info("collision"), RC)
    assert best_success == RC.offset
    assert worst_success > best_failed > worst_failed > 0.0 > collision
    assert collision == RC.offset - RC.loss.collision_penalty
    assert outcome_reward(_info(None), RC) == 0.0  # no reward before the end (or at a timeout)


def test_potential_is_zero_at_the_end_and_negative_before():
    assert potential(_info("contact"), RC) == 0.0
    assert potential(_info("collision"), RC) == 0.0
    assert potential(_info(None, rot=90.0, pos=0.1), RC) < 0.0
    # Aligned and touching the face center: nothing left to gain.
    assert potential(_info(None, rot=0.0, pos=0.0, gap=0.0), RC) == 0.0
    # Misaligned, the best place is the standoff in front of the face.
    far, at = (potential(_info(None, rot=30.0, gap=g), RC) for g in (0.0, RC.standoff))
    assert at > far


def _oracle_replay(seed=0):
    """The oracle's logged dap, replayed as 100 Hz velocity actions."""
    r, log = run_trial(CFG, Condition("oracle", 1.0), seed, controller_factory=oracle_factory,
                       return_log=True)
    env = DapReward(gym.make("dap/PalmMeet-v0", control_hz=100.0), RC)
    _, info0 = env.reset(options={"trial_seed": seed})
    q, terms = np.asarray(log.q), []
    for k in range(len(q) - 1):
        a = (q[k + 1] - q[k]) / (env.unwrapped.v_max * env.unwrapped.sim_dt)
        _, rew, term, trunc, info = env.step(a)
        terms.append(info["reward_terms"])
        assert abs(rew - sum(info["reward_terms"].values())) < 1e-12
        if term or trunc:
            break
    return r, info0, info, term, terms


def test_terminal_reward_is_the_trials_loss_and_shaping_telescopes():
    r, info0, info, term, terms = _oracle_replay()
    assert term and info["success"] and r.success
    row = {k: [v] for k, v in r.to_dict().items()}
    row["outcome"], row["success"] = np.array([r.outcome]), np.array([r.success])
    assert terms[-1]["outcome"] == pytest.approx(RC.offset - dap_loss(row)[0], abs=1e-9)
    assert all(t["outcome"] == 0.0 for t in terms[:-1])
    # Phi = 0 at the end: the discounted shaping sums to -Phi(s0), whatever the path.
    shaped = sum(RC.gamma ** i * t["shaping"] for i, t in enumerate(terms))
    assert shaped == pytest.approx(-potential(info0, RC), abs=1e-9)


def test_timeouts_have_no_terminal_reward_and_keep_their_potential():
    env = DapReward(gym.make("dap/PalmMeet-v0"), RC)
    env.reset(options={"trial_seed": 1})
    done = False
    while not done:
        _, _, term, trunc, info = env.step(np.zeros(7))
        done = term or trunc
        assert info["reward_terms"]["outcome"] == 0.0
        assert info["reward_terms"]["action_rate"] == 0.0
    assert trunc and not term  # holding still never daps
    assert potential(info, RC) < 0.0


def test_action_rate_uses_the_applied_action():
    env = DapReward(gym.make("dap/PalmMeet-v0"), RC)
    env.reset(options={"trial_seed": 1})
    a = np.full(7, 0.5)
    _, _, _, _, info = env.step(a)
    assert info["reward_terms"]["action_rate"] == pytest.approx(-RC.w_action_rate * 7 * 0.25)
    _, _, _, _, info = env.step(np.full(7, np.nan))  # the env applies 0 instead
    assert info["reward_terms"]["action_rate"] == pytest.approx(-RC.w_action_rate * 7 * 0.25)
    _, _, _, _, info = env.step(np.full(7, 3.0))  # clipped to 1
    assert info["reward_terms"]["action_rate"] == pytest.approx(-RC.w_action_rate * 7.0)
