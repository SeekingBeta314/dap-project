"""The PPO teacher's plumbing (dap/teacher.py): features, per-worker rollout
collection, checkpoints, scoring through dap.policy_eval, and the warm start
from the servo expert. Nothing here checks that anything learns well."""

import numpy as np

from dap.bank import build_bank
from dap.envs.dap_env import PRIVILEGED_DIM, PRIVILEGED_LAYOUT
from dap.experiment import Condition, ExperimentConfig
from dap.policy_eval import run_policy_trial
from dap.teacher import (FEATURE_DIM, ParallelPPO, ServoExpert, collect_expert, dagger,
                         load_teacher, make_teacher_env, teacher_features, warm_start)


def _vector(**parts):
    v = []
    for name, n in PRIVILEGED_LAYOUT:
        v.append(np.asarray(parts.get(name, np.zeros(n)), float))
    return np.concatenate(v)


def test_features_are_identity_when_antiparallel():
    x, z = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    s = _vector(palm_p=[0.1, 0, 0.5], palm_x=x, palm_n=z,
                partner_p=[0.4, 0.1, 0.5], partner_x=-x, partner_n=-z)
    f = teacher_features(s)
    assert f.shape == (FEATURE_DIM,) and np.allclose(f[:PRIVILEGED_DIM], s)
    assert np.allclose(f[PRIVILEGED_DIM:PRIVILEGED_DIM + 3], [0.3, 0.1, 0.0])
    assert np.allclose(f[-9:].reshape(3, 3), np.eye(3), atol=1e-6)
    # Fingers parallel instead of opposite: 180 degrees about the normal.
    f = teacher_features(_vector(palm_x=x, palm_n=z, partner_x=x, partner_n=-z))
    assert np.isclose(np.trace(f[-9:].reshape(3, 3)), -1.0)
    # Batches work like single vectors.
    assert teacher_features(np.stack([s, s])).shape == (2, FEATURE_DIM)


def test_teacher_env_spaces():
    env = make_teacher_env()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_parallel_ppo_collects_saves_and_scores(tmp_path):
    bank = build_bank(2, "nominal", first_seed=0)
    model = ParallelPPO(2, dict(bank=bank), seed=0, n_steps=64, batch_size=64, n_epochs=1)
    try:
        model.learn(2 * 64 * 2)
        assert model.num_timesteps == 256
        buf = model.rollout_buffer
        assert buf.full and buf.observations["state"].shape == (64 * 2, FEATURE_DIM)  # flattened by train()
        assert np.all(np.isfinite(buf.returns)) and np.all(np.isfinite(buf.advantages))
        assert model.episode_log and model.obs_rms.count > 256
        model.save_checkpoint(tmp_path / "ckpt")
    finally:
        model.close()
    teacher = load_teacher(tmp_path / "ckpt")
    cfg = ExperimentConfig(name="t", split="val", n_seeds=1)
    r = run_policy_trial(cfg, Condition("teacher", 1.0), 0, teacher, allow_privileged=True)
    assert not r.error and r.outcome in ("contact", "collision", "timeout")
    priv = np.zeros(PRIVILEGED_DIM, np.float32)
    assert teacher.label(priv).shape == (7,) and teacher.label(np.stack([priv] * 3)).shape == (3, 7)


def test_warm_start_fits_the_expert():
    bank = build_bank(2, "nominal", first_seed=0)
    episodes = collect_expert(4, dict(bank=bank), workers=2)
    assert len(episodes) == 4 and all(len(e["obs"]) == len(e["act"]) == len(e["rew"])
                                      for e in episodes)
    assert all(e["obs"].shape[1] == FEATURE_DIM and np.all(np.abs(e["act"]) <= 1)
               for e in episodes)
    model = ParallelPPO(1, dict(bank=bank), seed=0, n_steps=64, batch_size=64)
    try:
        before = warm_start(model, episodes, epochs=0)
        after = warm_start(model, episodes, epochs=100, log_std=np.log(0.3))
        assert after["pi_mse"] < 0.1 * before["pi_mse"]
        assert after["vf_mse"] < 0.1 * before["vf_mse"]
        assert np.allclose(model.policy.log_std.detach().numpy(), np.log(0.3))
    finally:
        model.close()


def test_kickstarting_pulls_toward_the_expert_on_own_states():
    bank = build_bank(2, "nominal", first_seed=0)
    model = ParallelPPO(2, dict(bank=bank), seed=0, kick_coef=100.0, kick_steps=10_000,
                        n_steps=64, batch_size=64, n_epochs=4)
    try:
        model.learn(2 * 64)  # one rollout, then one update
        obs, expert = model._kick
        assert obs.shape == (128, FEATURE_DIM) and expert.shape == (128, 7)
        assert np.all(np.abs(expert.numpy()) <= 1) and expert.abs().sum() > 0
        mean = lambda: model.policy.get_distribution({"state": obs}).distribution.mean.detach()  # noqa: E731
        before = float(((mean() - expert) ** 2).mean())
        for _ in range(20):  # more updates on the same rollout
            model.train()
        assert float(((mean() - expert) ** 2).mean()) < 0.6 * before
    finally:
        model.close()


def test_dagger_aggregates_expert_labels_on_policy_states():
    bank = build_bank(2, "nominal", first_seed=0)
    episodes = collect_expert(4, dict(bank=bank), workers=2)
    n0 = sum(len(e["obs"]) for e in episodes)
    model = ParallelPPO(2, dict(bank=bank), seed=0, expert_labels=True, n_steps=64, batch_size=64)
    try:
        warm_start(model, episodes, epochs=50)
        stats = dagger(model, episodes, 100, epochs=50)
        assert stats["n"] == n0 + 100 == sum(len(e["obs"]) for e in episodes)
        new = episodes[-1]
        assert new["obs"].shape[1] == FEATURE_DIM and np.all(np.abs(new["act"]) <= 1)
        assert np.isfinite(stats["gap_mse"]) and stats["pi_mse"] < stats["gap_mse"]
    finally:
        model.close()


def test_expert_labels_depend_on_the_state_alone():
    env, driver, fresh = make_teacher_env(), ServoExpert(), ServoExpert()
    obs, _ = env.reset(seed=0)
    done, closing = False, False
    while not done:
        priv = {"state": obs["state"][:PRIVILEGED_DIM]}
        a = driver(priv)
        fresh.reset()
        assert np.allclose(fresh(priv), a)
        closing |= driver.servo.closing
        obs, _, term, trunc, _ = env.step(a)
        done = term or trunc
    assert closing  # the episode reached the closing phase
