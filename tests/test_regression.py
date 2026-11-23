"""Behavior lock: digests of what the harness, the baselines, the env and the
bank produce, compared bit for bit with tests/data/regression_golden.json.
Refactors must leave every digest unchanged; a change in behavior must be
deliberate and re-recorded:

    REGRESSION_WRITE=1 pytest tests/test_regression.py

Only public outputs are digested (trial rows and logs, env observations,
rewards, flags and info, bank specs, estimator outputs), with dtypes and
shapes; wall-clock fields are left out."""

import hashlib
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import dap.envs  # noqa: F401  (registers dap/PalmMeet-v0)
from dap import baselines
from dap.bank import build_bank, scenario_from_spec, scenario_spec
from dap.envs import ENV_ID
from dap.estimator import PalmEstimator
from dap.experiment import Condition, ExperimentConfig, train_rngs
from dap.partner import NoisyPoseSensor
from dap.policy_eval import run_policy_trial
from dap.scenario import SCENARIO_PRESETS, sample_scenario
from dap.trial import run_trial

GOLDEN = Path(__file__).parent / "data" / "regression_golden.json"
WALL_CLOCK = {"wall_time_s", "step_ms_mean", "step_ms_p95", "step_ms_max", "sample_s",
              "solve_ms", "build_s"}


def _feed(h, x):
    if isinstance(x, dict):
        h.update(b"{")
        for k in sorted(x):
            if k not in WALL_CLOCK:
                h.update(repr(k).encode())
                _feed(h, x[k])
        h.update(b"}")
    elif isinstance(x, (list, tuple)):
        h.update(b"[%d" % len(x))
        for v in x:
            _feed(h, v)
    elif x is None or isinstance(x, (str, bool)):
        h.update(repr(x).encode())
    else:
        a = np.ascontiguousarray(np.asarray(x))
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())


def digest(x):
    h = hashlib.sha256()
    _feed(h, x)
    return h.hexdigest()[:20]


# -- cases -------------------------------------------------------------------

def _trials():
    out = {}
    for preset, names, seeds in (("nominal", ("hold", "blind", "servo", "servo_gt", "oracle"),
                                  (0, 1)),
                                 ("wide", ("servo",), (0,))):
        cfg = ExperimentConfig(name="r", scenario=preset)
        for name in names:
            factory = getattr(baselines, f"{name}_factory")
            for s in seeds:
                r, log = run_trial(cfg, Condition(name, 1.0), s, controller_factory=factory,
                                   return_log=True)
                out[f"trial/{preset}/{name}/{s}/row"] = digest(r.to_dict())
                out[f"trial/{preset}/{name}/{s}/log"] = digest(log.arrays())
    cfg = ExperimentConfig(name="r")
    for name in baselines.CAMERA_BASELINES:  # on the head camera's frames
        factory = getattr(baselines, f"{name}_factory")
        for s in (0, 1):
            r, log = run_trial(cfg, Condition(name, 1.0), s, camera=baselines.SERVO_DEPTH_CAMERA,
                               controller_factory=factory, return_log=True)
            out[f"trial/nominal/{name}/{s}/row"] = digest(r.to_dict())
            out[f"trial/nominal/{name}/{s}/log"] = digest(log.arrays())
    return out


def _rollout(env, reset_kw, n, actions):
    steps = [env.reset(**reset_kw)]
    for a in actions[:n]:
        steps.append(env.step(a))
        if steps[-1][2] or steps[-1][3]:
            break
    return digest(steps)


def _envs():
    out = {}
    actions = np.random.default_rng(7).uniform(-1, 1, (400, 7)).astype(np.float32)
    trial = {"options": {"trial_seed": 0}}
    for mode, kw in (("pixels", {}), ("privileged", {}), ("state", {}), ("state+depth", {}),
                     ("privileged", {"privileged_delay": False})):
        env = gym.make(ENV_ID, obs_mode=mode, camera_scale=0.125, **kw).unwrapped
        out[f"env/{mode}/{sorted(kw.items())}"] = _rollout(env, trial, 25, actions)
    # A whole episode to truncation, with the recorded log.
    env = gym.make(ENV_ID, obs_mode="state", record_log=True).unwrapped
    out["env/state/full"] = _rollout(env, {"options": {"trial_seed": 1}}, 400,
                                     np.zeros((400, 7), np.float32))
    out["env/state/full/log"] = digest(env.log.arrays())
    # Banked and freshly sampled resets.
    bank = build_bank(2, "nominal", first_seed=0)
    # Provenance (git commit, versions, platform) describes the machine, not behavior.
    out["bank/build"] = digest({k: v for k, v in bank.items() if k != "provenance"})
    env = gym.make(ENV_ID, obs_mode="privileged", bank=bank).unwrapped
    out["env/privileged/bank"] = _rollout(env, {"seed": 3}, 10, actions)
    env = gym.make(ENV_ID, obs_mode="privileged").unwrapped
    out["env/privileged/sampled"] = _rollout(env, {"seed": 5}, 10, actions)
    return out


class _ConstVel:
    obs_mode = "pixels"

    def __call__(self, obs):
        return np.array([0.3, 0.6, 0, -0.5, 0, 0.4, 0])


def _policy_eval():
    cfg = ExperimentConfig(name="r")
    r, log = run_policy_trial(cfg, Condition("student", 1.0), 1, _ConstVel(),
                              env_kwargs={"camera_scale": 0.125}, return_log=True)
    return {"policy_eval/row": digest(r.to_dict()), "policy_eval/log": digest(log.arrays())}


def _bank_specs():
    out = {}
    for preset, seed in (("nominal", 0), ("wide", 0), ("wide", 2)):
        spec = scenario_spec(sample_scenario(train_rngs(seed)["scenario"],
                                             SCENARIO_PRESETS[preset]))
        sc = scenario_from_spec(spec)
        ts = np.linspace(0, sc.partner.t_end, 50)
        out[f"spec/{preset}/{seed}"] = digest(spec)
        out[f"rebuilt/{preset}/{seed}"] = digest(
            [[sc.partner.q(t) for t in ts], sc.scene.q(), sc.q_ours0, sc.partner.P])
    return out


def _estimator():
    sc = sample_scenario(np.random.default_rng(0), SCENARIO_PRESETS["nominal"])
    sensor = NoisyPoseSensor(sc.partner, np.random.default_rng(1))
    est = PalmEstimator(pos_noise=sensor.noise_pos)
    rows = []
    for t in np.arange(0, sc.partner.t_end, 0.01):
        m = sensor.measure(t)
        est.update(t, *m)
        rows.append([m, est.predict(0.1), est.rotation(), est.normal])
    return {"estimator": digest(rows)}


@pytest.fixture(scope="module")
def current():
    return {**_trials(), **_envs(), **_policy_eval(), **_bank_specs(), **_estimator()}


def test_behavior_matches_the_recorded_golden(current):
    if os.environ.get("REGRESSION_WRITE"):
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"recorded {len(current)} digests to {GOLDEN}")
    golden = json.loads(GOLDEN.read_text())
    assert set(current) == set(golden)
    changed = sorted(k for k in golden if current[k] != golden[k])
    assert not changed, f"behavior changed: {changed}"
