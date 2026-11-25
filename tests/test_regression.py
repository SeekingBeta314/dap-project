"""Behavior lock: digests of what the harness, the baselines, the env and the
bank produce. The original exact-byte reference is for the team's Apple
silicon environment. On other systems a full Git clone runs the pinned
reference implementation in the same runtime. A small independent numerical
reference on fixed validation trajectories also runs on source exports.

To record an exact-byte reference for a specific runtime:

    REGRESSION_WRITE=1 REGRESSION_GOLDEN=/tmp/golden.json pytest tests/test_regression.py

Only public outputs are digested (trial rows and logs, env observations,
rewards, flags and info, bank specs, estimator outputs), with dtypes and
shapes; wall-clock fields are left out."""

import hashlib
import json
import io
import platform
import shutil
import subprocess
import sys
import tarfile
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


def all_digests():
    return {**_trials(), **_envs(), **_policy_eval(), **_bank_specs(), **_estimator()}


# The original bit digests were produced on Apple silicon. Raw solver and
# renderer bytes depend on CPU, libraries and driver. On Linux compare the
# same cases to the unmodified team implementation on the same machine.
REFERENCE_COMMIT = "d3f391445abaecdb287ac78fb47fc511f0d5381b"


def reference_on_this_runtime(tmp_path):
    root = Path(__file__).resolve().parents[1]
    if not shutil.which("git"):
        return None
    archive = subprocess.run(
        ["git", "archive", REFERENCE_COMMIT, "dap", "models/package.xml",
         "models/dap_hand.sdf", "tests/test_regression.py"],
        cwd=root, capture_output=True)
    if archive.returncode:  # e.g. source ZIP or a shallow clone
        return None
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as bundle:
        bundle.extractall(tmp_path, filter="data")
    # A separate process is essential: both programs import modules named dap.
    program = """
import json, runpy
from pathlib import Path
import dap
assert Path(dap.__file__).resolve().is_relative_to(Path.cwd())
cases = runpy.run_path('tests/test_regression.py')
result = {}
for name in ('_trials', '_envs', '_policy_eval', '_bank_specs', '_estimator'):
    result.update(cases[name]())
with open('digests.json', 'w') as f:
    json.dump(result, f)
"""
    subprocess.run([sys.executable, "-c", program], cwd=tmp_path, check=True)
    return json.loads((tmp_path / "digests.json").read_text())


def test_behavior_matches_the_recorded_golden(tmp_path):
    override = os.environ.get("REGRESSION_GOLDEN")
    golden_path = Path(override) if override else GOLDEN
    if os.environ.get("REGRESSION_WRITE"):
        golden_path.parent.mkdir(exist_ok=True)
        golden_path.write_text(json.dumps(all_digests(), indent=1, sort_keys=True) + "\n")
        pytest.skip(f"recorded digests to {golden_path}")
    current = all_digests()
    golden = json.loads(golden_path.read_text())
    if not override and current != golden:
        native = reference_on_this_runtime(tmp_path)
        if native is not None:
            golden = native
        elif (sys.platform, platform.machine()) != ("darwin", "arm64"):
            pytest.skip("Exact local reference requires full git history or REGRESSION_GOLDEN; "
                        "portable numerical regression runs independently")
    assert set(current) == set(golden)
    changed = sorted(k for k in golden if current[k] != golden[k])
    assert not changed, f"behavior changed: {changed}"


def test_portable_numerical_regression():
    """Replay validated trajectories against numeric results from the pristine reference."""
    from dap.eval_scenarios import FrozenScenario, decode_array
    data = json.loads((GOLDEN.parent / "portable_regression.json").read_text(),
                      object_hook=decode_array)
    assert data["source"] == REFERENCE_COMMIT

    class FixedDepthAction:
        obs_mode = "pixels"

        def __call__(self, obs):
            action = np.array([.3, .6, 0, -.5, 0, .4, 0])
            depth = obs["depth"][-1]
            valid = depth[depth > 0]
            if valid.size:
                action[0] = np.clip(valid.mean() / 3., 0, 1)
            return action

    for case in data["cases"]:
        cfg = ExperimentConfig(name="portable", scenario=case["preset"], split="val")
        seed = case["seed"]
        frozen = FrozenScenario(cfg.scenario, cfg.split, seed, "", case["spec"])
        for name, expected in case["results"].items():
            cond = Condition(name, 1.)
            options = dict(return_log=True, frozen_scenario=frozen)
            if name == "pixels":
                result, log = run_policy_trial(cfg, cond, seed, FixedDepthAction(), **options)
            else:
                factory = getattr(baselines, name + "_factory")
                result, log = run_trial(cfg, cond, seed, controller_factory=factory, **options)
            assert result.outcome == expected["outcome"] and result.success == expected["success"]
            assert result.n_steps == expected["n_steps"]
            np.testing.assert_allclose(result.min_palm_dist, expected["min_palm_dist"], atol=2e-4)
            if expected["t_event"] is None:
                assert np.isnan(result.t_event)
            else:
                np.testing.assert_allclose(result.t_event, expected["t_event"], atol=1e-8)
            q = np.asarray(log.q)[[0, len(log.q) // 2, -1]]
            np.testing.assert_allclose(q, expected["trace"], atol=2e-4)
