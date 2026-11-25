import dataclasses
import gzip
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_perception import compare, outcomes, publish
from dap.eval_scenarios import freeze_record
from dap.experiment import CONFIGS, PERCEPTION_METHODS
from dap.metrics import TrialResult
from publish_study import SCENARIO_FIELDS
from run_perception_study import CORE_FILES, atomic_json, digest


def test_cross_preset_analysis_rebuilds_from_raw_records(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for preset in ("nominal", "wide"):
        name = f"perception_{preset}"
        config = dataclasses.replace(CONFIGS[name], n_seeds=8)
        monkeypatch.setitem(CONFIGS, name, config)
        specs = {s: freeze_record(preset, "eval", s, {"reference": {"switch": None}}) for s in config.seeds()}
        scenario_path = source / "scenarios" / f"{preset}.jsonl.gz"
        scenario_path.parent.mkdir(parents=True, exist_ok=True)
        scenario_path.write_bytes(gzip.compress("".join(json.dumps(r) + "\n" for r in specs.values()).encode()))
        for method in PERCEPTION_METHODS:
            target = source / "eval" / preset / method
            target.mkdir(parents=True)
            manifest = dict(config=dataclasses.asdict(config), method=method, camera={},
                            scenario_archive=digest(scenario_path), depth_noise="med", sources=dict.fromkeys(CORE_FILES, "same"),
                            checkpoints={}, versions={k: "v" for k in ("drake", "numpy", "scipy", "gymnasium")})
            (target / "manifest.json").write_text(json.dumps(manifest))
            for seed in config.seeds():
                r = TrialResult(seed, method, 1.).to_dict()
                r.update(dict.fromkeys(SCENARIO_FIELDS, .34), scenario_sha256=specs[seed]["sha256"], outcome="collision", collision=True)
                if method == "student" and (preset == "nominal" or seed < 1007):
                    r.update(outcome="contact", contact=True, collision=False, success=True,
                             rot_err=2., pos_err=.01, gap=.001, time_err=.05, angle_err=2.,
                             finger_err=0., rel_speed=.1)
                (target / f"{seed}.json").write_text(json.dumps(r))
    built = tmp_path / "output"
    publish(source, built)
    pairs = pd.read_csv(built / "paired.csv").query("a == 'student'")
    assert list(pairs.a_only) == [8, 7]
    assert list(pairs.b_only) == [0, 0]
    assert list(pairs.p_holm) == [.015625, .015625]
    assert (built / "figures" / "success.pdf").stat().st_size > 0
    assert r"\label{tab:perception-paired}" in (built / "paired_table.tex").read_text()
    assert list(pd.read_csv(built / "by_switch.csv").n) == [8, 8, 8, 8]
    rebuilt = tmp_path / "rebuilt"
    publish(built, rebuilt)
    for f in built.glob("*.csv"):
        assert f.read_bytes() == (rebuilt / f.name).read_bytes()
    manifest_path = source / "eval" / "wide" / "student" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoints"]["weights"] = "changed"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checkpoint changed"):
        publish(source, rebuilt)


def test_pairing_and_failure_types():
    d = pd.DataFrame({"planner": ["a", "a", "b", "b"], "seed": [7, 8, 8, 7],
                      "outcome": ["contact", "timeout", "timeout", "contact"],
                      "success": [True, False, False, False], "loss": [1., 11., 13., 6.],
                      "min_palm_dist": [0., .05, .051, .01], "controller_crash": False})
    c = compare(d, "a", "b")
    assert (c["a_only"], c["b_only"], c["success_diff"], c["loss_diff"]) == (1, 0, .5, -3.5)
    f = outcomes(d).set_index("planner")
    assert f.loc["a", "near_miss"] == f.loc["b", "far_timeout"] == f.loc["b", "misaligned"] == 1
    with pytest.raises(ValueError, match="cannot pair"):
        compare(d.iloc[:-1], "a", "b")


def test_resume_checks_identity_but_allows_an_unrelated_git_commit(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_json(path, {"checkpoints": {"weights": "a"}, "git": "first"}, immutable=True)
    atomic_json(path, {"checkpoints": {"weights": "a"}, "git": "later"}, immutable=True)
    assert json.loads(path.read_text())["git"] == "first"
    with pytest.raises(ValueError, match="different experiment"):
        atomic_json(path, {"checkpoints": {"weights": "b"}, "git": "later"}, immutable=True)


@pytest.mark.parametrize("preset", ["nominal", "wide"])
def test_public_freeze_api_replays_original_camera_and_scenario(tmp_path, monkeypatch, preset):
    import numpy as np

    from dap.bank import scenario_spec
    from dap.baselines import hold_factory
    from dap.eval_scenarios import load_frozen_scenario
    from dap.experiment import Condition
    from dap.policy_eval import make_eval_env, run_policy_trial
    from dap.trial import run_trial

    cfg = dataclasses.replace(CONFIGS[f"perception_{preset}"], split="val", first_seed=500)
    seed = 500
    cond = Condition("hold", 1.)
    env = make_eval_env(cfg, cond, "pixels")
    direct = {"split": "val", "trial_seed": seed}
    env.reset(options=direct)
    stored = freeze_record(preset, cfg.split, seed, scenario_spec(env.scenario))
    (tmp_path / f"{preset}.jsonl.gz").write_bytes(gzip.compress(json.dumps(stored).encode()))
    frozen = load_frozen_scenario(cfg, seed, tmp_path)
    direct_frame = env.step(np.zeros(7))[0]["depth"].copy()
    # Both replay paths must use the stored solver output, and preserve noise streams.
    def unexpected(*_args, **_kwargs):
        pytest.fail("replay tried to resample a scenario")

    monkeypatch.setattr("dap.envs.dap_env.sample_scenario", unexpected)
    monkeypatch.setattr("dap.trial.sample_scenario", unexpected)
    env.reset(options=dict(direct, frozen_scenario=frozen))
    assert np.array_equal(env.step(np.zeros(7))[0]["depth"], direct_frame)
    result = run_trial(cfg, cond, seed, controller_factory=hold_factory,
                       frozen_scenario=frozen)

    class Hold:
        obs_mode = "privileged"

        def __call__(self, obs):
            return np.zeros(7)

    record = run_policy_trial(cfg, cond, seed, Hold(), frozen_scenario=frozen, allow_privileged=True)
    assert result.outcome == record.outcome
    for key in SCENARIO_FIELDS:
        assert getattr(record, key) == pytest.approx(getattr(result, key))
    with pytest.raises(ValueError, match="identity differs"):
        frozen.build(preset, "eval", seed)
