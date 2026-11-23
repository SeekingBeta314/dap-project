"""Scenario bank: rebuilt scenarios are identical to sampled ones, fresh env
scenarios are not evaluation scenarios, and the IK context cache leaves
sampling unchanged. Split disjointness: tests/test_experiment.py."""

import numpy as np
import pytest

from dap.bank import ScenarioBank, build_bank, save_bank, scenario_from_spec, scenario_spec
from dap.camera import HeadCameraConfig
from dap.experiment import train_rngs, trial_rngs
from dap.scenario import SCENARIO_PRESETS, sample_scenario
from dap.scene import OURS, PARTNER


def _same_scenario(a, b):
    ts = np.linspace(0.0, a.partner.t_end, 300)
    assert np.array_equal([a.partner.q(t) for t in ts], [b.partner.q(t) for t in ts])
    for k in ("P", "N", "V", "times"):
        assert np.array_equal(getattr(a.partner, k), getattr(b.partner, k)), k
    assert a.partner.t_meet == b.partner.t_meet and a.partner.t_end == b.partner.t_end
    assert np.array_equal(a.q_ours0, b.q_ours0)
    assert np.array_equal(a.q_ours_meet, b.q_ours_meet)
    assert a.layout == b.layout
    assert a.target_peak_speed == b.target_peak_speed
    assert a.reachable_in_time == b.reachable_in_time
    assert (a.partner.reference.switch is None) == (b.partner.reference.switch is None)
    # The rebuilt scene starts where the sampled one does.
    assert np.array_equal(a.scene.q(), b.scene.q())


@pytest.mark.parametrize("preset,seed", [("nominal", 0), ("nominal", 1), ("wide", 0), ("wide", 2)])
def test_rebuilt_scenario_is_identical(preset, seed):
    sc = sample_scenario(train_rngs(seed)["scenario"], SCENARIO_PRESETS[preset])
    _same_scenario(sc, scenario_from_spec(scenario_spec(sc)))
    # With the head camera too (the camera adds no geometry).
    _same_scenario(sc, scenario_from_spec(scenario_spec(sc), HeadCameraConfig().scaled(0.125)))


def test_a_wide_bank_contains_target_switches():
    specs = [scenario_spec(sample_scenario(train_rngs(s)["scenario"], SCENARIO_PRESETS["wide"]))
             for s in range(8)]
    assert any(sp["reference"]["switch"] is not None for sp in specs)
    for sp in specs:
        if sp["reference"]["switch"] is not None:
            rebuilt = scenario_from_spec(sp)
            assert rebuilt.partner.reference.switch is not None


def test_fresh_env_scenarios_are_not_evaluation_scenarios():
    # np_random(s) equals the evaluation scenario stream of seed s
    # (default_rng([s, 0]) == default_rng(s)); the env must not sample from it.
    import gymnasium as gym

    import dap.envs  # noqa: F401
    from dap.envs import ENV_ID

    env = gym.make(ENV_ID, obs_mode="privileged").unwrapped
    for s in range(2):
        env.reset(seed=s)
        ev = sample_scenario(trial_rngs(s)["scenario"], SCENARIO_PRESETS["nominal"])
        assert not np.array_equal(env.scenario.q_ours0, ev.q_ours0)


def test_parallel_build_matches_serial():
    a, b = build_bank(2, "nominal", first_seed=3), build_bank(2, "nominal", first_seed=3, workers=2)
    for x, y in zip(a["specs"], b["specs"]):
        x, y = scenario_from_spec(x), scenario_from_spec(y)
        _same_scenario(x, y)


def test_bank_round_trip(tmp_path):
    bank = build_bank(3, "nominal", first_seed=5)
    assert len(bank["specs"]) == 3 and bank["failed_seeds"] == []
    assert [s["seed"] for s in bank["specs"]] == [5, 6, 7]
    path = tmp_path / "b.pkl"
    save_bank(bank, path)
    loaded = ScenarioBank(path)
    assert len(loaded) == 3 and loaded.preset == "nominal"
    assert set(loaded.provenance) == {"git", "drake", "numpy", "python", "platform"}
    assert loaded.provenance["git"] != "unknown"  # the tests run inside the checkout
    sc = sample_scenario(train_rngs(6)["scenario"], SCENARIO_PRESETS["nominal"])
    _same_scenario(sc, loaded.get(1))


def test_bank_rejects_a_changed_preset_or_version():
    bank = build_bank(1, "nominal")
    with pytest.raises(ValueError, match="different"):
        ScenarioBank(dict(bank, config="something else"))
    with pytest.raises(ValueError, match="version"):
        ScenarioBank(dict(bank, version=0))


def test_scratch_context_is_cached_and_not_the_measured_one(scenario):
    scene = scenario.scene
    a, b = scene.scratch_context(), scene.scratch_context()
    assert a[0] is b[0]
    assert a[0] is not scene.context
    assert scene.scratch_context(filter_hands=True)[0] is not a[0]
    # Posing the scratch context does not move the measured arms.
    q = scene.q()
    scene.plant.SetPositions(a[1], q + 0.1)
    assert np.array_equal(scene.q(), q)
    assert np.array_equal(scene.arm_q(OURS), q[scene.idx[OURS]])
    assert np.array_equal(scene.arm_q(PARTNER), q[scene.idx[PARTNER]])
