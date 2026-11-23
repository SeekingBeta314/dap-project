import dataclasses
import json

import numpy as np
import pytest

from dap.experiment import (CONFIGS, EVAL_FIRST_SEED, SPLITS, Condition, ExperimentConfig,
                            trial_rngs)
from dap.metrics import TrialResult


def test_config_expands_to_full_paired_grid():
    cfg = ExperimentConfig(name="grid", planners=("servo", "servo_depth"),
                           noise_scales=(0.5, 1.0, 2.0), n_seeds=20)
    trials = list(cfg.trials())
    assert len(trials) == cfg.n_trials() == 2 * 3 * cfg.n_seeds
    # Every condition runs on exactly the same seeds.
    seeds = {}
    for cond, seed in trials:
        seeds.setdefault(cond, []).append(seed)
    assert len({tuple(v) for v in seeds.values()}) == 1


def test_condition_has_no_scenario_knobs():
    # Speed is a property of the sampled scenario, not a condition, and
    # there are no checkpoints.
    assert [f.name for f in dataclasses.fields(Condition)] == ["planner", "noise_scale"]


def test_rng_streams_are_independent_and_reproducible():
    a, b = trial_rngs(3), trial_rngs(3)
    assert a["scenario"].random() == b["scenario"].random()
    assert trial_rngs(3)["scenario"].random() != trial_rngs(3)["sensor"].random()


def test_splits_are_disjoint_at_any_seed():
    """No stream of one split equals a stream of another, also for the
    multi-word seeds that fresh env scenarios draw (up to 2**63)."""
    seeds = [*range(300), 2**32, 2**32 + 1, 2**33 + 2, 2**62 + 7, 2**63 - 1]
    first = {}
    for split, rngs in SPLITS.items():
        for s in seeds:
            if split == "eval" and s >= 2**32:
                continue
            for k, rng in rngs(s).items():
                first.setdefault(rng.random(), []).append((split, s, k))
    assert all(len(v) == 1 for v in first.values())
    # Evaluation seeds are at most one 32-bit word, which is what keeps the
    # tagged training and validation streams apart from them.
    for s in (-1, 2**32):
        with pytest.raises(ValueError):
            trial_rngs(s)


def test_results_come_only_from_unspent_evaluation_seeds():
    assert ExperimentConfig(name="x").first_seed == EVAL_FIRST_SEED
    assert ExperimentConfig(name="x", split="val").first_seed == 0
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", first_seed=EVAL_FIRST_SEED - 1)
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", split="train")
    for cfg in CONFIGS.values():
        assert cfg.split == "val" or min(cfg.seeds()) >= EVAL_FIRST_SEED
    # A trial draws from its config's split.
    cfg = ExperimentConfig(name="x", split="val")
    assert cfg.rngs(3)["scenario"].random() == SPLITS["val"](3)["scenario"].random()
    assert cfg.rngs(3)["scenario"].random() != trial_rngs(3)["scenario"].random()


def test_result_serializes_to_strict_json():
    r = TrialResult(seed=0, planner="replan", noise_scale=1.0)
    r.min_arm_dist = float("inf")
    d = r.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["rot_err"] is None and d["min_arm_dist"] is None
    assert not any(k.startswith(("t_meet_pred", "plan_times", "n_replans", "checkpoint"))
                   for k in d), "controller-reported fields must not be in the result"
    assert np.isnan(TrialResult(seed=0, planner="x", noise_scale=1.0).time_err)
