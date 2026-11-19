import dataclasses
import json

import numpy as np

from dap.experiment import CONFIGS, Condition, trial_rngs
from dap.metrics import TrialResult


def test_config_expands_to_full_paired_grid():
    cfg = CONFIGS["replan_vs_once"]
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


def test_result_serializes_to_strict_json():
    r = TrialResult(seed=0, planner="replan", noise_scale=1.0)
    r.min_arm_dist = float("inf")
    d = r.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["rot_err"] is None and d["min_arm_dist"] is None
    assert not any(k.startswith(("t_meet_pred", "plan_times", "n_replans", "checkpoint"))
                   for k in d), "controller-reported fields must not be in the result"
    assert np.isnan(TrialResult(seed=0, planner="x", noise_scale=1.0).time_err)
