import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from publish_study import SCENARIO_FIELDS, audit_run, timing_sensitivity


def test_audit_rejects_missing_trials_and_mismatched_scenarios(tmp_path):
    cfg = dict(config=dict(first_seed=0, n_seeds=2, planners=["replan"], noise_scales=[1]))
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    rows = [dict.fromkeys(SCENARIO_FIELDS, 1) | dict(seed=s, planner="replan",
            noise_scale=1, error_kind="") for s in range(2)]
    trials = tmp_path / "trials.jsonl"
    trials.write_text(json.dumps(rows[0]))
    with pytest.raises(ValueError, match="missing or unexpected"):
        audit_run(tmp_path, {})
    trials.write_text("\n".join(map(json.dumps, reversed(rows))))
    scenarios = {}
    assert audit_run(tmp_path, scenarios)[1] == rows
    rows[0]["layout_height"] = 0.4
    trials.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="scenario differs"):
        audit_run(tmp_path, scenarios)


def test_timing_sensitivity_uses_all_trials_in_the_denominator():
    trials = pd.DataFrame(dict(planner=["replan"] * 5,
                               success=[True, True, True, False, False],
                               time_err=[-0.1, 0.15, -1.0, 0.01, float("nan")]))
    table = timing_sensitivity(trials)
    assert list(table.k) == [1, 2, 3]
    assert list(table.n) == [5, 5, 5]


def test_resume_preserves_provenance_and_rejects_changed_weights():
    from run_experiment import _resume_metadata
    old = dict(config=dict(n_seeds=3, noise_scales=[1]), runtime={"sha256": {"net.npz": "v1"}},
               git="old", started="before", policy="m:net", controller="planner")
    current = {**old, "config": {**old["config"], "n_seeds": 6}, "git": "new", "started": "now"}
    merged = _resume_metadata(old, current)
    assert merged["git"] == "old" and merged["started"] == "before"
    assert merged["config"]["n_seeds"] == 6
    current["runtime"] = {"sha256": {"net.npz": "v2"}}
    with pytest.raises(SystemExit, match="different or unverified"):
        _resume_metadata(old, current)


def test_already_complete_run_leaves_config_alone(tmp_path, monkeypatch):
    import run_experiment
    run = tmp_path / "smoke"
    run.mkdir()
    (run / "config.json").write_text("existing metadata")
    rows = [dict(planner="servo", noise_scale=1.0, seed=i, error="") for i in range(3)]
    (run / "trials.jsonl").write_text("\n".join(map(json.dumps, rows)))
    monkeypatch.setattr(sys, "argv", ["run", "smoke", "--out", str(tmp_path)])
    run_experiment.main()
    assert (run / "config.json").read_text() == "existing metadata"
