"""scripts/analyze.py on synthetic results: pairing, the crash policy,
resumed sweeps, out-of-range speeds and degenerate inputs."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from dap.metrics import TrialResult

_spec = importlib.util.spec_from_file_location(
    "analyze", Path(__file__).resolve().parent.parent / "scripts" / "analyze.py")
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)

EDGES = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]


def _row(planner, seed, outcome="contact", success=True, speed=0.5, error_kind="", **kw):
    r = TrialResult(seed=seed, planner=planner, noise_scale=1.0)
    r.outcome, r.success = outcome, success
    r.contact, r.collision = outcome == "contact", outcome == "collision"
    r.partner_peak_speed = r.target_peak_speed = speed
    if outcome == "contact":
        r.pos_err, r.rot_err, r.time_err = 0.01, 5.0 if success else 40.0, 0.02
    if outcome == "timeout":
        r.min_palm_dist = 0.2
    if outcome == "error":
        r.error, r.error_kind = "Traceback ...", error_kind
        if error_kind != "controller":  # controller crashes keep the scenario
            r.partner_peak_speed = float("nan")
    for k, v in kw.items():
        setattr(r, k, v)
    return r.to_dict()


def _write(tmp_path, rows):
    d = tmp_path / "res"
    d.mkdir()
    (d / "trials.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_paired_counts_and_crash_policy(tmp_path):
    rows = []
    for s in range(20):
        rows.append(_row("a", s, speed=0.2 + 0.03 * s))
        rows.append(_row("b", s, outcome="timeout", success=False, speed=0.2 + 0.03 * s))
    # Seed 20: scenario crash for both -> excluded everywhere.
    rows += [_row("a", 20, "error", False, error_kind="scenario"),
             _row("b", 20, "error", False, error_kind="scenario")]
    # Seed 21: controller crash for b -> a failure of b, paired.
    rows += [_row("a", 21), _row("b", 21, "error", False, error_kind="controller")]
    # Seed 22: a harness crash for a, retried on resume -> last row counts.
    rows += [_row("a", 22, "error", False, error_kind="harness"), _row("b", 22), _row("a", 22)]
    d = _write(tmp_path, rows)
    table, pt, speed = analyze.analyze(d, EDGES, verbose=False)

    a = table[table.planner == "a"].iloc[0]
    b = table[table.planner == "b"].iloc[0]
    assert a.n == 22 and b.n == 22  # 20 + seed 21 + seed 22; seed 20 excluded
    assert b.controller_crashes == 1 and a.scenario_crashes == 1
    assert a.harness_crashes == 0  # the retried row superseded the crash
    assert a.success == 1.0 and b.success == pytest.approx(1 / 22)

    row = pt.iloc[0]
    assert row.n_pairs == 22 and (row.a_only, row.b_only) == (21, 0)
    assert row.p_mcnemar < 1e-5 and row.loss_diff < 0
    assert row.n_both_contact == 1  # only seed 22
    assert row.rot_err_diff == 0.0 and row.p_rot_err == 1.0  # one identical pair
    assert all(0 <= row[c] <= 1 or np.isnan(row[c]) for c in pt.columns if c.endswith("holm"))

    trials = speed[speed.kind == "trials"]
    for _, g in trials.groupby("planner"):
        assert g.n.sum() == 22  # every counted trial is in some speed bin
    crashes = speed[speed.kind != "trials"]
    # 2 scenario crashes + 1 controller crash; the harness crash was retried.
    assert crashes.n.sum() == 3


def test_out_of_range_speeds_are_binned_not_dropped(tmp_path):
    rows = [_row("a", s, speed=v) for s, v in enumerate([0.10, 0.2, 0.5, 0.95])]
    d = _write(tmp_path, rows)
    _, _, speed = analyze.analyze(d, EDGES, verbose=False)
    assert speed[speed.kind == "trials"].n.sum() == 4


def test_all_crashed_does_not_crash_the_analysis(tmp_path):
    rows = [_row(p, s, "error", False, error_kind="controller") for p in "ab" for s in range(3)]
    rows += [_row("c", 0, "error", False, error_kind="harness")]
    d = _write(tmp_path, rows)
    table, pt, _ = analyze.analyze(d, EDGES, verbose=False)
    assert (table[table.planner != "c"].success == 0).all()
    assert table[table.planner == "c"].iloc[0].n == 0
    # Paired: identical failures, so nothing to test and no fake p-values.
    assert (pt.a_only == 0).all() and (pt.b_only == 0).all()
    assert (pt.p_mcnemar == 1.0).all()
    assert pt.p_rot_err_holm.isna().all()
