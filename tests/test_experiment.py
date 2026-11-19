from types import SimpleNamespace

import numpy as np

from dap.experiment import CONFIGS, ExperimentConfig, trial_rngs
from dap.metrics import TrialResult
from dap.scenario import SCENARIO_PRESETS, sample_scenario
from dap.trial import TrialLog, score


def test_config_expands_to_full_grid():
    cfg = CONFIGS["replan_vs_once"]
    trials = list(cfg.trials())
    assert len(trials) == cfg.n_trials() == 2 * 3 * 3 * 2 * cfg.n_seeds


def test_seed_fixes_scenario():
    cfg = SCENARIO_PRESETS["nominal"]
    a = sample_scenario(trial_rngs(7)["scenario"], cfg)
    b = sample_scenario(trial_rngs(7)["scenario"], cfg)
    assert a.layout == b.layout
    assert np.allclose(a.q_ours0, b.q_ours0)
    assert np.allclose(a.partner.p(0.5), b.partner.p(0.5))


def _fake_controller(**kw):
    base = dict(plan_times_ms=[5.0], n_replans=3, n_infeasible=0,
                t_meet_pred=1.0, clasp_held=True)
    return SimpleNamespace(**{**base, **kw})


def _log_meeting_at(t_meet, offset=np.zeros(3)):
    """Our palm moves in +x and meets a static partner palm at x=0.7, t=t_meet."""
    log = TrialLog()
    for t in np.arange(0.0, t_meet + 0.3, 0.01):
        x = min(0.7, 0.5 + 0.2 * t / t_meet)
        log.append(t=t, q=np.zeros(7), palm_p=np.array([x, 0, 0.6]) + offset,
                   palm_n=np.array([1.0, 0, 0]), partner_p=np.array([0.7, 0, 0.6]),
                   partner_n=np.array([-1.0, 0, 0]), partner_v=np.zeros(3),
                   arm_dist=0.2, clearance=0.05, phase="approach")
    return log


def _result():
    return TrialResult(seed=0, planner="replan", speed_scale=1.0,
                       noise_scale=1.0, n_checkpoints=0)


def test_score_clean_meeting_succeeds():
    cfg = ExperimentConfig(name="t")
    r = score(cfg, _log_meeting_at(1.0), np.zeros((0, 3)), _fake_controller(), _result())
    assert r.contact and r.success
    assert r.angle_err < 1e-6 and r.pos_err < 1e-6
    assert abs(r.time_err) < 0.06  # contact_dist reached slightly before t_meet


def test_score_offset_palm_fails_position_tolerance():
    cfg = ExperimentConfig(name="t", contact_dist=0.1)
    log = _log_meeting_at(1.0, offset=np.array([0, 0.05, 0]))
    r = score(cfg, log, np.zeros((0, 3)), _fake_controller(), _result())
    assert r.contact and not r.success
    assert abs(r.pos_err - 0.05) < 1e-6


def test_score_collision_fails():
    cfg = ExperimentConfig(name="t")
    log = _log_meeting_at(1.0)
    log.clearance[5] = -0.001
    r = score(cfg, log, np.zeros((0, 3)), _fake_controller(), _result())
    assert r.contact and r.collision and not r.success
    assert abs(r.min_arm_dist - 0.2) < 1e-9


def test_contact_counts_pass_through_and_needs_overlap():
    from dap.metrics import palm_contact

    p_partner, n_partner = np.array([0.7, 0, 0.6]), np.array([-1.0, 0, 0])
    # 2 cm in front: no contact yet.
    assert not palm_contact([0.68, 0, 0.6], p_partner, n_partner, 0.01)[0]
    # A fast step that lands 1 cm *behind* the partner face still counts.
    hit, gap, in_plane = palm_contact([0.71, 0.0, 0.6], p_partner, n_partner, 0.01)
    assert hit and gap < 0 and in_plane < 1e-9
    # Touching the plane but 8 cm off to the side: palms do not overlap.
    assert not palm_contact([0.7, 0.08, 0.6], p_partner, n_partner, 0.01)[0]


def test_score_detects_step_over():
    cfg = ExperimentConfig(name="t")
    log = TrialLog()
    # Our palm jumps from 2 cm in front to 1.5 cm behind in one 10 ms step.
    for t, x in [(0.0, 0.5), (0.01, 0.68), (0.02, 0.715), (0.03, 0.72)]:
        log.append(t=t, q=np.zeros(7), palm_p=np.array([x, 0.004, 0.6]),
                   palm_n=np.array([1.0, 0, 0]), partner_p=np.array([0.7, 0, 0.6]),
                   partner_n=np.array([-1.0, 0, 0]), partner_v=np.zeros(3),
                   arm_dist=0.2, clearance=0.05, phase="approach")
    r = score(cfg, log, np.zeros((0, 3)), _fake_controller(t_meet_pred=0.02), _result())
    assert r.contact and abs(r.t_contact - 0.02) < 1e-9
    assert abs(r.pos_err - 0.004) < 1e-9 and r.success
