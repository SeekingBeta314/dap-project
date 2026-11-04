from types import SimpleNamespace

import numpy as np

from dap.experiment import CONFIGS, ExperimentConfig, trial_rngs
from dap.partner import PartnerPath, sample_params
from dap.trial import TrialLog, score
from dap.metrics import TrialResult


def test_config_expands_to_full_grid():
    cfg = CONFIGS["replan_vs_once"]
    trials = list(cfg.trials())
    assert len(trials) == cfg.n_trials() == 2 * 3 * 3 * 2 * cfg.n_seeds


def test_seed_fixes_partner_path():
    a = PartnerPath(sample_params(trial_rngs(7)["partner"]))
    b = PartnerPath(sample_params(trial_rngs(7)["partner"]))
    assert np.allclose(a.p(0.5), b.p(0.5))


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
                   phase="approach")
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
