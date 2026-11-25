"""Evaluate a learned policy on experiment seeds, scored exactly like a trial.

    result = run_policy_trial(cfg, Condition("student", 1.0), seed, policy)

The policy acts through the RL env (dap/envs), which shares the trial's arm
limits, event rule and horizon (dap.trial). The env replays seed `seed` of
the config's split (`reset(options={"trial_seed": seed, "split": cfg.split})`:
the same scenario, sensor and depth-noise streams as dap.trial.run_trial), records the trial's
ground-truth log, and the log is scored by dap.scoring.evaluate with the
same live/offline agreement check. The resulting TrialResult rows have the
same fields and covariates as a planner's, so scripts/analyze.py compares
them on paired seeds. Nothing the policy reports is scored.

A policy is any object with
    obs_mode: str          # an env observation mode, normally "pixels"
    __call__(obs) -> (7,)  # action in [-1, 1], fraction of joint speed limits
    reset()                # optional, called at the start of every trial

`cond.noise_scale` scales the stand-in pose sensor ("state" modes); it does
not change the depth noise a pixel policy sees. `n_steps`, `step_ms_*` and
`n_bad_commands` count policy calls (20 Hz), not 100 Hz controller steps.
"""

import time

from dap.envs.dap_env import DapEnv
from dap.experiment import Condition, ExperimentConfig
from dap.metrics import TrialResult
from dap.trial import record_crash, record_scenario, record_timing, score_log


def make_eval_env(cfg: ExperimentConfig, cond: Condition, obs_mode, **env_kwargs):
    """An env with the experiment's timing, tolerances and scenario preset,
    recording the ground-truth log."""
    return DapEnv(obs_mode=obs_mode, scenario=cfg.scenario, noise_scale=cond.noise_scale,
                  sim_dt=cfg.dt, t_extra=cfg.t_extra, touch_dist=cfg.tolerances.touch_dist,
                  rot_tol=cfg.tolerances.rot_tol, record_log=True, **env_kwargs)


def _check_env(env: DapEnv, cfg: ExperimentConfig, cond: Condition):
    want = dict(scenario_name=cfg.scenario, sim_dt=cfg.dt, t_extra=cfg.t_extra,
                touch_dist=cfg.tolerances.touch_dist, rot_tol=cfg.tolerances.rot_tol,
                noise_scale=cond.noise_scale, record_log=True)
    bad = {k: (getattr(env, k), v) for k, v in want.items() if getattr(env, k) != v}
    if bad:
        raise ValueError(f"eval env does not match the experiment (env, experiment): {bad}")


def run_policy_trial(cfg: ExperimentConfig, cond: Condition, seed: int, policy, env=None,
                     env_kwargs=None, allow_privileged=False, return_log=False, frozen_scenario=None):
    """One trial of `policy` on experiment seed `seed`. A crash inside the
    policy is a trial outcome (error_kind "controller"), as in run_trial.
    Policies that see ground truth ("privileged") are references, not
    methods, and must be asked for explicitly."""
    obs_mode = policy.obs_mode
    if obs_mode == "privileged" and not allow_privileged:
        raise ValueError("a privileged policy sees ground truth; pass allow_privileged=True "
                         "to evaluate it as a reference")
    if env is None:
        env = make_eval_env(cfg, cond, obs_mode, **(env_kwargs or {}))
    elif env.obs_mode != obs_mode:
        raise ValueError(f"env obs_mode {env.obs_mode!r} != policy obs_mode {obs_mode!r}")
    _check_env(env, cfg, cond)

    result = TrialResult(seed=seed, planner=cond.planner, noise_scale=cond.noise_scale)
    wall0 = time.perf_counter()
    obs, _ = env.reset(options={"trial_seed": int(seed), "split": cfg.split,
                                "frozen_scenario": frozen_scenario})
    record_scenario(result, env.scenario)

    def policy_crashed(what):
        record_crash(result, what, wall0)
        return (result, env.log) if return_log else result

    if hasattr(policy, "reset"):
        try:
            policy.reset()
        except Exception:
            return policy_crashed("policy reset failed")

    step_ms = []
    done = env.event is not None  # a contact or collision already at t = 0
    while not done:
        c0 = time.perf_counter()
        try:
            action = policy(obs)
        except Exception:
            return policy_crashed(f"policy failed at t={env.t:.2f}")
        step_ms.append(1e3 * (time.perf_counter() - c0))
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    result.n_bad_commands = env.n_bad_actions
    score_log(result, env.log, env.motion, cfg.tolerances, env.v_max, env.event)
    record_timing(result, step_ms, wall0)
    return (result, env.log) if return_log else result
