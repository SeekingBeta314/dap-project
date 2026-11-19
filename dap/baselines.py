"""Scripted controllers that implement the controller contract in
dap/planner.py. They are references, not methods: the oracle uses ground
truth.

    run_trial(cfg, cond, seed, controller_factory=oracle_factory)
"""

import numpy as np

from dap.partner import min_jerk


class OracleController:
    """Knows the scenario: moves our arm with a joint-space min-jerk profile
    from its start to `scenario.q_ours_meet` (our side of the certified dap),
    arriving at the partner's true meeting time, then holds. Not
    collision-checked along the way, and its commands are clipped to the
    joint speed limits like anyone's (so it can arrive late when the dap is
    not reachable in time). An upper reference, not a planner."""

    def __init__(self, scenario):
        self.q_meet = np.asarray(scenario.q_ours_meet, float)
        self.t_meet = scenario.partner.t_meet

    def reset(self, model, q0):
        self.q0 = np.asarray(q0, float).copy()
        self.dt = model.dt

    def step(self, t, q, estimator):
        s = min_jerk((t + self.dt) / self.t_meet)
        return self.q0 + (self.q_meet - self.q0) * s


class HoldController:
    """Never moves. The floor: scenarios are sampled so the partner never
    touches a resting arm, so holding still always times out."""

    def reset(self, model, q0):
        self.q0 = np.asarray(q0, float).copy()

    def step(self, t, q, estimator):
        return self.q0.copy()


def oracle_factory(cfg, cond, scenario):
    return OracleController(scenario)


def hold_factory(cfg, cond, scenario):
    return HoldController()
