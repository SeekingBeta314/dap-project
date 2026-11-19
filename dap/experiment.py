"""Experiment definitions: what to sweep, and the named configs we run.

A config expands to (Condition, seed) trials. Every condition uses the same
seeds, and a seed fixes the whole scenario (layout, partner path and speed,
our start configuration) and the sensor noise stream, so conditions are
compared on identical scenarios (a paired design; analysis in
scripts/analyze.py and dap/stats.py).

Partner speed is not a condition: every scenario draws its own target peak
palm speed (dap.partner.PathConfig.peak_speed), so a run covers the whole
speed range and results are analysed against the realized speed.
"""

import itertools
from dataclasses import asdict, dataclass, field

import numpy as np

from dap.scoring import Tolerances


@dataclass(frozen=True)
class Condition:
    planner: str  # "once" | "replan" | a baseline name
    noise_scale: float  # multiplies the stand-in sensor's noise std

    def key(self):
        return asdict(self)


@dataclass
class ExperimentConfig:
    name: str
    planners: tuple = ("once", "replan")
    noise_scales: tuple = (1.0,)
    n_seeds: int = 50
    first_seed: int = 0
    scenario: str = "nominal"  # key into dap.scenario.SCENARIO_PRESETS

    # Simulation
    dt: float = 0.01  # s, sensor + control period
    t_extra: float = 1.0  # s, keep simulating past the end of the partner's motion

    replan_hz: float = 10.0

    # How a dap is judged (dap.scoring). Placeholders (Ours) until the team
    # settles them.
    tolerances: Tolerances = Tolerances()

    notes: str = ""
    extra: dict = field(default_factory=dict)

    def conditions(self):
        for planner, noise in itertools.product(self.planners, self.noise_scales):
            yield Condition(planner, noise)

    def seeds(self):
        return range(self.first_seed, self.first_seed + self.n_seeds)

    def trials(self):
        for cond in self.conditions():
            for seed in self.seeds():
                yield cond, seed

    def n_trials(self):
        return sum(1 for _ in self.conditions()) * self.n_seeds


def trial_rngs(seed):
    """Independent streams for the scenario (layout, partner motion and
    speed, our start configuration) and the sensor noise."""
    return {
        "scenario": np.random.default_rng([seed, 0]),
        "sensor": np.random.default_rng([seed, 1]),
    }


CONFIGS = {
    # Plumbing check: a handful of trials end to end.
    "smoke": ExperimentConfig(
        name="smoke",
        planners=("replan",),
        n_seeds=3,
        notes="End-to-end plumbing check, not a result.",
    ),
    # H1: replanning beats planning once. 150 paired seeds per noise level
    # give ~80% power (alpha 0.05, McNemar) for a 10-point success-rate
    # difference when ~20% of pairs are discordant (docs/setup.md).
    "replan_vs_once": ExperimentConfig(
        name="replan_vs_once",
        planners=("once", "replan"),
        noise_scales=(0.5, 1.0, 2.0),
        n_seeds=150,
        notes="H1. Paired seeds across planners; McNemar per noise level.",
    ),
    # H2: success falls with partner speed once our joint velocity limits
    # bind. One planner, many seeds, analysed against realized partner speed.
    "speed_limits": ExperimentConfig(
        name="speed_limits",
        planners=("replan",),
        n_seeds=300,
        notes="H2. Success / loss vs. realized partner peak speed (binned).",
    ),
}
