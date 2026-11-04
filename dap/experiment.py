"""Experiment definitions: what to sweep, and the named configs we run.

A config expands to (Condition, seed) trials. Every condition uses the same
seeds, and a seed fixes the partner path, sensor noise, and checkpoints, so
"plan once" and "replan" are compared on identical approaches.
"""

import itertools
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class Condition:
    planner: str  # "once" | "replan"
    speed_scale: float  # > 1 = faster partner
    noise_scale: float  # multiplies the sensor noise std
    n_checkpoints: int
    velocity_match: bool = True  # match partner hand velocity at contact

    def key(self):
        return asdict(self)


@dataclass
class ExperimentConfig:
    name: str
    planners: tuple = ("once", "replan")
    speed_scales: tuple = (1.0,)
    noise_scales: tuple = (1.0,)
    checkpoint_counts: tuple = (0,)
    velocity_match: tuple = (True,)
    n_seeds: int = 50
    first_seed: int = 0

    # Simulation
    dt: float = 0.01  # s, sensor + control period
    t_extra: float = 1.0  # s, keep simulating past the partner's nominal arrival
    replan_hz: float = 10.0
    handoff_dist: float = 0.04  # m, palm gap at which the clasp controller takes over

    # Success tolerances. TODO: placeholders until we settle proposal section 7.
    contact_dist: float = 0.01  # m, palm gap that counts as "touching"
    success_pos_tol: float = 0.03  # m, palm-center offset in the palm plane
    success_angle_tol: float = 15.0  # deg, between palm normals
    vel_window: float = 0.2  # s, window before contact for velocity mismatch

    notes: str = ""
    extra: dict = field(default_factory=dict)

    def conditions(self):
        for planner, speed, noise, n_ck, vm in itertools.product(
            self.planners,
            self.speed_scales,
            self.noise_scales,
            self.checkpoint_counts,
            self.velocity_match,
        ):
            yield Condition(planner, speed, noise, n_ck, vm)

    def seeds(self):
        return range(self.first_seed, self.first_seed + self.n_seeds)

    def trials(self):
        for cond in self.conditions():
            for seed in self.seeds():
                yield cond, seed

    def n_trials(self):
        return sum(1 for _ in self.conditions()) * self.n_seeds


def trial_rngs(seed):
    """Independent streams for partner path, sensor noise, and checkpoints."""
    return {
        "partner": np.random.default_rng([seed, 0]),
        "sensor": np.random.default_rng([seed, 1]),
        "checkpoints": np.random.default_rng([seed, 2]),
    }


CONFIGS = {
    # Plumbing check: a handful of easy trials end to end.
    "smoke": ExperimentConfig(
        name="smoke",
        planners=("replan",),
        n_seeds=3,
        notes="End-to-end plumbing check, not a result.",
    ),
    # H1: replanning beats planning once when there is time to react.
    "replan_vs_once": ExperimentConfig(
        name="replan_vs_once",
        planners=("once", "replan"),
        speed_scales=(0.75, 1.0, 1.5),
        noise_scales=(0.5, 1.0, 2.0),
        checkpoint_counts=(0, 2),
        n_seeds=50,
        notes="H1. Paired seeds across planners.",
    ),
    # H2: fast partners and many checkpoints hit joint velocity limits.
    "speed_limits": ExperimentConfig(
        name="speed_limits",
        planners=("replan",),
        speed_scales=(0.5, 1.0, 1.5, 2.0, 3.0),
        checkpoint_counts=(0, 2, 4),
        n_seeds=30,
        notes="H2. Look at infeasible replans and success vs speed x checkpoints.",
    ),
    # H3: velocity matching at contact makes the clasp forgiving of timing error.
    "velocity_match": ExperimentConfig(
        name="velocity_match",
        planners=("replan",),
        speed_scales=(1.0, 1.5),
        noise_scales=(1.0, 2.0),
        velocity_match=(True, False),
        n_seeds=50,
        notes="H3. Compare clasp success and time_err with/without velocity matching.",
    ),
}
