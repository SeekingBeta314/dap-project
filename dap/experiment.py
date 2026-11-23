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
    # Seed split (SPLITS): "eval" for reported results, "val" for checks
    # and selection. Evaluation configs start at EVAL_FIRST_SEED or later.
    split: str = "eval"
    first_seed: int | None = None  # None: EVAL_FIRST_SEED for "eval", else 0
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

    def __post_init__(self):
        if self.split not in ("eval", "val"):
            raise ValueError(f"split must be 'eval' or 'val', not {self.split!r}")
        if self.first_seed is None:
            self.first_seed = EVAL_FIRST_SEED if self.split == "eval" else 0
        if self.split == "eval" and self.first_seed < EVAL_FIRST_SEED:
            raise ValueError(f"evaluation seeds below {EVAL_FIRST_SEED} are spent; "
                             "use split='val' for checks")

    def rngs(self, seed):
        """The streams of `seed` in this config's split."""
        return SPLITS[self.split](seed)

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


# Three seed splits, disjoint by construction at any number of seeds
# (docs/setup.md section 9):
#   train  learning (dap.bank, fresh env scenarios, the blind prior)
#   val    development checks, tuning, checkpoint and hyperparameter selection
#   eval   reported results only, from EVAL_FIRST_SEED on
# Evaluation streams have entropy [seed, k] with seed < 2**32, so at most two
# 32-bit words; training and validation streams end in their own nonzero tag
# word ([seed, k, TAG]), so they are at least three words long and differ in
# the last one, whatever the seed.
TRAIN_TAG = 0x7A11
VAL_TAG = 0x7A12
# Evaluation seeds below this are spent: before the validation split existed,
# development used evaluation seeds (0-15 for the baselines and perception,
# up to a few hundred for setup measurements), so no result comes from them.
EVAL_FIRST_SEED = 1000


def trial_rngs(seed):
    """Independent streams for the scenario (layout, partner motion and
    speed, our start configuration), the stand-in sensor's noise, and the
    head camera's depth noise. Experiments (evaluation) use these."""
    if not 0 <= seed < 2**32:
        raise ValueError(f"evaluation seeds lie in [0, 2**32), got {seed}")
    return {
        "scenario": np.random.default_rng([seed, 0]),
        "sensor": np.random.default_rng([seed, 1]),
        "camera": np.random.default_rng([seed, 2]),
    }


def _tagged_rngs(seed, tag):
    return {name: np.random.default_rng([seed, k, tag])
            for k, name in enumerate(("scenario", "sensor", "camera"))}


def train_rngs(seed):
    """Like `trial_rngs`, for training data: the same kinds of streams,
    disjoint from every evaluation and validation seed."""
    return _tagged_rngs(seed, TRAIN_TAG)


def val_rngs(seed):
    """Like `trial_rngs`, for validation: disjoint from every evaluation and
    training seed."""
    return _tagged_rngs(seed, VAL_TAG)


SPLITS = {"train": train_rngs, "val": val_rngs, "eval": trial_rngs}


CONFIGS = {
    # Plumbing check: a handful of trials end to end.
    "smoke": ExperimentConfig(
        name="smoke",
        planners=("replan",),
        n_seeds=3,
        split="val",
        notes="End-to-end plumbing check, not a result.",
    ),
    # Do the baselines work? (docs/setup.md section 1b; run with --controller.)
    "first_look": ExperimentConfig(
        name="first_look",
        planners=("replan",),
        n_seeds=12,
        split="val",
        notes="Baseline check on validation seeds, not a result.",
    ),
    "first_look_wide": ExperimentConfig(
        name="first_look_wide",
        planners=("replan",),
        n_seeds=12,
        split="val",
        scenario="wide",
        notes="Baseline check on validation seeds, not a result.",
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
