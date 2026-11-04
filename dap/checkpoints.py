"""Positional checkpoints our hand must pass, in order, before the meeting."""

import numpy as np


def sample_checkpoints(rng, n, p_start, p_goal, spread=0.08):
    """`n` points spaced between our palm's start and the nominal meeting
    point, jittered by `spread` m, ordered along the way. Returns (n, 3)."""
    if n == 0:
        return np.zeros((0, 3))
    s = np.sort(rng.uniform(0.2, 0.8, n))
    return p_start + s[:, None] * (p_goal - p_start) + rng.normal(0, spread, (n, 3))
