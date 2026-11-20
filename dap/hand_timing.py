"""Inward swing timing from the opening strike of a filmed right-hand dap."""

import numpy as np

# Fitted to frame annotations in scripts/fit_hand_timing.py.
SWING_COEFFICIENT = 0.64369713


def dap_swing_progress(u, coefficient=SWING_COEFFICIENT):
    """Travel from rest to contact, keeping inward speed at the strike."""
    u = np.clip(u, 0.0, 1.0)
    return u**2 * (1.0 + coefficient * (1.0 - u))
