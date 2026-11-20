"""Normalized approach and release travel fitted to an Inter-X handshake preview."""

import numpy as np


def _quintic(u):
    u = np.clip(u, 0., 1.)
    return u**3 * (10. + u * (-15. + 6. * u))


def progress(u, weight, first_end, second_start):
    """Blend two overlapping minimum-jerk movements, both at rest at the ends."""
    return (weight * _quintic(u / first_end)
            + (1. - weight) * _quintic((u - second_start) / (1. - second_start)))


def reach_progress(u):
    return progress(u, .23117829, .72606245, .46351567)


def release_progress(u):
    return progress(u, .33309918, .19852113, .05612635)
