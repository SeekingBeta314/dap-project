"""The partner hand: a randomized approach path plus a noisy pose sensor.

Only `NoisyPoseSensor` output may be used by our controller; the true
`PartnerPath` is for the partner arm and for scoring.
"""

from dataclasses import dataclass

import numpy as np
from pydrake.all import PiecewisePolynomial


@dataclass
class PartnerParams:
    """Randomized per trial."""

    duration: float  # s, start of approach to the nominal meeting point
    start: np.ndarray  # (3,) partner palm start position, world frame
    meet: np.ndarray  # (3,) where the partner palm would end up if unopposed
    bulge: np.ndarray  # (3,) offset of the path midpoint (sets curvature)
    noise_pos: float  # m, std of position measurement noise
    noise_ang: float  # rad, std of palm-normal measurement noise


def sample_params(rng, speed_scale=1.0, noise_scale=1.0):
    """Draw a random approach. `speed_scale` > 1 means a faster partner."""
    meet = np.array([0.7, 0.0, 0.6]) + rng.uniform(-0.1, 0.1, 3)
    start = meet + np.array([0.15, 0, 0]) + rng.uniform([0.0, -0.3, -0.2], [0.1, 0.3, 0.2])
    return PartnerParams(
        duration=rng.uniform(1.5, 3.0) / speed_scale,
        start=start,
        meet=meet,
        bulge=rng.normal(0, 0.08, 3),
        noise_pos=0.005 * noise_scale,
        noise_ang=np.deg2rad(2.0) * noise_scale,
    )


class PartnerPath:
    """Partner palm position p(t) and palm normal n(t) in the world frame."""

    def __init__(self, params: PartnerParams):
        self.params = params
        mid = 0.5 * (params.start + params.meet) + params.bulge
        times = [0.0, 0.5 * params.duration, params.duration]
        knots = np.column_stack([params.start, mid, params.meet])
        self.position = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            times, knots, np.zeros(3), np.zeros(3)
        )
        self.velocity = self.position.derivative(1)

    def p(self, t):
        return self.position.value(np.clip(t, 0, self.params.duration)).ravel()

    def v(self, t):
        if t < 0 or t > self.params.duration:
            return np.zeros(3)
        return self.velocity.value(t).ravel()

    def normal(self, t):
        # Nominal palm normal faces back toward our arm (-x).
        # TODO: randomize the approach angle and let it rotate along the path.
        return np.array([-1.0, 0.0, 0.0])


class NoisyPoseSensor:
    """What our controller gets: noisy palm position and normal."""

    def __init__(self, path: PartnerPath, rng):
        self.path = path
        self.rng = rng

    def measure(self, t):
        prm = self.path.params
        p = self.path.p(t) + self.rng.normal(0, prm.noise_pos, 3)
        n = self.path.normal(t) + self.rng.normal(0, prm.noise_ang, 3)
        return p, n / np.linalg.norm(n)


# TODO: drive the partner iiwa along PartnerPath (differential IK or an IK
# solve per knot, then a joint-space spline) so its arm is in the scene.
