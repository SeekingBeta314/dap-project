"""Kalman filter on the partner palm: constant-acceleration position model
plus a low-pass estimate of the palm normal."""

import numpy as np


class PalmEstimator:
    """State x = [p, v, a] (9,), world frame."""

    def __init__(self, pos_noise=0.005, accel_jerk=5.0, normal_alpha=0.2):
        self.r = pos_noise  # measurement std (m)
        self.q = accel_jerk  # process noise: jerk spectral density
        self.alpha = normal_alpha  # palm-normal smoothing factor
        self.x = None
        self.P = None
        self.t = None
        self.normal = None

    @staticmethod
    def _F(dt):
        I = np.eye(3)
        return np.block([
            [I, dt * I, 0.5 * dt**2 * I],
            [0 * I, I, dt * I],
            [0 * I, 0 * I, I],
        ])

    def _Q(self, dt):
        # Discrete white-jerk process noise.
        I = np.eye(3)
        q = self.q
        return q * np.block([
            [dt**5 / 20 * I, dt**4 / 8 * I, dt**3 / 6 * I],
            [dt**4 / 8 * I, dt**3 / 3 * I, dt**2 / 2 * I],
            [dt**3 / 6 * I, dt**2 / 2 * I, dt * I],
        ])

    def update(self, t, p_meas, n_meas):
        if self.x is None:
            self.x = np.concatenate([p_meas, np.zeros(6)])
            self.P = np.diag([self.r**2] * 3 + [1.0] * 3 + [10.0] * 3)
            self.t = t
            self.normal = n_meas
            return
        dt = t - self.t
        self.t = t
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

        H = np.hstack([np.eye(3), np.zeros((3, 6))])
        S = H @ self.P @ H.T + self.r**2 * np.eye(3)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (p_meas - H @ self.x)
        self.P = (np.eye(9) - K @ H) @ self.P

        n = (1 - self.alpha) * self.normal + self.alpha * n_meas
        self.normal = n / np.linalg.norm(n)

    def predict(self, horizon):
        """Predicted (position, velocity, position covariance) `horizon` s ahead."""
        F = self._F(horizon)
        x = F @ self.x
        P = F @ self.P @ F.T + self._Q(horizon)
        return x[:3], x[3:6], P[:3, :3]
