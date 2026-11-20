import numpy as np

from dap.estimator import PalmEstimator
from dap.partner import NoisyPoseSensor


def test_tracks_partner_motion(scenario):
    motion = scenario.partner
    sensor = NoisyPoseSensor(motion, np.random.default_rng(0))
    est = PalmEstimator(pos_noise=sensor.noise_pos)

    dt = 0.01
    t_end = 0.6 * motion.t_meet
    for t in np.arange(0, t_end, dt):
        est.update(t, *sensor.measure(t))

    p_now, v_now, _ = est.predict(0.0)
    assert np.linalg.norm(p_now - motion.p(est.t)) < 0.01
    # The constant-acceleration model lags a decelerating min-jerk reach, so
    # velocity is only roughly right mid-approach (a known model mismatch).
    assert np.linalg.norm(v_now - motion.v(est.t)) < 0.15

    # Uncertainty grows with the prediction horizon.
    _, _, P_near = est.predict(0.1)
    _, _, P_far = est.predict(0.5)
    assert np.trace(P_far) > np.trace(P_near)


def test_estimates_the_palm_frame(scenario):
    motion = scenario.partner
    sensor = NoisyPoseSensor(motion, np.random.default_rng(1))
    est = PalmEstimator(pos_noise=sensor.noise_pos)
    for t in np.arange(0, 0.5 * motion.t_meet, 0.01):
        est.update(t, *sensor.measure(t))
    R = est.rotation()
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)
    R_true = motion.rotation(est.t)
    angle = np.degrees(np.arccos(np.clip((np.trace(R.T @ R_true) - 1) / 2, -1, 1)))
    assert angle < 5.0
