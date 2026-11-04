import numpy as np

from dap.estimator import PalmEstimator
from dap.partner import NoisyPoseSensor, PartnerPath, sample_params


def test_tracks_partner_path():
    rng = np.random.default_rng(0)
    path = PartnerPath(sample_params(rng))
    sensor = NoisyPoseSensor(path, rng)
    est = PalmEstimator(pos_noise=path.params.noise_pos)

    dt = 0.01
    t_end = 0.6 * path.params.duration
    for t in np.arange(0, t_end, dt):
        est.update(t, *sensor.measure(t))

    p_now, v_now, _ = est.predict(0.0)
    assert np.linalg.norm(p_now - path.p(est.t)) < 0.01
    assert np.linalg.norm(v_now - path.v(est.t)) < 0.1

    # Uncertainty grows with the prediction horizon.
    _, _, P_near = est.predict(0.1)
    _, _, P_far = est.predict(0.5)
    assert np.trace(P_far) > np.trace(P_near)
