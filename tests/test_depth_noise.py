import numpy as np

from dap.depth_noise import NOISE_PRESETS, DepthNoiseModel, DepthNoiseParams, axial_rms

FX = FY = 433.0
CX, CY = 211.5, 119.5
SHAPE = (120, 212)


def _quiet(**kw):
    """Only the axial noise, everything else off."""
    base = dict(sigma_shift=0.0, sigma_pattern=0.0, sigma_scale=0.0, theta_max_deg=None,
                shadow=False, hole_prob=0.0, n_blobs=0, quantum=0.0)
    return DepthNoiseParams(**{**base, **kw})


def _model(params, seed=0):
    m = DepthNoiseModel(params, FX, FY, CX, CY)
    m.reset(np.random.default_rng(seed), SHAPE)
    return m


def test_axial_noise_matches_intel_formula():
    for z in (0.5, 1.0, 1.5, 2.5):
        m = _model(_quiet())
        out = m.apply(np.full(SHAPE, z))
        expected = axial_rms(z, FX)
        assert abs(out.std() - expected) / expected < 0.1
        assert abs(out.mean() - z) < 3 * expected


def test_range_gate_and_invalid_inputs_become_zero():
    m = _model(_quiet())
    depth = np.full(SHAPE, 1.0)
    depth[0, :10] = 0.1  # closer than Min-Z
    depth[1, :10] = 4.0  # beyond max range
    depth[2, :10] = np.inf  # Drake's "too far"
    out = m.apply(depth)
    assert np.all(out[:3, :10] == 0)
    assert np.all(out[3:] > 0)


def test_quantized_to_millimetres():
    m = _model(_quiet(quantum=0.001))
    out = m.apply(np.full(SHAPE, 1.2345)).astype(np.float64)
    valid = out[out > 0]
    assert np.allclose(valid * 1000, np.round(valid * 1000), atol=1e-3)


def test_fixed_pattern_and_scale_are_constant_within_an_episode():
    m = _model(_quiet(sigma_subpixel=0.0, sigma_pattern=0.2, sigma_scale=0.02))
    a = m.apply(np.full(SHAPE, 1.0))
    b = m.apply(np.full(SHAPE, 1.0))
    assert np.array_equal(a, b)  # no per-frame noise -> identical frames
    assert a.std() > 0  # but the pattern is not flat
    m.reset(np.random.default_rng(1), SHAPE)
    assert not np.array_equal(a, m.apply(np.full(SHAPE, 1.0)))


def test_holes_and_shadow():
    m = _model(_quiet(hole_prob=0.05, n_blobs=0))
    out = m.apply(np.full(SHAPE, 1.0))
    assert 0.03 < (out == 0).mean() < 0.07

    # A 0.6 m foreground block on a 1.2 m background: the stereo shadow lies
    # to its left, f*B*(1/0.6 - 1/1.2) ~ 18 px wide.
    depth = np.full(SHAPE, 1.2)
    depth[:, 100:150] = 0.6
    out = _model(_quiet(shadow=True, sigma_subpixel=0.0)).apply(depth)
    width = int(round(FX * 0.05 * (1 / 0.6 - 1 / 1.2)))
    assert np.all(out[:, 100 - width + 1:100] == 0)
    assert np.all(out[:, 60:100 - width - 1] > 0)
    assert np.all(out[:, 150:] > 0)  # nothing to the right


def test_grazing_angle_dropout():
    # Plane z = 1 + tan(80 deg) * X, normal (-tan80, 0, 1). The angle between
    # each pixel's ray (x, y, 1) and that normal varies across the image.
    v, u = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    x = (u - CX) / FX
    y = (v - CY) / FY
    k = np.tan(np.deg2rad(80))
    z = 1.0 / (1.0 - k * x)
    z[(z < 0.2) | (z > 3)] = 0
    n = np.array([-k, 0.0, 1.0]) / np.hypot(k, 1.0)
    rays = np.stack([x, y, np.ones_like(x)], -1)
    cos = np.abs(rays @ n) / np.linalg.norm(rays, axis=-1)
    angle = np.degrees(np.arccos(cos))
    out = _model(_quiet(theta_max_deg=75.0)).apply(z)
    inner = (z > 0)
    inner[[0, -1], :] = False  # gradients are one-sided on the border
    inner[:, [0, -1]] = False
    grazing = inner & (angle > 77)
    facing = inner & (angle < 73)
    assert grazing.sum() > 100 and facing.sum() > 100
    assert (out[grazing] == 0).mean() > 0.95
    assert (out[facing] == 0).mean() < 0.05


def test_presets_run_and_are_seeded():
    depth = np.full(SHAPE, 1.0)
    depth[:, 80:130] = 0.7
    for name, params in NOISE_PRESETS.items():
        if params is None:
            continue
        a = _model(params, seed=3).apply(depth)
        b = _model(params, seed=3).apply(depth)
        assert a.dtype == np.float32 and a.shape == SHAPE
        assert np.array_equal(a, b)
        assert a.min() >= 0 and a.max() <= params.max_z
