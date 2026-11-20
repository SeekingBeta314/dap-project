import numpy as np

from dap.pixel_policy import PixelPolicy, pool_depth


def test_depth_pool_filters_missing_returns_and_preserves_image_location():
    frame = np.zeros((120, 212), np.float32)
    frame[5, 6], frame[6, 5] = 1.8, 0.8
    pooled = pool_depth(frame)
    np.testing.assert_array_equal(pooled.shape, (30, 53))
    assert pooled[0, 0] == 3.0
    assert np.isclose(pooled[1, 1], 0.8)


def make_checkpoint(path, vision):
    # Choose weights implementing one positive scalar signal, replicated on
    # every output. The visual policy reads the first pixel and the blind
    # policy reads the first joint of the episode's initial configuration.
    size = 15 + (30 * 53 if vision else 0)
    weights = np.zeros((size, 1), np.float32)
    weights[15 if vision else 0] = 1
    np.savez(path, vision=vision, x_mean=np.zeros(size), x_scale=np.ones(size),
             y_mean=np.zeros(7), y_scale=np.ones(7), v_max=np.ones(7),
             weight0=weights, bias0=np.zeros(1), weight1=np.ones((1, 1)),
             bias1=np.zeros(1), weight2=np.ones((1, 7)), bias2=np.zeros(7))


def test_visual_action_responds_to_depth_and_waits_for_first_frame(tmp_path):
    path = tmp_path / "depth.npz"
    make_checkpoint(path, True)
    student = PixelPolicy(path)
    depth = np.zeros((3, 120, 212), np.float32)
    obs = {"proprio": np.zeros(31), "depth": depth}
    np.testing.assert_array_equal(student(obs), np.zeros(7))
    depth[-1, :4, :4] = .11
    np.testing.assert_allclose(student(obs), np.full(7, .5), rtol=1e-6)
    depth[-1, :4, :4] = 2
    np.testing.assert_array_equal(student(obs), np.ones(7))


def test_proprioception_ignores_camera_and_resets_between_episodes(tmp_path):
    path = tmp_path / "proprio.npz"
    make_checkpoint(path, False)
    policy = PixelPolicy(path)
    state = np.full(31, .11, np.float32)
    policy({"proprio": state})
    state.fill(0)
    np.testing.assert_allclose(policy({"proprio": state}), np.full(7, .5), rtol=1e-6)
    policy.reset()
    np.testing.assert_array_equal(policy({"proprio": state}), np.zeros(7))


def test_shipped_students_match_manifests_and_act_from_observations():
    import hashlib
    import json

    from dap.pixel_policy import CHECKPOINTS, pixel_factory, proprio_factory

    depth = np.random.default_rng(4).uniform(.2, 2.5, (3, 120, 212)).astype("float32")
    obs = {"proprio": np.zeros(31, dtype="float32"), "depth": depth}
    for name, factory in [("depth", pixel_factory), ("proprio", proprio_factory)]:
        model = CHECKPOINTS / f"{name}.npz"
        meta = json.loads(model.with_suffix(".json").read_text())
        assert hashlib.sha256(model.read_bytes()).hexdigest() == meta["checkpoint_sha256"]
        student = factory(None, None)
        action = student(obs)
        assert action.shape == (7,) and np.isfinite(action).all()
        assert np.max(np.abs(action)) <= 1.


def test_convolutional_export_preserves_stride_padding_and_channel_order():
    from dap.pixel_policy import conv_features

    rng = np.random.default_rng(12)
    x = rng.uniform(1.5, 2.2, (56, 68)).astype("float32")
    params = {}
    for i, (size, n_in) in enumerate([(5, 1), (3, 2), (3, 2)]):
        kernel = np.zeros((2, n_in, size, size), dtype="float32")
        kernel[0, 0, size // 2, size // 2] = 1
        kernel[1, n_in - 1, size // 2, size // 2] = 2 if i == 0 else 1
        params[f"kernel{i}"] = kernel
        params[f"offset{i}"] = np.zeros(2, dtype="float32")
    base = ((x[::8, ::8] - 1.5) / .75).ravel()
    np.testing.assert_allclose(conv_features(x, params), np.r_[base, 2 * base])
