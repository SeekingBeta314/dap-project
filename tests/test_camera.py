import numpy as np

from dap.camera import FrameBuffer, HeadCameraConfig, look_at
from dap.scene import OURS, PARTNER, Q_READY, Scene


def test_look_at_convention():
    X = look_at([0, 0, 1], [1, 0, 1])
    R = X.rotation().matrix()
    assert np.allclose(R[:, 2], [1, 0, 0])  # +z forward
    assert np.allclose(R[:, 1], [0, 0, -1])  # +y down
    assert np.allclose(R[:, 0], [0, -1, 0])  # +x right (looking along +x)


def test_intrinsics_match_drake_and_palm_is_visible():
    cfg = HeadCameraConfig().scaled(0.5)
    scene = Scene(camera=cfg)
    scene.set_ours(Q_READY)
    scene.set_partner(Q_READY)
    info = scene.camera.default_depth_render_camera().core().intrinsics()
    fx, fy, cx, cy = cfg.intrinsics()
    assert np.allclose([info.focal_x(), info.focal_y(), info.center_x(), info.center_y()],
                       [fx, fy, cx, cy])
    assert (info.width(), info.height()) == (cfg.width, cfg.height)

    depth = scene.render_depth()
    label = scene.render_label()
    assert depth.shape == (cfg.height, cfg.width)
    mask = label == scene.palm_label(PARTNER)
    assert mask.sum() > 50  # partner palm face is in view and segmented
    p, _ = scene.palm(PARTNER)
    expected = np.linalg.norm(p - np.array(cfg.position))
    assert abs(np.median(depth[mask]) - expected) < 0.15
    assert (label == scene.palm_label(OURS)).sum() > 0


def test_frame_buffer_rate_and_latency():
    fb = FrameBuffer(fps=30.0, latency_frames=1)
    captured = []
    for t in np.arange(0.0, 0.2, 0.01):
        if fb.due(t):
            fb.push(t, t)
            captured.append(t)
        got = fb.latest(t)
        if got is not None:
            t_cap, frame = got
            assert t - t_cap >= 1 / 30 - 1e-9  # at least one frame of delay
    assert np.allclose(np.diff(captured), np.round(np.diff(captured) * 30) / 30, atol=0.011)
    assert 5 <= len(captured) <= 7  # ~30 Hz over 0.2 s
