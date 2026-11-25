"""Guard the boundary between supervised privileged labels and policy observations."""
import numpy as np
import pytest
import torch

from dap.student import DepthStudent, actor_features, load_student, pack_depth
from dap.teacher import teacher_features


def test_teacher_features_match_for_distillation():
    state = np.random.default_rng(7).uniform(-1, 1, (12, 43)).astype("float32")
    t = torch.from_numpy(state)
    actual = actor_features(t[:, :31], t[:, 31:]).numpy()
    np.testing.assert_allclose(actual, teacher_features(state), atol=3e-7)


def test_depth_packing_preserves_nearest_valid_return_and_history():
    image = np.zeros((3, 120, 212), np.float32)
    image[0, 0, 1] = 1.234
    image[0, 1, 0] = 2.0
    packed = pack_depth(image)
    assert packed.shape == (3, 60, 106)
    assert packed[:, 0, 0].tolist() == [1234, 3000, 3000]


def test_deployed_student_only_needs_pixels_and_proprio(tmp_path):
    torch.manual_seed(5)
    model = DepthStudent().eval()
    path = tmp_path / "model.pt"
    torch.save({"format": 1, "teacher": {"policy": "teacher"},
                "weights": model.state_dict()}, path)
    policy = load_student(path)
    obs = {"depth": np.ones((3, 120, 212), np.float32),
           "proprio": np.zeros(31, np.float32)}
    np.testing.assert_allclose(policy(obs), model(obs), atol=1e-6)
    assert policy.obs_mode == "pixels" and policy(obs).shape == (7,)
    obs["depth"].fill(0)
    np.testing.assert_array_equal(policy(obs), np.zeros(7))
    torch.save({"format": 1, "teacher": None, "weights": model.state_dict()}, path)
    with pytest.raises(ValueError, match="pose-only"):
        load_student(path)


def test_spatial_student_projects_calibrated_depth_and_loads(tmp_path):
    from dap.student_spatial import SpatialStudent
    model = SpatialStudent().eval()
    model.configure_camera({"R_WC": torch.eye(3), "p_WC": torch.zeros(3),
                            "intrinsics": torch.tensor([40., 40., 52.5, 29.5])})
    point = torch.tensor([[.10, -.20, 2.]])
    torch.testing.assert_close(model.project(point), torch.tensor([[54.5, 25.5]]))
    with torch.no_grad():
        model.partner_mean.copy_(torch.tensor([0., 0., 2., 0., 0., 0., -1., 0., 0., 0., 1., 0.]))
    path = tmp_path / "spatial.pt"
    torch.save({"format": 1, "architecture": "spatial", "weights": model.state_dict(),
                "teacher": {"policy": "teacher"}}, path)
    policy = load_student(path)
    with torch.no_grad():
        p, a = policy.predict(torch.ones(2, 3, 60, 106), torch.zeros(2, 31))
    assert a.shape == (2, 7) and torch.isfinite(a).all()
    torch.testing.assert_close(torch.cross(p[:, 6:9], p[:, 9:], dim=1).norm(dim=1),
                               torch.ones(2))
