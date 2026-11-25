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
