import numpy as np

from dap.meeting import target_palm_pose
from dap.planner import ReplanningController
from dap.scene import P_PALM_FACE
from dap.experiment import CONFIGS, Condition
from dap.trial import run_trial


def test_face_target_uses_face_offset_and_is_orthonormal():
    # Include a normal parallel to the roll prior (projection degeneracy).
    for n in [np.array([0., 0., 1.]), np.array([-1., 0., 0.])]:
        p = np.array([0.61, 0.13, 0.56])
        X = target_palm_pose(p, n, standoff=0.04)
        R = X.rotation().matrix()
        np.testing.assert_allclose(X @ P_PALM_FACE, p + 0.04 * n)
        np.testing.assert_allclose(R[:, 2], -n)
        np.testing.assert_allclose(R.T @ R, np.eye(3))
        assert np.linalg.det(R) > 0.999


def test_controllers_move_and_share_the_initial_plan():
    # Development seed, outside the benchmark cohort. Both controllers
    # receive an identical observation at their first planning instant.
    logs = []
    for mode in ["once", "replan"]:
        result, log = run_trial(CONFIGS["smoke"], Condition(mode, 1), 499900,
                                controller_factory=lambda *_: ReplanningController(replan=mode == "replan"),
                                return_log=True)
        assert not result.error and result.n_bad_commands == 0
        q = np.array(log.q)
        assert np.linalg.norm(q[-1] - q[0]) > 0.1
        logs.append(q)
    np.testing.assert_allclose(logs[0][:10], logs[1][:10], atol=1e-7)


def test_spline_respects_its_boundary_conditions_and_joint_limits():
    from dap.meeting import MeetingPoint
    from dap.planner import controller_model, plan_to_meeting
    from dap.scene import OURS, PARTNER, Scene

    scene = Scene()
    model = controller_model(scene, 0.01)
    start = scene.arm_q(OURS)
    goal = start.copy()
    goal[0] += 0.06
    scene.set_ours(goal)
    meeting = MeetingPoint(1.0, scene.palm_pose(), np.zeros(3), goal, 0.,
                           scene.arm_q(PARTNER))
    path = plan_to_meeting(model, start, np.zeros(7), meeting, 0.0)
    assert path is not None
    np.testing.assert_allclose(path.value(0).ravel()[model.idx], start, atol=1e-5)
    np.testing.assert_allclose(path.value(1).ravel()[model.idx], goal, atol=1e-5)
    for t in np.linspace(0, 1, 51):
        q = path.value(t).ravel()[model.idx]
        v = path.EvalDerivative(t, 1).ravel()[model.idx]
        assert np.all(q >= model.q_lower - 1e-5)
        assert np.all(q <= model.q_upper + 1e-5)
        assert np.all(abs(v) <= model.v_max + 1e-5)
    for t in (0, 1):
        np.testing.assert_allclose(path.EvalDerivative(t, 1), 0., atol=1e-5)
