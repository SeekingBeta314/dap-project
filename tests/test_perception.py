"""Label-free palm perception (dap.perception) and the harness's camera path
for the servo_depth baseline."""

import gymnasium as gym
import numpy as np
import pytest

import dap.envs  # noqa: F401  (registers dap/PalmMeet-v0)
from dap.baselines import SERVO_DEPTH_CAMERA, HoldController, servo_depth_factory
from dap.camera import sensed_depth
from dap.envs import ENV_ID
from dap.experiment import Condition, ExperimentConfig, train_rngs
from dap.perception import DepthPalmTracker
from dap.scenario import SCENARIO_PRESETS, sample_scenario
from dap.scene import PARTNER
from dap.trial import run_trial

CAM = SERVO_DEPTH_CAMERA
CFG = ExperimentConfig(name="t")


@pytest.fixture(scope="module")
def scenarios():
    # Training scenarios: perception is checked off the evaluation seeds.
    return [sample_scenario(train_rngs(s)["scenario"], SCENARIO_PRESETS["nominal"], camera=CAM)
            for s in range(3)]


def _clean(scene):
    return sensed_depth(scene.render_depth(), None, CAM.z_far)


def test_segmentation_keeps_only_the_partner(scenarios):
    for sc in scenarios:
        S, tr = sc.scene, DepthPalmTracker(sc.layout, CAM)
        S.set_ours(sc.q_ours0)
        S.set_partner(sc.partner.q(0.5 * sc.partner.t_end))
        P = tr.foreign_points(_clean(S), sc.q_ours0)
        # Project the points back to pixels and read the true labels there.
        Pc = (P - tr.p_WC) @ tr.R_WC
        u = np.round(tr.fx * Pc[:, 0] / Pc[:, 2] + tr.cx).astype(int)
        v = np.round(tr.fy * Pc[:, 1] / Pc[:, 2] + tr.cy).astype(int)
        labels = S.render_label()[v, u]
        plant = S.plant
        partner = {int(b) for mi in (S.models[PARTNER], S.hands[PARTNER])
                   for b in plant.GetBodyIndices(mi)}
        assert len(P) > 50 and set(labels.tolist()) <= partner
        assert S.palm_label(PARTNER) in labels


def test_clean_frames_give_the_palm_pose(scenarios):
    """Our arm still, no depth noise, frames at 30 Hz along the partner's
    reach: the measured palm face center, normal and finger direction."""
    pos, normal, finger, n = [], [], [], 0
    for sc in scenarios:
        S, m, tr = sc.scene, sc.partner, DepthPalmTracker(sc.layout, CAM)
        S.set_ours(sc.q_ours0)
        for t in np.arange(0, m.t_end, 1 / 30):
            n += 1
            S.set_partner(m.q(t))
            z = tr.observe(t, _clean(S), sc.q_ours0)
            if z is None:
                continue
            R = m.rotation(t)
            pos.append(np.linalg.norm(z[0] - m.p(t)))
            normal.append(np.degrees(np.arccos(np.clip(z[1] @ R[:, 2], -1, 1))))
            finger.append(np.degrees(np.arccos(np.clip(z[2] @ R[:, 0], -1, 1))))
    assert len(pos) >= 0.5 * n
    assert np.median(pos) < 0.025 and max(normal) < 5.0
    assert np.mean(np.array(finger) < 30.0) >= 0.9


class _Recorder(HoldController):
    def observe(self, t_capture, depth, q_capture):
        self.frames.append((t_capture, depth, q_capture))


def test_trial_frames_are_the_env_frames():
    """With our arm held still, run_trial delivers the student's frames:
    same capture times, delivery latency and noisy depth (trial seed 2)."""
    rec = _Recorder()
    rec.frames, delivered = [], []
    rec.step = lambda t, q, est: (delivered.append((t, len(rec.frames))), rec.q0.copy())[1]
    run_trial(CFG, Condition("hold", 1.0), 2, camera=CAM, controller_factory=lambda *a: rec)

    env = gym.make(ENV_ID, obs_mode="pixels").unwrapped
    env.reset(options={"trial_seed": 2})
    frames = {t_capture: f[0] for _, t_capture, f in env.frames.pending}  # every capture
    push = env.frames.push
    env.frames.push = lambda t, f: (frames.__setitem__(t, f[0]), push(t, f))[1]
    while not any(env.step(np.zeros(7, np.float32))[2:4]):
        pass
    assert len(rec.frames) > 10
    for t_capture, depth, q in rec.frames:
        assert np.array_equal(q, rec.q0)
        assert np.array_equal(depth, frames[t_capture])
    # Each frame reaches the controller one frame period after capture (at
    # the first 10 ms step at or after it).
    first_seen = {}
    for t, k in delivered:
        for t_capture, *_ in rec.frames[:k]:
            first_seen.setdefault(t_capture, t)
    for t_capture, t in first_seen.items():
        assert 0 <= t - (t_capture + 1 / CAM.fps) < CFG.dt + 1e-9


def test_servo_depth_needs_the_camera():
    r = run_trial(CFG, Condition("servo_depth", 1.0), 0, controller_factory=servo_depth_factory)
    assert r.outcome == "error" and "head camera" in r.error
    r = run_trial(CFG, Condition("servo_depth", 1.0), 0, camera=CAM,
                  controller_factory=servo_depth_factory)
    assert r.error == "" and r.outcome in ("contact", "collision", "timeout")
