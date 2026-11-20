"""Depth-conditioned imitation policy. Runtime only requires NumPy.

The student regresses the expert's terminal joint position. A proportional
joint servo converts its estimate into velocity actions. The initial joint
position is retained for the redundant arm's choice of IK solution.
"""
from pathlib import Path

import numpy as np


CHECKPOINTS = Path(__file__).resolve().parents[1] / "models" / "policies"


def pool_depth(frame):
    """Closest valid return in each 4x4 patch of the 120x212 depth frame."""
    frame = np.asarray(frame, np.float32)
    if frame.shape != (120, 212):
        raise ValueError(f"expected a 120x212 head camera frame, received {frame.shape}")
    valid = np.where(frame > 0, frame, np.float32(3.0))
    return valid.reshape(30, 4, 53, 4).min(axis=(1, 3))


def fine_depth(frame):
    """Half-resolution central view, preserving the narrow finger plate."""
    frame = np.asarray(frame, np.float32)
    if frame.shape != (120, 212):
        raise ValueError(f"expected a 120x212 frame, received {frame.shape}")
    valid = np.where(frame[0:112, 38:174] > 0, frame[0:112, 38:174], np.float32(3.0))
    return valid.reshape(56, 2, 68, 2).min(axis=(1, 3))


def conv_features(image, params):
    """Evaluate the three stride-two PyTorch cross-correlations with NumPy."""
    x = ((image - 1.5) / .75)[None]
    for i in range(3):
        weight = params[f"kernel{i}"]
        k = weight.shape[-1]
        x = np.pad(x, ((0, 0), (k // 2, k // 2), (k // 2, k // 2)))
        windows = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(1, 2))
        patches = windows[:, ::2, ::2].transpose(1, 2, 0, 3, 4)
        height, width = patches.shape[:2]
        conv = patches.reshape(height * width, -1) @ weight.reshape(weight.shape[0], -1).T
        x = np.maximum(conv + params[f"offset{i}"], 0).T.reshape(-1, height, width)
    return x.ravel()


class PixelPolicy:
    obs_mode = "pixels"

    def __init__(self, checkpoint):
        with np.load(checkpoint, allow_pickle=False) as data:
            self.params = dict(data)
        self.reset()

    def reset(self):
        self.q0 = None
        self.branch_logit = None

    def __call__(self, obs):
        p = self.params
        q = obs["proprio"][:7]
        if self.q0 is None:
            self.q0 = q.copy()
        if p["vision"].item() and not obs["depth"][-1].any():
            return np.zeros(7, dtype=np.float32)
        features = [self.q0, q, obs["proprio"][23:24]]
        if p["vision"].item() and "kernel0" not in p:
            features.append(pool_depth(obs["depth"][-1]).ravel())
        x = (np.concatenate(features) - p["x_mean"]) / p["x_scale"]
        if "kernel0" in p:
            x = np.r_[x, conv_features(fine_depth(obs["depth"][-1]), p)]
        for i in range(3):
            x = x @ p[f"weight{i}"] + p[f"bias{i}"]
            if i < 2:
                x = np.maximum(x, 0.)
        if x.size == 15:
            # Keep some evidence from earlier frames when our own hand hides
            # the partner near contact. The first prediction starts the filter.
            self.branch_logit = (x[14] if self.branch_logit is None
                                 else .25 * x[14] + .75 * self.branch_logit)
            x = x[7:14] if self.branch_logit > 0 else x[:7]
        goal = x * p["y_scale"] + p["y_mean"]
        return np.clip((goal - q) / (0.22 * p["v_max"]), -1., 1.).astype(np.float32)


def pixel_factory(cfg, cond):
    return PixelPolicy(CHECKPOINTS / "depth.npz")


def proprio_factory(cfg, cond):
    return PixelPolicy(CHECKPOINTS / "proprio.npz")
