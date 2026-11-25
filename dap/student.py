"""A depth student that estimates the delayed partner pose and drives a frozen teacher.

The teacher is an MLP distilled from the privileged servo. During deployment its
partner inputs come entirely from the depth network. Privileged labels are used
only by the offline training script. The checkpoint includes both networks.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def pack_depth(frames):
    """Nearest return in each 2x2 block of the whole image, in millimetres.

    Zero (no return) becomes 3 m, the camera's far plane. Returns are kept
    to 1 mm so that the three-frame training set fits in memory.
    """
    frames = np.asarray(frames, np.float32)
    if frames.shape != (3, 120, 212):
        raise ValueError(f"expected three 120 x 212 frames, received {frames.shape}")
    valid = np.where(frames > 0, frames, 3.).reshape(3, 60, 2, 106, 2)
    return np.rint(valid.min(axis=(2, 4)).clip(0, 3) * 1000).astype(np.int16)


def actor_features(proprio, partner):
    """Torch equivalent of dap.teacher.teacher_features, with gradients to partner."""
    def frame(x, z):
        return torch.stack([x, torch.linalg.cross(z, x), z], dim=-1)

    ours = frame(proprio[:, 20:23], proprio[:, 17:20])
    theirs = frame(partner[:, 9:12], partner[:, 6:9])
    goal = theirs * theirs.new_tensor([-1., 1., -1.])
    rotation_error = goal @ ours.transpose(1, 2)
    return torch.cat([proprio, partner, partner[:, :3] - proprio[:, 14:17],
                      rotation_error.flatten(1)], dim=1)


class DepthStudent(nn.Module):
    """Regress the 12 teacher inputs (position, velocity, normal, finger axis).

    Spatial features retain their locations in the image. Proprioception lets
    the network distinguish the robot's own palm during self-occlusion.
    """
    obs_mode = "pixels"

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 40, 3, stride=2, padding=1), nn.SiLU(),
            nn.Flatten())
        self.pose = nn.Sequential(nn.Linear(40 * 4 * 7 + 31, 192), nn.SiLU(),
                                  nn.Linear(192, 128), nn.SiLU(), nn.Linear(128, 12))
        self.actor = nn.Sequential(nn.Linear(55, 256), nn.Tanh(), nn.Linear(256, 256),
                                   nn.Tanh(), nn.Linear(256, 7))
        self.actor.requires_grad_(False)
        for name, width in [("proprio", 31), ("partner", 12), ("actor", 55)]:
            self.register_buffer(name + "_mean", torch.zeros(width))
            self.register_buffer(name + "_scale", torch.ones(width))

    def predict(self, depth, proprio):
        z = self.encoder((depth.float() - 1500.) / 750.)
        local = (proprio - self.proprio_mean) / self.proprio_scale
        if self.training:
            # On demonstrations our own wrist tracks the partner. Occasionally
            # hiding this input discourages a shortcut through those correlations.
            local = local * (torch.rand((len(local), 1), device=local.device) >= .35)
        raw = self.pose(torch.cat([z, local], 1))
        p = raw * self.partner_scale + self.partner_mean
        # Orthogonal axes define a rotation even during imperfect estimation.
        n = F.normalize(p[:, 6:9], dim=1)
        x = p[:, 9:12] - (p[:, 9:12] * n).sum(1, keepdim=True) * n
        partner = torch.cat([p[:, :6], n, F.normalize(x, dim=1)], dim=1)
        return partner, self.action(proprio, partner)

    def action(self, proprio, partner):
        x = (actor_features(proprio, partner) - self.actor_mean) / self.actor_scale
        return self.actor(x.clamp(-10., 10.)).clamp(-1., 1.)

    @torch.inference_mode()
    def forward(self, obs):
        if not np.any(obs["depth"][-1]):
            return np.zeros(7, np.float32)
        device = self.proprio_mean.device
        depth = torch.as_tensor(pack_depth(obs["depth"])[None], device=device)
        proprio = torch.as_tensor(obs["proprio"][None], device=device)
        return self.predict(depth, proprio)[1][0].cpu().numpy()


def load_student(path, *, allow_pose_only=False):
    """Load a single versioned checkpoint. No teacher files are needed at deployment."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    if ckpt["format"] != 1:
        raise ValueError("unrecognized depth student checkpoint")
    if not ckpt["teacher"] and not allow_pose_only:
        raise ValueError("a pose-only initialization has no fitted actor")
    model = DepthStudent()
    model.load_state_dict(ckpt["weights"])
    return model.eval()


def student_factory(cfg, cond):
    root = Path(__file__).resolve().parents[1]
    return load_student(root / "models" / "policies" / "student.pt")
