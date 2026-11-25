"""Depth student with explicit image-space supervision for the moving palm."""
import torch
from torch import nn
from torch.nn import functional as F

from dap.student import DepthStudent


def conv(in_ch, out_ch, stride=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride, 1),
                         nn.GroupNorm(4, out_ch), nn.SiLU())


class SpatialStudent(DepthStudent):
    """Locate the palm, read its neighborhood, and resolve its signed finger axis."""
    def __init__(self):
        super().__init__()
        del self.encoder, self.pose
        self.stem = nn.Sequential(conv(3, 16, 2), conv(16, 16))
        self.mid = nn.Sequential(conv(16, 24, 2), conv(24, 24))
        self.deep = nn.Sequential(conv(24, 32, 2), conv(32, 32))
        self.context = nn.Sequential(nn.Conv2d(32, 24, 3, stride=2, padding=1),
                                     nn.SiLU(), nn.Flatten(), nn.Linear(24*4*7, 48), nn.SiLU())
        self.heat = nn.Sequential(conv(16 + 24 + 32, 24), nn.Conv2d(24, 1, 1))
        self.patch = nn.Sequential(conv(3, 16), conv(16, 24, 2), conv(24, 32, 2),
                                   nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(64 + 48 + 31 + 2, 160), nn.SiLU(),
                                  nn.Linear(160, 96), nn.SiLU(), nn.Linear(96, 11))
        # The archived checkpoint supplies the simulator's actual extrinsics
        # and native-resolution intrinsics, rescaled to the 2x-pooled images.
        self.register_buffer("R_WC", torch.eye(3))
        self.register_buffer("p_WC", torch.zeros(3))
        self.register_buffer("intrinsics", torch.ones(4))
        y, x = torch.meshgrid(torch.arange(30)*2+.5, torch.arange(53)*2+.5, indexing="ij")
        self.register_buffer("grid", torch.stack([x, y], dim=-1).reshape(-1, 2))
        # 20 pooled pixels across, capturing the plate and the wrist behind it.
        y, x = torch.meshgrid(torch.arange(20)-9.5, torch.arange(20)-9.5, indexing="ij")
        self.register_buffer("offset", torch.stack([x, y], dim=-1))

    def configure_camera(self, camera):
        self.R_WC.copy_(camera["R_WC"])
        self.p_WC.copy_(camera["p_WC"])
        self.intrinsics.copy_(camera["intrinsics"])

    def project(self, p):
        xyz = (p - self.p_WC) @ self.R_WC
        return xyz[:, :2] / xyz[:, 2:] * self.intrinsics[:2] + self.intrinsics[2:]

    def _parts(self, depth, proprio):
        image = (depth.float() - 1500.) / 750.
        first = self.stem(image)
        middle = self.mid(first)
        deep = self.deep(middle)
        def up(x):
            return F.interpolate(x, size=first.shape[2:], mode="bilinear", align_corners=False)
        logit = self.heat(torch.cat([first, up(middle), up(deep)], dim=1)).flatten(1)
        uv = logit.softmax(1) @ self.grid
        scale = self.intrinsics.new_tensor([105., 59.])
        xy = (uv[:, None, None, :] + self.offset) * (2. / scale) - 1.
        cropped = F.grid_sample(image, xy, mode="bilinear", padding_mode="border",
                                align_corners=True)
        local = (proprio - self.proprio_mean) / self.proprio_scale
        if self.training:
            local = local * (torch.rand((len(local), 1), device=local.device) >= .35)
        head = self.head(torch.cat([self.patch(cropped), self.context(deep), local,
                                    uv / scale], dim=1))
        # The image location determines a calibrated ray. The head estimates
        # optical depth, with the training-pose mean as its starting value.
        z0 = ((self.partner_mean[:3] - self.p_WC) @ self.R_WC)[2]
        z = z0 + .2 * head[:, :1]
        xy_cam = (uv - self.intrinsics[2:]) / self.intrinsics[:2] * z
        p = torch.cat([xy_cam, z], 1) @ self.R_WC.T + self.p_WC
        v = self.partner_mean[3:6] + self.partner_scale[3:6] * head[:, 1:4]
        normal = F.normalize(self.partner_mean[6:9] + self.partner_scale[6:9] * head[:, 4:7])
        # The sign of world y gives two distinct labels for the two common
        # wrist orientations; learn the unsigned axis and the sign separately.
        raw = torch.stack([head[:, 7], F.softplus(head[:, 8]), head[:, 9]], dim=1)
        axis = F.normalize(raw - (normal * raw).sum(1, keepdim=True) * normal)
        axis = torch.where(axis[:, 1:2] >= 0., axis, -axis)
        gate = head[:, 10]
        finger = torch.where(gate[:, None] >= 0., axis, -axis)
        partner = torch.cat([p, v, normal, finger], dim=1)
        return partner, logit, axis, gate

    def predict(self, depth, proprio):
        partner, _, _, _ = self._parts(depth, proprio)
        return partner, self.action(proprio, partner)

    def losses(self, depth, proprio, truth):
        p, logit, axis, gate = self._parts(depth, proprio)
        # Supervise image location even when our hand hides the partner. The
        # labels are derived only from training-time capture-time truth.
        uv = self.project(truth[:, :3])
        d2 = (self.grid[None] - uv[:, None]).square().sum(2)
        target_heat = (-d2 / (2 * 1.6**2)).softmax(1)
        heat = -(target_heat * logit.log_softmax(1)).sum(1).mean()
        sign = truth[:, 10] >= 0
        true_axis = torch.where(sign[:, None], truth[:, 9:], -truth[:, 9:])
        residual = torch.cat([(p[:, :9] - truth[:, :9]) / self.partner_scale[:9],
                             (axis - true_axis) / .24], 1)
        pose_loss = (F.smooth_l1_loss(residual, torch.zeros_like(residual)) +
                     .15 * heat + .35 * F.binary_cross_entropy_with_logits(gate, sign.float()))
        with torch.no_grad():
            teacher_action = self.action(proprio, truth)
        action_loss = F.mse_loss(self.action(proprio, p), teacher_action)
        return pose_loss, action_loss
