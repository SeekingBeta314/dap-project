"""Realistic noise for rendered depth images (D435-class active stereo).

Drake renders perfect depth. `DepthNoiseModel` corrupts it with the noise
sources a real RealSense D435 shows at 0.2-3 m. Each step and its parameter
values are cited in docs/setup.md (section "Depth noise"); in short:

  1. Range gate        z < Min-Z or z > max range -> invalid (0)    D400 datasheet
  2. Lateral jitter    N(0, sigma_s^2) px shift + bilinear resample  Handa et al. 2014
  3. Axial noise       Gaussian noise in disparity d = f*B/z          Intel tuning whitepaper
                       (gives RMS_z = z^2 * sigma_sp / (f*B))
  4. Fixed pattern     coarse disparity field, held for the episode   Mahler et al. 2017 (Dex-Net 2.0)
  5. Scale bias        depth *= alpha, alpha ~ N(1, sigma_a^2)/episode D400 datasheet (+-2% Z-accuracy)
  6. Grazing dropout   invalid where surface angle > theta_max       Nguyen et al. 2012
  7. Stereo shadow     invalid band beside foreground edges           D400 datasheet sec. 4.5
  8. Holes             Bernoulli pixels + rectangular blobs           D400 fill rate; Zhuang et al. 2023
  9. Quantize          1 mm depth units                               D400 datasheet

Latency is not handled here (see `dap.camera.FrameBuffer`).

Convention: invalid depth is 0, as on a real D400 (Drake uses 0 for too
close and +inf for too far; `apply` maps both to 0).

Pixel units: every pixel-valued parameter (disparity noise, jitter, pattern
grid, blob size) is in NATIVE sensor pixels (848 px wide). When rendering at
a lower resolution, pass `pixel_scale = width / 848` (or use `for_camera`)
so the noise has the same metric and angular size at any resolution.
"""

from dataclasses import dataclass, replace

import numpy as np

# D435 stereo baseline (m). D400 datasheet, Table 4-1.
D435_BASELINE = 0.050
# Minimum depth at 848x480 (m). D400 datasheet, Table 4-12.
D435_MIN_Z_848 = 0.195
# Native depth width (px) that the pixel-valued parameters refer to.
NATIVE_WIDTH = 848


@dataclass(frozen=True)
class DepthNoiseParams:
    min_z: float = D435_MIN_Z_848  # m
    max_z: float = 3.0  # m
    baseline: float = D435_BASELINE  # m
    sigma_subpixel: float = 0.08  # px, disparity noise std (Intel: typical 0.08)
    sigma_shift: float = 0.5  # px, lateral jitter std (Handa et al. 2014)
    sigma_pattern: float = 0.04  # px, fixed-pattern disparity std (0.5 * sigma_subpixel)
    pattern_downsample: int = 8  # coarse grid factor for the fixed pattern
    sigma_scale: float = 0.005  # per-episode multiplicative depth bias std
    theta_max_deg: float = 75.0  # grazing-angle dropout threshold; None disables
    shadow: bool = True  # stereo occlusion band beside foreground edges
    shadow_min_jump: float = 0.05  # m, depth step that counts as an edge
    hole_prob: float = 0.01  # per-pixel dropout probability
    n_blobs: int = 2  # rectangular holes per frame
    blob_size_px: tuple = (4, 16)  # min/max side length of a blob
    quantum: float = 0.001  # m, output depth unit


# Ablation presets (docs/setup.md, "Depth noise" table).
NOISE_PRESETS = {
    "none": None,
    "low": DepthNoiseParams(
        sigma_subpixel=0.05, sigma_shift=0.25, sigma_pattern=0.0, sigma_scale=0.0,
        theta_max_deg=None, hole_prob=0.005, n_blobs=0,
    ),
    "med": DepthNoiseParams(),
    "high": DepthNoiseParams(
        sigma_subpixel=0.2, sigma_shift=1.0, sigma_pattern=0.2, sigma_scale=0.02,
        theta_max_deg=65.0, hole_prob=0.05, n_blobs=5,
    ),
}


def axial_rms(z, fx, sigma_subpixel=0.08, baseline=D435_BASELINE):
    """Intel's D400 depth RMS error (m) at depth z (m) for focal length fx (px)."""
    return np.asarray(z) ** 2 * sigma_subpixel / (fx * baseline)


def _bilinear_upsample(coarse, shape):
    """Upsample a 2D grid to `shape` by bilinear interpolation (numpy only)."""
    h, w = shape
    ch, cw = coarse.shape
    ys = np.linspace(0, ch - 1, h)
    xs = np.linspace(0, cw - 1, w)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, ch - 1)
    x1 = np.minimum(x0 + 1, cw - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = coarse[y0][:, x0] * (1 - wx) + coarse[y0][:, x1] * wx
    bot = coarse[y1][:, x0] * (1 - wx) + coarse[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def _remap_bilinear(z, dx, dy):
    """Sample z at (u + dx, v + dy), interpolating over valid neighbours
    only. A sample is invalid (0) when valid neighbours carry less than half
    the interpolation weight, so holes neither blend into fake depths nor
    grow by a pixel."""
    h, w = z.shape
    v, u = np.mgrid[0:h, 0:w]
    us = np.clip(u + dx, 0, w - 1)
    vs = np.clip(v + dy, 0, h - 1)
    u0 = np.floor(us).astype(int)
    v0 = np.floor(vs).astype(int)
    u1 = np.minimum(u0 + 1, w - 1)
    v1 = np.minimum(v0 + 1, h - 1)
    a = us - u0
    b = vs - v0
    zs = (z[v0, u0], z[v0, u1], z[v1, u0], z[v1, u1])
    ws = ((1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b)
    num = np.zeros_like(a)
    den = np.zeros_like(a)
    for zk, wk in zip(zs, ws):
        ok = zk > 0
        num += np.where(ok, zk * wk, 0.0)
        den += np.where(ok, wk, 0.0)
    return np.where(den >= 0.5, num / np.maximum(den, 1e-12), 0.0)


def depth_edges(z, min_jump):
    """Pixels on a depth discontinuity (or next to an invalid pixel): any
    4-neighbour differs by more than `min_jump` m. Surface angles from
    finite differences are meaningless there (silhouettes, not slopes)."""
    edge = np.zeros(z.shape, dtype=bool)
    for axis in (0, 1):
        d = np.abs(np.diff(z, axis=axis))
        bad = (d > min_jump) | (np.diff(z > 0, axis=axis) != 0)
        if axis == 0:
            edge[:-1] |= bad
            edge[1:] |= bad
        else:
            edge[:, :-1] |= bad
            edge[:, 1:] |= bad
    return edge


def surface_angle(z, fx, fy, cx, cy):
    """Angle (rad) between each pixel's surface normal and its viewing ray,
    from the depth image itself. NaN where undefined."""
    h, w = z.shape
    v, u = np.mgrid[0:h, 0:w]
    P = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    n = np.cross(du, dv)
    nn = np.linalg.norm(n, axis=-1)
    pn = np.linalg.norm(P, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.abs(np.einsum("ijk,ijk->ij", n, P)) / (nn * pn)
    ang = np.arccos(np.clip(c, 0.0, 1.0))
    ang[(z <= 0) | (nn == 0)] = np.nan
    return ang


def stereo_shadow_mask(z, fx, baseline, min_jump):
    """Pixels the second (right) imager cannot see. The left imager is the
    D400 depth reference, so the occluded band lies to the LEFT of each
    foreground edge, with width f*B*(1/z_near - 1/z_far) px (D400 datasheet
    sec. 4.5 gives the same f*B/z geometry for the left-edge invalid band)."""
    h, w = z.shape
    mask = np.zeros_like(z, dtype=bool)
    # Edge where depth drops (background on the left, foreground on the right).
    zl, zr = z[:, :-1], z[:, 1:]
    edge = (zl > 0) & (zr > 0) & (zl - zr > min_jump)
    rows, cols = np.nonzero(edge)
    for r, c in zip(rows, cols):
        width = int(round(fx * baseline * (1.0 / z[r, c + 1] - 1.0 / z[r, c])))
        if width > 0:
            mask[r, max(0, c + 1 - width):c + 1] = True
    return mask


class DepthNoiseModel:
    """Call `reset(rng, shape)` once per episode, then `apply(depth)` per frame."""

    def __init__(self, params: DepthNoiseParams, fx, fy, cx, cy, pixel_scale=1.0):
        # Convert native-pixel parameters to this image's pixels.
        k = pixel_scale
        lo, hi = params.blob_size_px
        self.p = replace(
            params,
            sigma_subpixel=params.sigma_subpixel * k,
            sigma_shift=params.sigma_shift * k,
            sigma_pattern=params.sigma_pattern * k,
            pattern_downsample=max(1, int(round(params.pattern_downsample * k))),
            blob_size_px=(max(1, int(round(lo * k))), max(1, int(round(hi * k)))),
        )
        self.native = params
        self.pixel_scale = k
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.rng = None
        self.pattern = None  # fixed-pattern disparity field (px), per episode
        self.scale = 1.0  # depth scale bias, per episode

    @classmethod
    def from_preset(cls, name, fx, fy, cx, cy, pixel_scale=1.0, **overrides):
        params = NOISE_PRESETS[name]
        if params is None:
            return None
        return cls(replace(params, **overrides), fx, fy, cx, cy, pixel_scale)

    @classmethod
    def for_camera(cls, camera_cfg, preset="med", **overrides):
        """Noise model for a `dap.camera.HeadCameraConfig` at its resolution."""
        return cls.from_preset(preset, *camera_cfg.intrinsics(),
                               pixel_scale=camera_cfg.width / NATIVE_WIDTH, **overrides)

    def reset(self, rng, shape):
        p = self.p
        self.rng = rng
        h, w = shape
        if p.sigma_pattern > 0:
            ch = max(2, h // p.pattern_downsample)
            cw = max(2, w // p.pattern_downsample)
            self.pattern = _bilinear_upsample(rng.normal(0, p.sigma_pattern, (ch, cw)), shape)
        else:
            self.pattern = np.zeros(shape)
        self.scale = 1.0 + (rng.normal(0, p.sigma_scale) if p.sigma_scale > 0 else 0.0)

    def apply(self, depth):
        """`depth`: (H, W) or (H, W, 1) float, metres. Returns float32 (H, W)."""
        if self.rng is None:
            raise RuntimeError("call reset(rng, shape) before apply()")
        p, rng = self.p, self.rng
        z = np.array(depth, dtype=np.float64).reshape(depth.shape[0], depth.shape[1])
        z[~np.isfinite(z)] = 0.0

        # 1. Range gate.
        z[(z < p.min_z) | (z > p.max_z)] = 0.0
        clean = z.copy()

        # 2. Lateral jitter (mixed pixels at edges).
        if p.sigma_shift > 0:
            z = _remap_bilinear(z, rng.normal(0, p.sigma_shift, z.shape),
                                rng.normal(0, p.sigma_shift, z.shape))

        # 3-4. Axial noise in disparity, plus the fixed pattern.
        valid = z > 0
        f_b = self.fx * p.baseline
        d = np.where(valid, f_b / np.where(valid, z, 1.0), 0.0)
        d = d + rng.normal(0, p.sigma_subpixel, z.shape) + self.pattern
        ok = valid & (d > 0)
        z = np.where(ok, f_b / np.where(ok, d, 1.0), 0.0)

        # 5. Per-episode scale bias.
        z *= self.scale

        # 6. Grazing-angle dropout (geometry from the clean image).
        drop = np.zeros(z.shape, dtype=bool)
        if p.theta_max_deg is not None:
            ang = surface_angle(clean, self.fx, self.fy, self.cx, self.cy)
            grazing = np.nan_to_num(ang, nan=0.0) > np.deg2rad(p.theta_max_deg)
            drop |= grazing & ~depth_edges(clean, p.shadow_min_jump)

        # 7. Stereo occlusion shadow.
        if p.shadow:
            drop |= stereo_shadow_mask(clean, self.fx, p.baseline, p.shadow_min_jump)

        # 8. Random holes.
        if p.hole_prob > 0:
            drop |= rng.random(z.shape) < p.hole_prob
        h, w = z.shape
        lo, hi = p.blob_size_px
        for _ in range(p.n_blobs):
            bh, bw = rng.integers(lo, hi + 1, size=2)
            r, c = rng.integers(0, max(1, h - bh)), rng.integers(0, max(1, w - bw))
            drop[r:r + bh, c:c + bw] = True
        z[drop] = 0.0

        # Noise can push depth outside the sensor range.
        z[(z < p.min_z) | (z > p.max_z)] = 0.0

        # 9. Quantize.
        if p.quantum > 0:
            z = np.round(z / p.quantum) * p.quantum
        return z.astype(np.float32)
