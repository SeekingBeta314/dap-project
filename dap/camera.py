"""Fixed head camera behind/above our arm, modelled on an Intel RealSense D435.

Parameter sources (full table in docs/setup.md, "Head camera"):
  - 848x480 depth, vertical FOV 58 deg (-> ~89 deg horizontal): D435 depth
    FOV 87x58 deg (D400 datasheet Table 3-49) and Intel's recommended
    848x480 depth mode (Intel tuning whitepaper).
  - 30 fps: D435 / Azure Kinect / DexYCB rate.
  - depth range 0.195-3.0 m: D435 Min-Z at 848x480; ideal range 0.3-3 m.
  - mount ~0.4 m behind and ~1.45 m above our base, pitched ~38 deg down:
    RoboCasa puts its cameras 0.5-0.6 m behind the base, 0.95-1.15 m up,
    pitched 26-35 deg down; Fetch's head camera is 1.06-1.45 m above its
    base link (torso down/up).
  - latency: one frame (33 ms) by default, modelled by `FrameBuffer`.

The camera is welded to the world, i.e. fixed relative to our base (our base
frame is the world frame).
"""

from collections import deque
from dataclasses import dataclass, replace

import numpy as np
from pydrake.all import (
    CameraConfig,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    RgbdSensor,
    RigidTransform,
    RotationMatrix,
)

RENDERER = "dap_vtk"


@dataclass(frozen=True)
class HeadCameraConfig:
    name: str = "head_camera"
    width: int = 848
    height: int = 480
    fov_y_deg: float = 58.0
    fps: float = 30.0
    z_near: float = 0.195  # m, depth sensing range (D435 Min-Z at 848x480)
    z_far: float = 3.0  # m
    clip_near: float = 0.01  # m, renderer clipping planes
    clip_far: float = 10.0
    position: tuple = (-0.4, 0.0, 1.45)  # m, in our base (= world) frame
    target: tuple = (0.7, 0.0, 0.6)  # m, point the optical axis passes through
    latency_frames: int = 1
    rgb: bool = False  # color is only needed for rendering / figures
    label: bool = True  # ground-truth segmentation, used only by tests

    def scaled(self, factor):
        """Same camera at a lower resolution (e.g. 0.25 -> 212x120 for RL)."""
        return replace(self, width=int(round(self.width * factor)),
                       height=int(round(self.height * factor)))

    def intrinsics(self):
        """(fx, fy, cx, cy) in pixels, matching Drake's CameraInfo for a
        fov_y camera (square pixels, principal point at the image center)."""
        fy = self.height / 2 / np.tan(np.deg2rad(self.fov_y_deg) / 2)
        return fy, fy, self.width / 2 - 0.5, self.height / 2 - 0.5

    def X_WC(self):
        return look_at(self.position, self.target)


def look_at(position, target, up=(0.0, 0.0, 1.0)):
    """Camera pose with Drake's convention: +z forward (optical axis),
    +x right, +y down in the image."""
    p = np.asarray(position, float)
    z = np.asarray(target, float) - p
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return RigidTransform(RotationMatrix(np.column_stack([x, y, z])), p)


def _drake_config(cfg: HeadCameraConfig):
    c = CameraConfig()
    c.name = cfg.name
    c.width, c.height = cfg.width, cfg.height
    c.focal = CameraConfig.FovDegrees(y=cfg.fov_y_deg)
    c.fps = cfg.fps
    c.z_near, c.z_far = cfg.z_near, cfg.z_far
    c.clipping_near, c.clipping_far = cfg.clip_near, cfg.clip_far
    c.rgb, c.depth, c.label = cfg.rgb, True, cfg.label
    c.renderer_name = RENDERER
    return c


def add_head_camera(builder, plant, scene_graph, cfg: HeadCameraConfig):
    """Add a world-fixed RgbdSensor named `cfg.name`. We pull images by
    evaluating its output ports, so no LCM publishing is set up."""
    if not scene_graph.HasRenderer(RENDERER):
        scene_graph.AddRenderer(RENDERER, MakeRenderEngineVtk(RenderEngineVtkParams()))
    color, depth = _drake_config(cfg).MakeCameras()
    world_id = plant.GetBodyFrameIdOrThrow(plant.world_body().index())
    sensor = builder.AddSystem(RgbdSensor(world_id, cfg.X_WC(), color, depth))
    sensor.set_name(cfg.name)
    builder.Connect(scene_graph.get_query_output_port(), sensor.query_object_input_port())
    return sensor


def sensed_depth(depth, noise, z_far):
    """What the head camera reports for a rendered depth image: D435-style
    noise from `noise` (a dap.depth_noise.DepthNoiseModel), or with
    `noise=None` the clean image with out-of-range pixels invalid (0).
    float32 (H, W), metres."""
    if noise is not None:
        return noise.apply(depth)
    return np.nan_to_num(np.where(depth > z_far, 0.0, depth), posinf=0.0).astype(np.float32)


class FrameBuffer:
    """Camera timing: frames are captured every 1/fps s and delivered
    `latency_frames` frame periods later. Call `due(t)` each control step;
    when it returns True, render and `push(t, frame)`. `latest(t)` is the
    newest frame already delivered at time t (None before the first one)."""

    def __init__(self, fps, latency_frames=0):
        self.period = 1.0 / fps
        self.delay = latency_frames * self.period
        self.next_capture = 0.0
        self.pending = deque()
        self.delivered = None

    def due(self, t):
        return t + 1e-9 >= self.next_capture

    def push(self, t, frame):
        self.pending.append((t + self.delay, t, frame))
        self.next_capture += self.period
        while self.next_capture <= t:
            self.next_capture += self.period

    def latest(self, t):
        while self.pending and self.pending[0][0] <= t + 1e-9:
            self.delivered = self.pending.popleft()
        if self.delivered is None:
            return None
        _, t_capture, frame = self.delivered
        return t_capture, frame
