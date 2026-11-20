"""Render both views of the articulated hand dap and save an MP4 and a looping GIF.

    python scripts/show_hand_dap.py --out docs
"""

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image, ImageDraw, ImageFont
from pydrake.all import (
    CameraInfo, ClippingRange, ColorRenderCamera, LightParameter, RenderCameraCore,
    PerceptionProperties, RenderEngineVtkParams, Rgba, RigidTransform, Role, SceneGraph,
    SsaoParameter,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dap.camera import look_at
from dap.handshake import HandshakeScene, MEETING


def studio_lights():
    settings = RenderEngineVtkParams()
    settings.default_clear_color = np.array([0.85, 0.88, 0.89])
    lights = []
    for direction, intensity, rgb in [
        ((0.4, 0.6, -1.0), 1.25, (1.0, 0.97, 0.93)),
        ((-0.6, 0.8, 0.1), 0.93, (0.87, 0.94, 1.0)),
        ((0.1, -1.0, -0.4), 0.75, (1.0, 0.99, 0.98)),
        ((0.2, 0.4, 1.0), 0.66, (1.0, 0.95, 0.91)),
    ]:
        light = LightParameter()
        light.frame = "world"
        light.direction = np.array(direction) / np.linalg.norm(direction)
        light.intensity = intensity
        light.color = Rgba(*rgb, 1)
        lights.append(light)
    settings.lights = lights
    # Local shadows at a scale comparable to a finger joint.
    ao = SsaoParameter()
    ao.radius, ao.bias, ao.sample_count = 0.027, 0.001, 32
    settings.ssao = ao
    return settings


def shade_hands(scene):
    # Each participant gets a distinct skin tone so overlapping digits stay legible.
    tones = {"iiwa": ((.77, .46, .31), (.84, .59, .49)),
             "partner": ((.98, .75, .60), (.99, .81, .76))}
    sg, ctx = scene.scene_graph, scene.sg_context
    source = scene.plant.get_source_id()
    inspector = scene.query_object().inspector()
    for side, model in scene.hands.items():
        skin, nail = tones[side]
        for body in scene.plant.GetBodyIndices(model):
            frame = scene.plant.GetBodyFrameIdOrThrow(body)
            for geometry in inspector.GetGeometries(frame, Role.kPerception):
                name = inspector.GetName(geometry).split("::")[-1]
                if name not in ("skin", "nail"):
                    continue
                props = PerceptionProperties(inspector.GetPerceptionProperties(geometry))
                props.UpdateProperty("phong", "diffuse", Rgba(*(nail if name == "nail" else skin), 1))
                sg.RemoveRole(ctx, source, geometry, Role.kPerception)
                sg.AssignRole(ctx, source, geometry, props)


def camera(width, height, fov_deg):
    info = CameraInfo(width, height, math.radians(fov_deg))
    core = RenderCameraCore("dap_studio", info, ClippingRange(0.01, 8), RigidTransform())
    return ColorRenderCamera(core)


def render_view(query, cam, eye, target):
    view = query.RenderColorImage(cam, SceneGraph.world_frame_id(), look_at(eye, target))
    return Image.fromarray(np.ascontiguousarray(view.data[:, :, :3]))


def render_frame(scene, t, close_cam, wide_cam):
    scene.set_time(t)
    first, second = scene.palm_centers()
    center = (np.asarray(first) + second) * 0.5
    query = scene.query_object()
    # Track part of the reach so the fingers stay framed and the pull stays visible.
    target = MEETING + 0.52 * (center - MEETING)
    # Step back as the hands separate to keep the fingertips in the close view.
    distance = 0.48 + 0.55 * np.linalg.norm(first - second)
    direction = np.array([-0.16, -1., -0.66])
    eye = target + distance * direction / np.linalg.norm(direction)
    close = render_view(query, close_cam, eye, target)
    wide = render_view(query, wide_cam, (0.76, -1.94, 1.10), (0.64, 0, 0.47))

    width, height = close.size
    canvas = Image.new("RGB", (width, height + 48), (24, 37, 42))
    canvas.paste(close)
    inset_x = width - wide.width - 16
    ink = ImageDraw.Draw(canvas)
    ink.rectangle((inset_x - 2, 14, width - 14, wide.height + 18), fill=(245, 243, 236))
    canvas.paste(wide, (inset_x, 16))
    font = ImageFont.load_default(size=21)
    ink.text((20, height + 12), scene.phase(t).replace("_", " ").capitalize(),
             font=font, fill="white")
    ink.text((width - 20, height + 12), f"{t:.2f} s", anchor="ra",
             font=ImageFont.load_default(size=19), fill=(194, 213, 216))
    ink.rectangle((0, height, round(width * t / scene.duration), height + 2),
                  fill=(106, 211, 193))
    return canvas


def export(out, fps):
    ffmpeg = get_ffmpeg_exe()
    scene = HandshakeScene(renderer=studio_lights())
    shade_hands(scene)
    close_cam = camera(960, 492, 28)
    wide_cam = camera(228, 128, 35)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as temp:
        mp4, gif = (Path(temp) / f"hand_dap.{ext}" for ext in ("mp4", "gif"))
        command = [ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", "960x540", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                   "-pix_fmt", "yuv420p", "-crf", "17", "-movflags", "+faststart", str(mp4)]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            try:
                for index in range(math.ceil(scene.duration * fps) + 1):
                    frame = render_frame(scene, min(index / fps, scene.duration), close_cam, wide_cam)
                    encoder.stdin.write(frame.tobytes())
            finally:
                encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("FFmpeg failed to encode the hand dap.")
        filters = (f"fps={min(20, fps)},scale=720:-1:flags=lanczos,split[a][b];"
                   "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3")
        subprocess.run([ffmpeg, "-v", "error", "-i", str(mp4), "-filter_complex", filters,
                        "-loop", "0", str(gif)], check=True)
        for artifact in (mp4, gif):
            artifact.replace(out / artifact.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs"), help="output directory")
    parser.add_argument("--fps", type=int, default=30, help="video frames per second")
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("--fps must be positive")
    export(args.out, args.fps)
