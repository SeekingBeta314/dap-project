"""Render the cupped-palm dap at real-time speed as an MP4 and a looping GIF.

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


def render_frame(scene, t, close_cam, profile_cam, wide_cam):
    scene.set_time(t)
    query = scene.query_object()
    # Leave space for the striking hand's longer backswing on the left.
    target = MEETING + [-0.036, -0.016, -0.031]
    direction = np.array([0.48, -1.0, -0.24])
    eye = target + 0.74 * direction / np.linalg.norm(direction)
    close = render_view(query, close_cam, eye, target)

    # The reverse three-quarter view exposes the hollow between the palms.
    # Follow some of the wrist travel so both cups remain in this tighter view.
    center = scene.palm_centers().mean(axis=0)
    target = MEETING + 0.65 * (center - MEETING) + [-0.006, 0.0, 0.014]
    direction = np.array([-0.14, 0.34, 1.0])
    profile = render_view(query, profile_cam, target + 0.72 * direction / np.linalg.norm(direction), target)
    wide = render_view(query, wide_cam, (0.82, -1.94, 1.08), (0.67, 0.0, 0.46))

    split = close.width + 4
    width = split + profile.width
    canvas = Image.new("RGB", (width, close.height + 48), (24, 37, 42))
    canvas.paste(close)
    canvas.paste(profile, (split, 0))
    ink = ImageDraw.Draw(canvas)
    ink.rectangle((7, 7, wide.width + 10, wide.height + 10), fill=(245, 243, 236))
    canvas.paste(wide, (9, 9))
    height = close.height
    ink.text((20, height + 12), scene.phase(t), font=ImageFont.load_default(size=21), fill="white")
    ink.text((split + 16, height + 12), "Palm profile", font=ImageFont.load_default(size=19),
             fill=(194, 213, 216))
    ink.text((width - 20, height + 12), f"{t:.2f} s", anchor="ra",
             font=ImageFont.load_default(size=19), fill=(194, 213, 216))
    ink.rectangle((0, height, round(width * t / scene.duration), height + 2), fill=(106, 211, 193))
    return canvas


def export(out, fps):
    ffmpeg = get_ffmpeg_exe()
    scene = HandshakeScene(renderer=studio_lights())
    shade_hands(scene)
    close_cam = camera(640, 492, 31)
    profile_cam = camera(508, 492, 30)
    wide_cam = camera(172, 97, 35)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as temp:
        mp4, gif = (Path(temp) / f"hand_dap.{ext}" for ext in ("mp4", "gif"))
        command = [ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", "1152x540", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
                   "-pix_fmt", "yuv420p", "-crf", "17", "-movflags", "+faststart", str(mp4)]
        # Anchor the instant of palm contact to a frame retained by both encoders.
        # Shift playback by at most one GIF frame; the motion itself runs in real time.
        gif_stride = max(1, math.ceil(fps / 30))
        impact_frame = math.ceil(scene.impact_time * fps / gif_stride) * gif_stride
        lead_in = impact_frame / fps - scene.impact_time
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            try:
                for index in range(math.ceil((scene.duration + lead_in) * fps) + 1):
                    t = float(np.clip(index / fps - lead_in, 0, scene.duration))
                    frame = render_frame(scene, t, close_cam, profile_cam, wide_cam)
                    encoder.stdin.write(frame.tobytes())
            finally:
                encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("FFmpeg failed to encode the hand dap.")
        filters = (f"select='not(mod(n,{gif_stride}))',setpts=N*{gif_stride}/({fps}*TB),"
                   "scale=896:-1:flags=lanczos,split[a][b];"
                   "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3")
        subprocess.run([ffmpeg, "-v", "error", "-i", str(mp4), "-filter_complex", filters,
                        "-r", f"{fps}/{gif_stride}", "-loop", "0", str(gif)], check=True)
        for artifact in (mp4, gif):
            artifact.replace(out / artifact.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs"), help="output directory")
    parser.add_argument("--fps", type=int, default=50, help="video frames per second")
    args = parser.parse_args()
    if args.fps < 1:
        parser.error("--fps must be positive")
    export(args.out, args.fps)
