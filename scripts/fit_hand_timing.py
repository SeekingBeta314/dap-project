"""Fit the normalized inward swing to OMFGrant's Contemporary Dap Up (CC0).

Run without arguments to refit the published annotations. Supply the original
GIF and --overlay to inspect the hand-marked points on the six source frames.
"""

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dap.hand_timing import dap_swing_bow, dap_swing_progress

# Frame number, then an approximate palm center for each person in the
# original 720 x 720 GIF. Origin is top left. The interval ends at first touch.
TRACK = [
    (20, (104, 343), (416, 330)),
    (21, (132, 306), (415, 305)),
    (22, (153, 290), (409, 295)),
    (23, (189, 284), (397, 293)),
    (24, (261, 280), (369, 294)),
    (25, (296, 279), (346, 295)),
]
SOURCE_SHA256 = "003d685276f91ec19a8a004e27b95a1909482206e6b57d74d0c953f8942e6ac5"


def export_reference(clip_path, dest):
    """Transcode the original greeting at native speed with an attribution overlay."""
    import imageio_ffmpeg
    from matplotlib import get_data_path
    from PIL import ImageFont
    fontdir = Path(get_data_path()) / "fonts/ttf"
    font = ImageFont.truetype(str(fontdir / "DejaVuSans.ttf"), 18)
    bold = ImageFont.truetype(str(fontdir / "DejaVuSans-Bold.ttf"), 21)
    points = {k: (p, q) for k, p, q in TRACK}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as temp, Image.open(clip_path) as clip:
        output = Path(temp) / "reference.mp4"
        encoder = imageio_ffmpeg.write_frames(str(output), (720, 800), fps=25, codec="libx264",
            output_params=["-crf", "19", "-movflags", "+faststart"], macro_block_size=1)
        encoder.send(None)
        try:
            for k in range(clip.n_frames):
                clip.seek(k)
                if clip.info["duration"] != 40:
                    raise ValueError("expected 25 fps in original reference")
                frame = Image.new("RGB", (720, 800), "#152426")
                frame.paste(clip.convert("RGB"))
                ink = ImageDraw.Draw(frame)
                if k in points:
                    for x, y in points[k]:
                        ink.ellipse((x-8, y-8, x+8, y+8), outline="#ffd15c", width=3)
                ink.text((15, 727), "Human dap   /   1× playback", fill="white", font=bold)
                ink.text((15, 765), "OMFGrant • Contemporary Dap Up (2025) • Wikimedia Commons • CC0",
                         fill="white", font=font)
                encoder.send(frame.tobytes())
        finally:
            encoder.close()
        output.replace(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", nargs="?", type=Path, help="original Contemporary_Dap_Up.gif")
    parser.add_argument("--overlay", type=Path, help="write the marked-up frames as a contact sheet")
    parser.add_argument("--video", type=Path, help="export reference footage with its source on screen")
    args = parser.parse_args()
    if (args.overlay or args.video) and not args.clip:
        parser.error("an overlay or video requires the original clip")
    if args.clip:
        if hashlib.sha256(args.clip.read_bytes()).hexdigest() != SOURCE_SHA256:
            parser.error("expected the November 2025 720 x 720 Commons release")
        if args.video:
            export_reference(args.clip, args.video)
        if args.overlay:
            sheet = Image.new("RGB", (3 * 410, 2 * 200))
            with Image.open(args.clip) as clip:
                for i, (frame, *points) in enumerate(TRACK):
                    clip.seek(frame)
                    marked = clip.convert("RGB")
                    draw = ImageDraw.Draw(marked)
                    for x, y in points:
                        draw.ellipse((x-4, y-4, x+4, y+4), fill="yellow", outline="black")
                    draw.text((53, 191), f"Frame {frame}", fill="white", stroke_width=1, stroke_fill="black")
                    sheet.paste(marked.crop((50, 188, 460, 388)), (i % 3 * 410, i // 3 * 200))
            sheet.save(args.overlay)
    gap = np.array([right[0] - left[0] for _, left, right in TRACK])
    travel = (gap[0] - gap) / (gap[0] - gap[-1])
    # Frames in this segment are 40 ms each. Endpoints fix offset and scale.
    u = np.linspace(0, 1, len(TRACK))
    basis = u**2 * (1.0 - u)
    coefficient = np.dot(basis, travel - u**2) / np.dot(basis, basis)
    residual = dap_swing_progress(u, coefficient) - travel
    print(f"timing coefficient {coefficient:.8f}, normalized RMS error {np.sqrt(np.mean(residual**2)):.4f}")

    # Work in the image plane. The striker moves to the right; flip image y
    # to make positive offsets point above the straight line to contact.
    points = np.array([left for _, left, _ in TRACK], dtype=float)
    points[:, 1] *= -1
    displacement = points - points[0]
    chord = displacement[-1]
    length_sq = chord @ chord
    along = displacement @ chord / length_sq
    above = displacement @ np.array([-chord[1], chord[0]]) / length_sq
    basis = np.column_stack((3 * along * (1 - along)**2, 3 * along**2 * (1 - along)))
    controls = np.linalg.lstsq(basis, above, rcond=None)[0]
    residual = dap_swing_bow(along, controls) - above
    print(f"striker arc controls {controls[0]:.8f} {controls[1]:.8f}, "
          f"chord-normalized RMS error {np.sqrt(np.mean(residual**2)):.4f}")


if __name__ == "__main__":
    main()
