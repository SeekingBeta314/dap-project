"""Fit the normalized inward swing to OMFGrant's Contemporary Dap Up (CC0).

Run without arguments to refit the published annotations. Supply the original
GIF and --overlay to inspect the hand-marked points on the six source frames.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dap.hand_timing import dap_swing_progress

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", nargs="?", type=Path, help="original Contemporary_Dap_Up.gif")
    parser.add_argument("--overlay", type=Path, help="write the marked-up frames as a contact sheet")
    args = parser.parse_args()
    if args.overlay and not args.clip:
        parser.error("--overlay requires the original clip")
    if args.clip:
        if hashlib.sha256(args.clip.read_bytes()).hexdigest() != SOURCE_SHA256:
            parser.error("expected the November 2025 720 x 720 Commons release")
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
    print(f"coefficient {coefficient:.8f}, normalized RMS error {np.sqrt(np.mean(residual**2)):.4f}")


if __name__ == "__main__":
    main()
