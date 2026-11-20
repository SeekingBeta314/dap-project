"""Refit the two travel envelopes from the public Inter-X handshake preview."""

import argparse
import hashlib
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dap.hand_timing import progress


def projected_gaps(path):
    """Smallest horizontal gap between the actors' silhouettes at arm height."""
    frames = imageio_ffmpeg.read_frames(str(path))
    info = next(frames)
    gaps = []
    for pixels in frames:
        image = Image.frombytes("RGB", info["size"], pixels)
        image = image.resize((468, 257), Image.Resampling.BICUBIC)
        red, green, blue = np.asarray(image, dtype=np.int16).transpose(2, 0, 1)
        left = (red > green + 45) & (green > blue + 10)
        right = (blue > red + 60) & (green > red + 25)
        gaps.append(min(np.flatnonzero(right[y])[0] - np.flatnonzero(left[y])[-1]
                        for y in range(76, 126) if left[y].any() and right[y].any()))
    return np.array(gaps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path, help="Inter-X static/videos/handshake.mp4")
    args = parser.parse_args()
    sha = hashlib.sha256(args.clip.read_bytes()).hexdigest()
    if sha != "c6dba1217193a3a607f413abb225604e7da98ffb8a923982da4be6bb48856a82":
        parser.error("Expected the Inter-X preview used to select these frames and colors")
    gaps = projected_gaps(args.clip)
    for name, start, end in (("reach", 0, 26), ("release", 91, 108)):
        gap = gaps[start:end + 1]
        target = (gap - gap[0]) / (gap[-1] - gap[0])
        u = np.linspace(0, 1, len(gap))
        fit = least_squares(lambda p: progress(u, *p) - target, [.4, .7, .45],
                            bounds=([0, .1, 0], [1, 1, .9]))
        rms = np.sqrt(np.mean(fit.fun**2))
        print(f"{name}: coefficients {fit.x}, normalized RMS error {rms:.4f}")


if __name__ == "__main__":
    main()
