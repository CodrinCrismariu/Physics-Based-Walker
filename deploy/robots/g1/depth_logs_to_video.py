#!/usr/bin/env python3
"""Convert a run's depth binary logs into an MP4 video.

Usage:
    python3 depth_logs_to_video.py <run_dir>

Example:
    python3 depth_logs_to_video.py logs/hlip_mdn_20260505_123456

Reads all depth/*.bin files (raw float32, 24×32), normalises to 0-255,
applies a colormap, upscales for visibility, and writes depth_video.mp4
at 50 fps into the run directory.
"""

import argparse
import pathlib
import struct
import sys

import cv2
import numpy as np

DEPTH_HEIGHT = 24
DEPTH_WIDTH = 32
DEPTH_PIXELS = DEPTH_HEIGHT * DEPTH_WIDTH
FPS = 50
UPSCALE = 16  # 32×24 → 512×384 for visibility


def load_depth_frame(path: pathlib.Path) -> np.ndarray:
    """Load a raw float32 binary depth frame."""
    data = path.read_bytes()
    expected = DEPTH_PIXELS * 4  # 4 bytes per float32
    if len(data) != expected:
        raise ValueError(f"{path.name}: expected {expected} bytes, got {len(data)}")
    arr = np.frombuffer(data, dtype=np.float32).reshape(DEPTH_HEIGHT, DEPTH_WIDTH)
    return arr


def depth_to_color(frame: np.ndarray, vmin: float = 0.0, vmax: float = 10.0) -> np.ndarray:
    """Normalise depth to 0-255 and apply TURBO colormap."""
    normed = np.clip((frame - vmin) / (vmax - vmin), 0.0, 1.0)
    gray = (normed * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return colored


def main():
    parser = argparse.ArgumentParser(description="Convert depth logs to video")
    parser.add_argument("run_dir", type=str, help="Path to the run directory (e.g. logs/hlip_mdn_...)")
    parser.add_argument("--fps", type=int, default=FPS, help=f"Video frame rate (default {FPS})")
    parser.add_argument("--upscale", type=int, default=UPSCALE, help=f"Upscale factor (default {UPSCALE})")
    parser.add_argument("--vmin", type=float, default=0.0, help="Depth min for colormap (m)")
    parser.add_argument("--vmax", type=float, default=10.0, help="Depth max for colormap (m)")
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    depth_dir = run_dir / "depth"

    if not depth_dir.exists():
        print(f"Error: depth directory not found: {depth_dir}")
        sys.exit(1)

    # Collect and sort depth files
    bin_files = sorted(depth_dir.glob("depth_*.bin"))
    if not bin_files:
        print(f"Error: no depth_*.bin files in {depth_dir}")
        sys.exit(1)

    print(f"Found {len(bin_files)} depth frames in {depth_dir}")

    # Output video
    out_w = DEPTH_WIDTH * args.upscale
    out_h = DEPTH_HEIGHT * args.upscale
    out_path = run_dir / "depth_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (out_w, out_h))

    if not writer.isOpened():
        print(f"Error: could not open video writer for {out_path}")
        sys.exit(1)

    for i, f in enumerate(bin_files):
        frame = load_depth_frame(f)
        colored = depth_to_color(frame, vmin=args.vmin, vmax=args.vmax)
        upscaled = cv2.resize(colored, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        # Add step number overlay
        cv2.putText(upscaled, f"step {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        writer.write(upscaled)

        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(bin_files)} frames...")

    writer.release()
    duration = len(bin_files) / args.fps
    print(f"Done — {out_path} ({len(bin_files)} frames, {duration:.1f}s at {args.fps} fps)")


if __name__ == "__main__":
    main()
