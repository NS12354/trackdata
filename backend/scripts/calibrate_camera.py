#!/usr/bin/env python3
"""Camera calibration — produce backend/camera_intrinsics.json from a checkerboard.

This gives the *accurate* de-fisheye: real intrinsics K + distortion D so straight
lines become straight and the rectified pinhole intrinsics are exact (which makes
the exported hand/body pose spatially correct for labs).

How to capture:
  1. Print a checkerboard (e.g. https://github.com/opencv/opencv/blob/4.x/doc/pattern.png)
     and tape it flat to a rigid board. Count the INNER corners (where black squares
     meet) — a standard 10x7-square board has 9x6 inner corners.
  2. With the SAME camera + lens + resolution you record footage on, take a ~20-40s
     video slowly moving the board around the frame: center, all four corners, tilted
     at various angles, near and far. Cover the edges — that's where distortion lives.

Usage:
  python scripts/calibrate_camera.py CLIP.mp4 --cols 9 --rows 6 --square-mm 25
  python scripts/calibrate_camera.py CLIP.mp4 --cols 9 --rows 6 --fisheye   # very-wide lenses

Outputs backend/camera_intrinsics.json. Once it exists, the pipeline automatically
switches from estimate mode to calibrated mode on the next processed clip.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BACKEND_DIR / "camera_intrinsics.json"


def _sample_corners(video: Path, cols: int, rows: int, every: int, max_views: int):
    """Detect checkerboard inner corners across sampled frames."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"could not open {video}")
    pattern = (cols, rows)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    img_points, size, idx, kept = [], None, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            found, corners = cv2.findChessboardCorners(gray, pattern, flags)
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners)
                kept += 1
                print(f"  frame {idx}: board found ({kept})")
                if kept >= max_views:
                    break
        idx += 1
    cap.release()
    return img_points, size


def main() -> None:
    ap = argparse.ArgumentParser(description="Checkerboard camera calibration")
    ap.add_argument("video", type=Path, help="checkerboard calibration clip")
    ap.add_argument("--cols", type=int, required=True, help="inner corners per row")
    ap.add_argument("--rows", type=int, required=True, help="inner corners per column")
    ap.add_argument("--square-mm", type=float, default=25.0, help="square size (mm)")
    ap.add_argument("--fisheye", action="store_true", help="use the fisheye model (very-wide lenses)")
    ap.add_argument("--every", type=int, default=10, help="sample every Nth frame")
    ap.add_argument("--max-views", type=int, default=40, help="max boards to use")
    args = ap.parse_args()

    print(f"scanning {args.video} for a {args.cols}x{args.rows} board…")
    img_points, size = _sample_corners(args.video, args.cols, args.rows, args.every, args.max_views)
    if len(img_points) < 8:
        sys.exit(f"only {len(img_points)} good views found — need ≥8. Record more board angles/positions.")
    print(f"calibrating from {len(img_points)} views at {size[0]}x{size[1]}…")

    # 3D object points for one board (Z=0 plane), scaled to real mm.
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square_mm
    obj_points = [objp for _ in img_points]

    if args.fisheye:
        K = np.zeros((3, 3)); D = np.zeros((4, 1))
        op = [o.reshape(-1, 1, 3) for o in obj_points]
        rms, K, D, _, _ = cv2.fisheye.calibrate(
            op, img_points, size, K, D,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
        )
        model, Dlist = "fisheye", D.flatten().tolist()
    else:
        rms, K, D, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
        model, Dlist = "pinhole", D.flatten().tolist()

    fov_x = math.degrees(2 * math.atan(size[0] / (2 * K[0, 0])))
    record = {
        "model": model,
        "image_size": [int(size[0]), int(size[1])],
        "K": K.tolist(),
        "D": Dlist,
        "rms_reproj_error_px": round(float(rms), 4),
        "fov_deg": round(fov_x, 1),
        "n_views": len(img_points),
        "square_mm": args.square_mm,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }
    OUT_PATH.write_text(json.dumps(record, indent=2))
    print(f"\n✓ wrote {OUT_PATH}")
    print(f"  model={model}  rms={rms:.3f}px  hfov≈{fov_x:.1f}°")
    if rms > 1.0:
        print("  ⚠ reprojection error >1px — consider recapturing with more varied board poses.")
    print("  The pipeline will use this automatically on the next processed clip.")


if __name__ == "__main__":
    main()
