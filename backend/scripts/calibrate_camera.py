#!/usr/bin/env python3
"""Camera calibration — populate a per-camera-model profile from a checkerboard clip.

Run once per camera model. It detects the checkerboard across sampled frames,
calibrates with BOTH the standard (pinhole) and fisheye models, keeps whichever
has the lower reprojection error, and writes the profile to the registry at
``backend/pipeline/calibration_profiles/{camera_model}.json``. The pipeline then
undistorts any video tagged with that camera model.

Usage (from the backend/ directory):
  python scripts/calibrate_camera.py \\
      --input ../data/calibration/transcend_dpb30_checkerboard.mp4 \\
      --camera-model transcend_dpb30 \\
      --pattern-size 9x6 \\
      --square-size 25

Capture tips: print the OpenCV checkerboard (9x6 INNER corners), tape it flat to a
rigid board, and record ~30-60s slowly moving it to all corners of the frame, near
and far, tilted at various angles, well-lit and in focus. Edge coverage matters —
that's where distortion lives. See calibration_profiles/README.md.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent          # .../revisent-mvp/backend
REPO_ROOT = BACKEND_DIR.parent                                 # .../revisent-mvp
PROFILES_DIR = BACKEND_DIR / "pipeline" / "calibration_profiles"
DEBUG_ROOT = REPO_ROOT / "data" / "calibration"

MIN_FRAMES = 15
FISHEYE_TRY_THRESHOLD = 1.0   # if standard reproj error exceeds this, also try fisheye
POOR_QUALITY_THRESHOLD = 2.0  # warn above this


def _parse_pattern(s: str) -> tuple[int, int]:
    try:
        c, r = s.lower().split("x")
        return int(c), int(r)
    except Exception:
        raise SystemExit(f"--pattern-size must look like '9x6', got {s!r}")


def _collect_corners(video: Path, pattern: tuple[int, int], every: int, debug_dir: Path):
    """Detect + refine chessboard corners across sampled frames; save debug viz."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video}")

    find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    subpix_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    debug_dir.mkdir(parents=True, exist_ok=True)

    img_points, size, idx, kept, attempted = [], None, 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            attempted += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = (gray.shape[1], gray.shape[0])
            found, corners = cv2.findChessboardCorners(gray, pattern, find_flags)
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_crit)
                img_points.append(corners)
                kept += 1
                vis = frame.copy()
                cv2.drawChessboardCorners(vis, pattern, corners, found)
                cv2.imwrite(str(debug_dir / f"corners_{kept:03d}_frame{idx:06d}.jpg"), vis)
                print(f"  frame {idx}: board found ({kept} kept)")
        idx += 1
    cap.release()
    print(f"sampled {attempted} frames; detected the board in {kept}")
    return img_points, size


def _calibrate_standard(obj_points, img_points, size):
    rms, K, D, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    return float(rms), K, D


def _calibrate_fisheye(obj_points, img_points, size):
    n = len(obj_points)
    op = [o.reshape(1, -1, 3).astype(np.float64) for o in obj_points]
    ip = [p.reshape(1, -1, 2).astype(np.float64) for p in img_points]
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), np.float64) for _ in range(n)]
    tvecs = [np.zeros((1, 1, 3), np.float64) for _ in range(n)]
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    rms, K, D, _, _ = cv2.fisheye.calibrate(op, ip, size, K, D, rvecs, tvecs, flags, crit)
    return float(rms), K, D


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-camera-model checkerboard calibration")
    ap.add_argument("--input", type=Path, required=True, help="checkerboard calibration clip")
    ap.add_argument("--camera-model", required=True, help="profile key, e.g. transcend_dpb30")
    ap.add_argument("--pattern-size", default="9x6", help="INNER corners, COLSxROWS (default 9x6)")
    ap.add_argument("--square-size", type=float, default=25.0, help="square edge length in mm")
    ap.add_argument("--sample-fps", type=float, default=1.0, help="frames/sec to sample (default 1)")
    ap.add_argument("--every", type=int, default=0, help="sample every Nth frame (overrides --sample-fps)")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")
    pattern = _parse_pattern(args.pattern_size)

    cap = cv2.VideoCapture(str(args.input))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    every = args.every if args.every > 0 else max(1, round(fps / max(0.1, args.sample_fps)))
    print(f"calibrating '{args.camera_model}' from {args.input.name} "
          f"(pattern {pattern[0]}x{pattern[1]}, square {args.square_size}mm, every {every} frames)")

    debug_dir = DEBUG_ROOT / f"{args.camera_model}_debug"
    img_points, size = _collect_corners(args.input, pattern, every, debug_dir)
    if len(img_points) < MIN_FRAMES:
        raise SystemExit(
            f"only {len(img_points)} usable board views (need ≥{MIN_FRAMES}). "
            f"Re-capture with the board at more positions/angles, well-lit and in focus. "
            f"Debug images: {debug_dir}")

    # One board's 3D object points (Z=0 plane), scaled to real mm.
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= args.square_size
    obj_points = [objp for _ in img_points]

    err_std, K, D = _calibrate_standard(obj_points, img_points, size)
    lens, rms, Kc, Dc = "standard", err_std, K, D
    print(f"standard model: reproj error = {err_std:.3f}px")

    # Body cams sit on the standard/fisheye boundary — try fisheye if standard is weak.
    if err_std > FISHEYE_TRY_THRESHOLD:
        try:
            err_fish, Kf, Df = _calibrate_fisheye(obj_points, img_points, size)
            print(f"fisheye model:  reproj error = {err_fish:.3f}px")
            if err_fish < err_std:
                lens, rms, Kc, Dc = "fisheye", err_fish, Kf, Df
        except cv2.error as exc:
            print(f"fisheye calibration failed ({str(exc).splitlines()[-1][:120]}); keeping standard")

    profile = {
        "camera_model": args.camera_model,
        "calibration_date": date.today().isoformat(),
        "lens_model": lens,
        "image_size": [int(size[0]), int(size[1])],
        "intrinsic_matrix": np.asarray(Kc).reshape(3, 3).tolist(),
        "distortion_coefficients": np.asarray(Dc).reshape(-1).tolist(),
        "reprojection_error_pixels": round(rms, 4),
        "num_calibration_frames": len(img_points),
        "pattern_size": [pattern[0], pattern[1]],
        "square_size_mm": args.square_size,
        "notes": f"Auto-calibrated from {args.input.name}. Debug corners in {debug_dir}.",
    }
    out = PROFILES_DIR / f"{args.camera_model}.json"
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2))

    print("\n" + "=" * 60)
    print(f"✓ wrote {out}")
    print(f"  frames used : {len(img_points)}")
    print(f"  lens model  : {lens}")
    print(f"  reproj error: {rms:.3f}px")
    print(f"  debug images: {debug_dir}")
    if rms > POOR_QUALITY_THRESHOLD:
        print(f"  ⚠ reproj error >{POOR_QUALITY_THRESHOLD}px — calibration quality may be poor. "
              f"Re-capture with more varied angles, distances, and edge coverage.")
    print("  The pipeline will undistort videos tagged with this camera model.")


if __name__ == "__main__":
    main()
