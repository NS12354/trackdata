#!/usr/bin/env python3
"""Camera intrinsics calibration — non-optional for spatially-grounded data.

Without intrinsics (focal length, principal point, distortion) nobody can use
your data for anything 3D. Calibrate ONCE per camera model: print a checkerboard,
record ~20 images of it at varied angles/distances, run this.

Usage:
    # from a folder of checkerboard photos:
    python scripts/calibrate_camera.py --images calib/*.jpg --cols 9 --rows 6 --square-mm 25

    # or sample frames from a calibration video:
    python scripts/calibrate_camera.py --video calib.mp4 --cols 9 --rows 6 --square-mm 25

Writes backend/camera_intrinsics.json — the per-camera-model record that ships in
every dataset's capture metadata. (cols/rows = INNER corners of the board.)
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backend" / "camera_intrinsics.json"


def _frames_from_video(path: str, stride: int):
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            yield f
        i += 1
    cap.release()


def calibrate(images, cols, rows, square_mm):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    obj_points, img_points = [], []
    size = None
    found = 0
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
            obj_points.append(objp); img_points.append(corners); found += 1
    if found < 5:
        raise SystemExit(f"only {found} usable checkerboard views found (need >=5). "
                         "Check --cols/--rows (INNER corners) and image quality.")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    w, h = size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fov_x = float(np.degrees(2 * np.arctan(w / (2 * fx))))
    fov_y = float(np.degrees(2 * np.arctan(h / (2 * fy))))
    return {
        "resolution": {"width": int(w), "height": int(h)},
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "distortion": [float(v) for v in dist.flatten()],
        "distortion_model": "opencv_radtan (k1 k2 p1 p2 k3)",
        "fov_deg": {"horizontal": round(fov_x, 2), "vertical": round(fov_y, 2)},
        "reprojection_rms_px": round(float(rms), 4),
        "views_used": found,
        "board": {"inner_cols": cols, "inner_rows": rows, "square_mm": square_mm},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="*", default=[], help="glob(s) of checkerboard images")
    ap.add_argument("--video", default=None, help="calibration video (frames sampled)")
    ap.add_argument("--video-stride", type=int, default=15)
    ap.add_argument("--cols", type=int, required=True, help="inner corners across")
    ap.add_argument("--rows", type=int, required=True, help="inner corners down")
    ap.add_argument("--square-mm", type=float, required=True)
    ap.add_argument("--camera-model", default="unknown", help="label stored in the record")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    imgs = []
    for pat in args.images:
        imgs += [cv2.imread(p) for p in sorted(glob.glob(pat))]
    if args.video:
        imgs += list(_frames_from_video(args.video, args.video_stride))
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        raise SystemExit("no input images/frames — pass --images or --video")

    rec = calibrate(imgs, args.cols, args.rows, args.square_mm)
    rec["camera_model"] = args.camera_model
    Path(args.out).write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    print(f"\nwrote {args.out}")
    print(f"reprojection RMS {rec['reprojection_rms_px']}px "
          f"({'good' if rec['reprojection_rms_px'] < 1.0 else 'recalibrate — aim <1px'})")
    print(f"Set ego_camera_fov_deg={rec['fov_deg']['horizontal']} in config to match.")


if __name__ == "__main__":
    main()
