# Camera calibration profiles

One JSON per camera model. The pipeline undistorts each video using the profile
that matches its `camera_model` (set on upload). Unknown / un-calibrated cameras
fall back to `default.json`, which is an **identity transform — no correction**,
so footage is never warped by guesswork.

## Files

- `default.json` — identity (no-op). Fallback for unknown cameras.
- `transcend_dpb30.json`, `gopro_hero.json` — placeholders (currently identity).
  Replace by calibrating the real camera.

## Add / populate a profile

1. Print a checkerboard — the OpenCV pattern (9×6 **inner** corners):
   https://github.com/opencv/opencv/blob/4.x/doc/pattern.png
   Tape it flat to a rigid board (a clipboard works).
2. With the actual camera + lens + recording resolution, capture ~30–60s slowly
   moving the board to **all corners of the frame**, near and far, tilted at
   varied angles. Keep it well-lit and in focus. Edge coverage matters most —
   that's where lens distortion lives.
3. Run the calibrator (from `backend/`):
   ```
   python scripts/calibrate_camera.py \
       --input ../data/calibration/transcend_dpb30_checkerboard.mp4 \
       --camera-model transcend_dpb30 \
       --pattern-size 9x6 \
       --square-size 25
   ```
   It tries both the standard and fisheye lens models and keeps the better one,
   writing `transcend_dpb30.json` here and debug corner images to
   `data/calibration/transcend_dpb30_debug/` for visual verification.
4. Upload videos with `camera_model=transcend_dpb30` and they'll be undistorted
   automatically.

No real camera yet? Calibrate any webcam/phone and use a model name like
`laptop_webcam` — the plumbing is identical.

## Schema

| field | meaning |
|---|---|
| `camera_model` | profile key; matches the upload's `camera_model` |
| `lens_model` | `"standard"` (cv2.undistort) or `"fisheye"` (cv2.fisheye) |
| `image_size` | `[width, height]` the calibration was done at (scaled to the video) |
| `intrinsic_matrix` | 3×3 `[[fx,0,cx],[0,fy,cy],[0,0,1]]` |
| `distortion_coefficients` | `[k1,k2,p1,p2,k3]` (standard) or `[k1,k2,k3,k4]` (fisheye) |
| `reprojection_error_pixels` | mean calibration error — quality metric |
| `num_calibration_frames` | board views used |
| `pattern_size`, `square_size_mm` | the board geometry used |
| `notes` | freeform |

## Is the calibration good?

Judge by `reprojection_error_pixels`:

- **< 1.0 px** — excellent.
- **< 2.0 px** — acceptable.
- **> 2.0 px** — poor; recapture with more varied angles/distances and better
  edge coverage. The calibrator prints a warning above 2.0.

## Notes

- Undistortion adds ~20–40% per-frame processing cost and stores correction maps
  per resolution (built once, reused).
- The raw upload is always preserved; undistortion never deletes it.
