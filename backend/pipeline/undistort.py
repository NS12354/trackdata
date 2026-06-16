"""Lens distortion removal — turn wide-angle / fisheye footage into rectilinear
("normal") footage.

Two modes:

* **Calibrated** — if ``camera_intrinsics.json`` exists (produced by
  ``scripts/calibrate_camera.py`` from a checkerboard clip), we undistort with the
  real intrinsics K + distortion D. This is metrically accurate: straight lines
  become straight, and the rectified pinhole intrinsics are exact — which makes the
  hand/body pose data spatially correct for labs.

* **Estimate** — no calibration yet. We synthesize a plausible pinhole camera from
  an assumed field-of-view and a single ``strength`` knob (negative radial k1/k2)
  that flattens barrel distortion by eye. Not metrically exact, but makes footage
  look normal immediately and is tunable via the UI slider.

The undistort maps are built once per (width, height) and reused for every frame.
``apply()`` returns a same-size BGR frame. ``new_K`` exposes the rectified pinhole
intrinsics so downstream geometry (back-projection in body_pose) stays consistent.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from config import settings

log = logging.getLogger("revisent.undistort")

BACKEND_DIR = Path(__file__).resolve().parent
INTRINSICS_PATH = BACKEND_DIR / "camera_intrinsics.json"
# Runtime overrides written by the UI slider (takes precedence over static config
# so a new strength applies to the next job without a server restart).
RUNTIME_PATH = settings.data_dir / "undistort.json"


def get_runtime_params() -> dict:
    """Effective de-warp params: runtime override file on top of static config."""
    params = {
        "enabled": settings.enable_undistort,
        "strength": settings.undistort_strength,
        "zoom": settings.undistort_zoom,
        "fov_deg": settings.undistort_assumed_fov_deg,
    }
    try:
        if RUNTIME_PATH.exists():
            params.update({k: v for k, v in json.loads(RUNTIME_PATH.read_text()).items()
                           if k in params and v is not None})
    except Exception:  # noqa: BLE001
        pass
    return params


def set_runtime_params(**kw) -> dict:
    """Persist UI slider changes; returns the merged params."""
    current = get_runtime_params()
    current.update({k: v for k, v in kw.items() if k in current and v is not None})
    RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_PATH.write_text(json.dumps(current, indent=2))
    return current


def _f_from_fov(width: int, fov_deg: float) -> float:
    """Focal length (px) for a horizontal field of view."""
    fov = max(20.0, min(170.0, fov_deg))
    return (0.5 * width) / math.tan(math.radians(fov) / 2.0)


class Undistorter:
    """Builds + caches a remap for one frame size and applies it per frame."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        strength: Optional[float] = None,
        zoom: Optional[float] = None,
        fov_deg: Optional[float] = None,
    ):
        self.width = int(width)
        self.height = int(height)
        # Per-call overrides (preview) fall back to the runtime params (UI slider),
        # which themselves fall back to static config.
        rp = get_runtime_params()
        self.strength = rp["strength"] if strength is None else float(strength)
        self.zoom = rp["zoom"] if zoom is None else float(zoom)
        self.fov_deg = rp["fov_deg"] if fov_deg is None else float(fov_deg)

        self.mode = "estimate"
        self.new_K: Optional[np.ndarray] = None
        self._map1 = None
        self._map2 = None
        self._build()

    # ---- map construction --------------------------------------------------
    def _build(self) -> None:
        calib = _load_calibration()
        if calib is not None:
            try:
                self._build_calibrated(calib)
                self.mode = f"calibrated:{calib.get('model', 'pinhole')}"
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("calibration unusable (%s); falling back to estimate", exc)
        self._build_estimate()

    def _build_calibrated(self, calib: dict) -> None:
        W, H = self.width, self.height
        cw, ch = calib["image_size"]
        sx, sy = W / float(cw), H / float(ch)
        K = np.array(calib["K"], dtype=np.float64)
        # Scale intrinsics from the calibration resolution to this frame size.
        K = K.copy()
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        D = np.array(calib["D"], dtype=np.float64).reshape(-1, 1)

        if calib.get("model") == "fisheye":
            balance = float(np.clip(self.zoom, 0.0, 1.0))
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D[:4], (W, H), np.eye(3), balance=balance
            )
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(
                K, D[:4], np.eye(3), new_K, (W, H), cv2.CV_16SC2
            )
        else:
            alpha = float(np.clip(self.zoom, 0.0, 1.0))
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), alpha, (W, H))
            m1, m2 = cv2.initUndistortRectifyMap(
                K, D, None, new_K, (W, H), cv2.CV_16SC2
            )
        self.new_K, self._map1, self._map2 = new_K, m1, m2

    def _build_estimate(self) -> None:
        """Bounded barrel-removal remap, normalized by the half-diagonal so the
        radius stays in [0,1] across the whole frame (any aspect ratio). This
        avoids the polynomial blow-up that OpenCV's f-normalized model suffers on
        tall portrait frames, and never produces black borders (it samples
        inward). strength 0 => identity."""
        W, H = self.width, self.height
        s = float(np.clip(self.strength, 0.0, 1.0))
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        R = 0.5 * math.hypot(W, H)               # half-diagonal: max radius ≈ 1
        k = 0.6 * s                              # bounded barrel-removal coefficient

        ys, xs = np.indices((H, W), dtype=np.float32)
        nx = (xs - cx) / R
        ny = (ys - cy) / R
        r2 = nx * nx + ny * ny                   # ∈ [0, ~1]
        # Sample inward at the edges to flatten barrel bulge; scale ∈ [1-k, 1].
        scale = 1.0 - k * r2
        # zoom: slightly tighten the crop as the knob increases (no black border).
        scale *= 1.0 - 0.25 * float(np.clip(self.zoom, 0.0, 1.0))

        self._map1 = (cx + nx * scale * R).astype(np.float32)
        self._map2 = (cy + ny * scale * R).astype(np.float32)
        f = _f_from_fov(W, self.fov_deg)
        self.new_K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]], dtype=np.float64)

    # ---- application -------------------------------------------------------
    @property
    def active(self) -> bool:
        # In estimate mode, strength 0 is a no-op; skip the remap entirely.
        return not (self.mode == "estimate" and self.strength <= 0.0)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.active or self._map1 is None:
            return frame
        return cv2.remap(frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    def meta(self) -> dict:
        K = self.new_K
        return {
            "applied": bool(self.active),
            "mode": self.mode,
            "strength": round(self.strength, 3),
            "zoom": round(self.zoom, 3),
            "assumed_fov_deg": round(self.fov_deg, 1) if "estimate" in self.mode else None,
            "rectified_intrinsics": (
                {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                 "cx": float(K[0, 2]), "cy": float(K[1, 2])}
                if K is not None else None
            ),
        }


def _load_calibration() -> Optional[dict]:
    if not INTRINSICS_PATH.exists():
        return None
    try:
        c = json.loads(INTRINSICS_PATH.read_text())
        if "K" in c and "D" in c and "image_size" in c:
            return c
    except Exception:  # noqa: BLE001
        pass
    return None


def is_calibrated() -> bool:
    return _load_calibration() is not None


def make_undistorter(width: int, height: int, **overrides) -> Optional[Undistorter]:
    """Return an Undistorter if undistortion is enabled, else None."""
    if not get_runtime_params()["enabled"]:
        return None
    u = Undistorter(width, height, **overrides)
    return u if u.active else None
