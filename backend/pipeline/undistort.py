"""Lens distortion removal — turn wide-angle / fisheye footage into rectilinear
("normal") footage, using per-camera-model calibration profiles.

Correction is **profile-driven only** (no guessing): each video carries a
``camera_model``; we load ``calibration_profiles/{camera_model}.json`` (falling
back to ``default.json``, an identity no-op) and undistort with its real
intrinsics K + distortion D. A real calibration is the only thing that warps
pixels — an unknown camera is passed through untouched.

Applied inside the anonymization re-encode (see anonymize.py), so the stored
output AND all downstream pose run on rectilinear frames. The undistort maps are
built once per (width, height) and reused. ``new_K`` exposes the rectified pinhole
intrinsics so downstream geometry stays consistent.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import settings

log = logging.getLogger("revisent.undistort")

PROFILES_DIR = Path(__file__).resolve().parent / "calibration_profiles"
DEFAULT_MODEL = "default"


def profiles_dir() -> Path:
    return PROFILES_DIR


def list_profiles() -> list[dict]:
    """All available camera profiles (for the upload picker / dashboard)."""
    out = []
    for p in sorted(PROFILES_DIR.glob("*.json")):
        try:
            j = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        out.append({
            "camera_model": j.get("camera_model", p.stem),
            "lens_model": j.get("lens_model", "standard"),
            "calibrated": not _is_identity(j),
            "reprojection_error_pixels": j.get("reprojection_error_pixels"),
        })
    return out


def load_profile(camera_model: Optional[str]) -> dict:
    """Load a camera profile, falling back to the identity default if missing."""
    name = camera_model or DEFAULT_MODEL
    path = PROFILES_DIR / f"{name}.json"
    if not path.exists():
        log.warning("no calibration profile for camera_model=%r; using default (no-op)", name)
        path = PROFILES_DIR / f"{DEFAULT_MODEL}.json"
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        log.warning("profile %s unreadable (%s); using identity", path.name, exc)
        return _identity_profile()


def _identity_profile() -> dict:
    return {
        "camera_model": "default", "lens_model": "standard",
        "image_size": [1920, 1080],
        "intrinsic_matrix": [[1920, 0, 960], [0, 1920, 540], [0, 0, 1]],
        "distortion_coefficients": [0, 0, 0, 0, 0],
        "reprojection_error_pixels": 0.0,
    }


def _is_identity(profile: dict) -> bool:
    """A profile is a no-op if all distortion coefficients are zero."""
    D = profile.get("distortion_coefficients") or []
    return all(float(c) == 0.0 for c in D)


class Undistorter:
    """Builds + caches a remap for one frame size from a calibration profile."""

    def __init__(self, width: int, height: int, profile: dict):
        self.width = int(width)
        self.height = int(height)
        self.profile = profile
        self.camera_model = profile.get("camera_model", "default")
        self.lens_model = profile.get("lens_model", "standard")
        self.reproj_error = profile.get("reprojection_error_pixels")
        self.identity = _is_identity(profile)
        self.new_K: Optional[np.ndarray] = None
        self._map1 = None
        self._map2 = None
        if not self.identity:
            self._build()

    def _build(self) -> None:
        W, H = self.width, self.height
        cw, ch = self.profile["image_size"]
        sx, sy = W / float(cw), H / float(ch)
        # Scale the calibration intrinsics from their resolution to this frame size.
        K = np.array(self.profile["intrinsic_matrix"], dtype=np.float64).copy()
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        D = np.array(self.profile["distortion_coefficients"], dtype=np.float64)

        if self.lens_model == "fisheye":
            Df = D.reshape(-1, 1)[:4]
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, Df, (W, H), np.eye(3), balance=0.0
            )
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(
                K, Df, np.eye(3), new_K, (W, H), cv2.CV_16SC2
            )
        else:
            Ds = D.reshape(-1)
            # alpha=0 crops to valid pixels (removes black borders from correction).
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, Ds, (W, H), 0, (W, H))
            m1, m2 = cv2.initUndistortRectifyMap(K, Ds, None, new_K, (W, H), cv2.CV_16SC2)
        self.new_K, self._map1, self._map2 = new_K, m1, m2

    @property
    def active(self) -> bool:
        return (not self.identity) and self._map1 is not None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.active:
            return frame
        return cv2.remap(frame, self._map1, self._map2,
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def meta(self) -> dict:
        K = self.new_K
        return {
            "applied": bool(self.active),
            "camera_model": self.camera_model,
            "lens_model": self.lens_model if self.active else None,
            "reprojection_error_pixels": self.reproj_error if self.active else None,
            "rectified_intrinsics": (
                {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                 "cx": float(K[0, 2]), "cy": float(K[1, 2])}
                if K is not None else None
            ),
        }


def make_undistorter(width: int, height: int, camera_model: Optional[str] = None) -> Optional[Undistorter]:
    """Return an Undistorter for the camera's profile, or None when it's a no-op
    (undistort disabled, or an identity/uncalibrated profile)."""
    if not settings.enable_undistort:
        return None
    u = Undistorter(width, height, load_profile(camera_model))
    return u if u.active else None


def is_calibrated(camera_model: Optional[str] = None) -> bool:
    return not _is_identity(load_profile(camera_model))
