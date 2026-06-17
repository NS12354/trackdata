"""Camera intrinsics from device metadata (Phase 0.5) — better than guessing.

Wrist depth-from-hand-scale divides by the focal length, so an assumed 90 deg
FOV on an iPhone (~68 deg for 1x video) biases every depth ~25% short. Phone
clips carry their device model in QuickTime metadata; we map known devices to
their typical VIDEO horizontal FOV.

Accuracy ladder (worst to best):
  90 deg default            -> ~25% depth bias on phone footage
  device preset (this file) -> within ~5% (stabilization crop varies a few deg)
  checkerboard calibration  -> ~1% (scripts/calibrate_camera.py; always wins
                               when camera_intrinsics.json exists)

Presets assume the DEFAULT (1x main) lens — ultrawide (0.5x) recordings are not
tagged distinguishably; if you shoot 0.5x, set EGO_CAMERA_FOV_DEG explicitly.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

try:
    from config import settings
except Exception:  # pragma: no cover - allows import without app config
    settings = None

log = logging.getLogger("revisent.intrinsics")

# (substring of the lowercased model string) -> horizontal video FOV in degrees.
# iPhone main camera: 24-26mm equivalent stills; standard video stabilization
# crops in ~10%, landing around 68 deg horizontal for 16:9 video.
_DEVICE_FOV_PRESETS: Tuple[Tuple[str, float, str], ...] = (
    ("iphone", 68.0, "iPhone main (1x) video"),
    ("gopro", 118.0, "GoPro wide video"),
    ("hero", 118.0, "GoPro wide video"),
)


def device_model(video_path: Path) -> Optional[str]:
    """The recording device's model string from container metadata, if any."""
    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format_tags",
             "-of", "json", str(video_path)],
            capture_output=True, text=True, check=True,
        )
        tags = json.loads(out.stdout).get("format", {}).get("tags", {}) or {}
        for key in ("com.apple.quicktime.model", "model", "com.android.model"):
            if tags.get(key):
                return str(tags[key])
    except Exception:  # noqa: BLE001
        return None
    return None


def fov_from_model(model: Optional[str]) -> Optional[Tuple[float, str]]:
    """(fov_deg, preset_name) for a known device model string, else None."""
    if not model:
        return None
    low = model.lower()
    for needle, fov, name in _DEVICE_FOV_PRESETS:
        if needle in low:
            return fov, name
    return None


def fov_for_video(video_path: Path) -> Tuple[float, str]:
    """Best-available horizontal FOV (degrees) for this clip, with provenance.

    Precedence: checkerboard calibration file > device preset (when
    EGO_CAMERA_FOV_AUTO) > configured EGO_CAMERA_FOV_DEG.
    """
    default = float(getattr(settings, "ego_camera_fov_deg", 90.0)) if settings else 90.0
    # 1) Real calibration always wins.
    try:
        from .capture_meta import load_intrinsics
        intr = load_intrinsics()
        if intr and intr.get("fov_deg"):
            return float(intr["fov_deg"]), "calibrated (camera_intrinsics.json)"
    except Exception:  # noqa: BLE001
        pass
    # 2) Device preset.
    auto = bool(getattr(settings, "ego_camera_fov_auto", True)) if settings else True
    if auto:
        model = device_model(Path(video_path))
        preset = fov_from_model(model)
        if preset:
            fov, name = preset
            log.info("FOV %s: %.1f deg from device preset '%s' (%s)",
                     Path(video_path).name, fov, name, model)
            return fov, f"device preset: {name} ({model})"
    # 3) Configured default.
    return default, f"configured default ({default:.0f} deg)"
