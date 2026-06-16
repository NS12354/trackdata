"""Runtime processing flags — toggles that take effect on the next job without a
server restart (persisted to data/processing.json), layered over static config.

Currently: face-blur on/off. Keep the privacy-safe default (on); turn off only for
footage known to contain no faces.
"""
from __future__ import annotations

import json

from config import settings

RUNTIME_PATH = settings.data_dir / "processing.json"


def _read() -> dict:
    try:
        if RUNTIME_PATH.exists():
            return json.loads(RUNTIME_PATH.read_text())
    except Exception:  # noqa: BLE001
        pass
    return {}


def get_flags() -> dict:
    """Effective flags: runtime override file on top of static config defaults."""
    flags = {"face_blur": settings.enable_face_blur}
    override = _read()
    for k in flags:
        if k in override and override[k] is not None:
            flags[k] = bool(override[k])
    return flags


def set_flags(**kw) -> dict:
    current = _read()
    for k, v in kw.items():
        if v is not None:
            current[k] = bool(v)
    RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_PATH.write_text(json.dumps(current, indent=2))
    return get_flags()


def face_blur_enabled() -> bool:
    return get_flags()["face_blur"]
