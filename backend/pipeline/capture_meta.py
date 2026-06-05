"""Capture metadata — what a research engineer must know to use the data correctly.

A dataset is only as usable as its provenance. This assembles the per-clip capture
record (camera, mount geometry, intrinsics, conditions, operator, consent) that
ships in every export manifest. Camera intrinsics are pulled from the calibration
record (scripts/calibrate_camera.py) — without them the data isn't spatially usable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from config import settings

BACKEND_DIR = Path(__file__).resolve().parent
INTRINSICS_PATH = BACKEND_DIR / "camera_intrinsics.json"


def load_intrinsics() -> Optional[dict]:
    if INTRINSICS_PATH.exists():
        try:
            return json.loads(INTRINSICS_PATH.read_text())
        except Exception:
            return None
    return None


def capture_record(vdict: dict, capture: dict) -> dict:
    """Build the capture-metadata block for a clip's export manifest.

    vdict: video.to_dict(); capture: {fps,width,height} from the anonymization meta.
    """
    intr = load_intrinsics()
    return {
        "camera": {
            "mount": "chest (shirt-clipped)",
            "mount_geometry_body_frame": {
                "height_frac_of_operator": settings.ego_chest_mount_height_frac,
                "forward_frac_of_operator": settings.ego_chest_mount_forward_frac,
                "pitch_deg_down": settings.ego_chest_mount_pitch_deg,
            },
            "intrinsics": intr or {
                "status": "NOT CALIBRATED — run scripts/calibrate_camera.py",
                "assumed_fov_deg": settings.ego_camera_fov_deg,
            },
            "intrinsics_calibrated": bool(intr),
        },
        "recording": {
            "resolution": {"width": capture.get("width"), "height": capture.get("height")},
            "fps": capture.get("fps"),
            "color": "RGB",
            "audio": "removed (privacy)",
        },
        "operator": {
            "anonymized_id": vdict.get("worker_id_anonymized") or vdict.get("operator_id"),
            "height_cm": vdict.get("operator_height_cm"),
        },
        "environment": {
            "scene": vdict.get("scene"),
            "property_tag": vdict.get("property_tag"),
            "indoor_outdoor": None,   # fill from capture context if known
            "lighting": None,
        },
        "consent": {
            "reference": None,
            "note": "Attach the consent record id covering this worker/route before "
                    "external sharing. Training/redistribution rights must be explicit.",
        },
    }


if __name__ == "__main__":
    print(json.dumps(capture_record(
        {"worker_id_anonymized": "w-001", "operator_height_cm": 178, "scene": "loading dock"},
        {"width": 1080, "height": 1920, "fps": 30.0},
    ), indent=2))
