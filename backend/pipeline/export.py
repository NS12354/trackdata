"""Structured export bundle (Phase 7) — the lab-facing piece.

Packages a processed video's extracts into a single .zip under data/exports/:

    {video_id}.zip
      ├── anonymized.mp4      face-blurred H.264 video (audio removed)
      ├── hand_pose.parquet   per-frame 21-point hand keypoints (normalized)
      ├── segments.json       temporal task segments
      ├── events.json         derived operational events + summary
      └── manifest.json       contents, formats, capture metadata, consent ref

Format is JSON/Parquet for v1. LeRobot / RLDS / HDF5 conversion is planned future
work (via a tool like Forge) and is intentionally NOT done here.
"""
from __future__ import annotations

import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import settings
from db import session_scope
from models import Video
from storage import get_storage
from .hand_pose import read_hand_pose_metadata
from .events import summarize_video
from .video_meta import probe

log = logging.getLogger("revisent.export")

EXPORT_VERSION = "1.0"
TASK_TAXONOMY = [
    "approaching property", "moving container", "opening gate/enclosure",
    "manipulating lock or latch", "handling overflow/contamination",
    "loading/unloading", "transit/walking", "idle/waiting",
]


def _anon_path(video_id: str) -> Path:
    return get_storage().local_path(f"anonymized/{video_id}.mp4")


def _hand_pose_path(video_id: str) -> Path:
    return get_storage().local_path(f"processed/{video_id}/hand_pose.parquet")


def _segments_path(video_id: str) -> Path:
    return get_storage().local_path(f"processed/{video_id}/segments.json")


def _capture_meta(anon: Path, anon_meta: dict) -> dict:
    """Capture metadata from anonymization meta, falling back to probing the file."""
    fps = anon_meta.get("fps")
    width = anon_meta.get("width")
    height = anon_meta.get("height")
    if not (fps and width and height):
        try:
            m = probe(anon)
            fps, width, height = m.fps, m.width, m.height
        except Exception:  # noqa: BLE001
            pass
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "note": "Display-oriented (rotation already applied); audio removed.",
    }


def build_export(video_id: str) -> Path:
    """Build (or rebuild) the export zip for a video and return its path."""
    with session_scope() as s:
        video = s.get(Video, video_id)
        if video is None:
            raise FileNotFoundError(f"video {video_id} not found")
        vdict = video.to_dict()
        anon_meta = json.loads(video.anonymization_meta) if video.anonymization_meta else {}

    anon = _anon_path(video_id)
    if not anon.exists():
        raise FileNotFoundError(f"anonymized video missing for {video_id}; not exportable")

    hand_pose = _hand_pose_path(video_id)
    segments = _segments_path(video_id)
    seg_data = json.loads(segments.read_text()) if segments.exists() else None

    # Events: write the per-video summary (includes the event list) into the bundle.
    summary = summarize_video(video_id)

    # Provenance for the manifest.
    hp_meta = read_hand_pose_metadata(hand_pose) if hand_pose.exists() else None

    contents = [
        {"path": "anonymized.mp4", "type": "video/mp4",
         "description": "Face-blurred H.264 video (audio removed)."},
        {"path": "events.json", "type": "application/json",
         "description": "Derived operational events and per-video summary."},
        {"path": "manifest.json", "type": "application/json",
         "description": "This manifest."},
    ]
    if hand_pose.exists():
        contents.insert(1, {
            "path": "hand_pose.parquet", "type": "application/vnd.apache.parquet",
            "description": "Per-frame 21-point hand keypoints, normalized x,y,z.",
        })
    if seg_data is not None:
        contents.insert(-2, {
            "path": "segments.json", "type": "application/json",
            "description": "Temporal task segments with labels and descriptions.",
        })

    manifest = {
        "export_version": EXPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "video": {
            "id": vdict["id"],
            "original_filename": vdict["original_filename"],
            "operator_id": vdict["operator_id"],
            "worker_id_anonymized": vdict["worker_id_anonymized"],
            "property_tag": vdict["property_tag"],
            "uploaded_at": vdict["uploaded_at"],
            "duration_seconds": vdict["duration_seconds"],
            "status": vdict["status"],
        },
        "capture": _capture_meta(anon, anon_meta),
        "anonymization": {
            "method": vdict["anonymization_method"],
            "coverage": vdict["anonymization_coverage"],
            "details": anon_meta or None,
        },
        "processing": {
            "hand_pose": ({
                "model": hp_meta.get("model"),
                "landmark_count": int(hp_meta.get("landmark_count", "21")),
                "sample_fps": hp_meta.get("sample_fps"),
                "coord_order": hp_meta.get("coord_order"),
            } if hp_meta else None),
            "segmentation": ({
                "provider": seg_data.get("provider"),
                "model": seg_data.get("model"),
                "sample_fps": seg_data.get("sample_fps"),
                "frames_classified": seg_data.get("frames_classified"),
                "cost_usd": seg_data.get("cost_usd"),
                "taxonomy": TASK_TAXONOMY,
            } if seg_data else None),
            "events": {
                "count": summary["event_count"],
                "time_per_type": summary["time_per_type"],
                "service_event_count": summary["service_event_count"],
                "downtime_seconds": summary["downtime_seconds"],
                "contamination_event_count": summary["contamination_event_count"],
            },
        },
        "consent": {
            "reference": None,
            "note": "Operator-managed. Attach the consent record id for this "
                    "worker/route here before sharing externally.",
        },
        "contents": contents,
        "future_work": "LeRobot / RLDS / HDF5 conversion is planned (e.g. via Forge) "
                       "and is not included in this v1 export.",
    }

    out = get_storage().local_path(f"exports/{video_id}.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(anon, "anonymized.mp4")
        if hand_pose.exists():
            zf.write(hand_pose, "hand_pose.parquet")
        if seg_data is not None:
            zf.writestr("segments.json", json.dumps(seg_data, indent=2))
        zf.writestr("events.json", json.dumps(summary, indent=2))
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    log.info("built export %s (%d bytes, %d files)", video_id, out.stat().st_size, len(contents))
    return out
