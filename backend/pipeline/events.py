"""Event extraction & operational metrics (Phase 4).

Derives operational events from the task segments (Phase 3) and hand-pose
(Phase 2), and writes them to the ``events`` table — the table the dashboard and
(future) chatbot query. Pure local computation; no model calls, no cost.

Event taxonomy → ``type``:
  service        — container/lock/gate/load manipulation (the billable work)
  contamination  — handling overflow/contamination (a flag)
  idle           — short idle/waiting
  downtime       — idle/waiting >= idle_downtime_seconds
  transit        — walking/approaching between work
  task           — anything else

A per-segment Event row carries (type, label, start/end, duration, property_tag,
description). Aggregates (time-per-task, downtime totals, per-property) are
computed by querying these rows — see ``summarize_video`` / ``operator_overview``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from sqlalchemy import select, func

from config import settings
from db import session_scope
from models import Event, Video
from storage import get_storage
from .hand_pose import load_hand_pose

log = logging.getLogger("revisent.events")

# Map a taxonomy task label to an event type.
SERVICE_LABELS = {
    "moving container",
    "opening gate/enclosure",
    "manipulating lock or latch",
    "loading/unloading",
}
_TYPE_BY_LABEL = {
    "handling overflow/contamination": "contamination",
    "idle/waiting": "idle",  # may be promoted to "downtime" by duration
    "transit/walking": "transit",
    "approaching property": "transit",
}


def _event_type(label: str, duration: float) -> str:
    if label in SERVICE_LABELS:
        return "service"
    base = _TYPE_BY_LABEL.get(label, "task")
    if base == "idle" and duration >= settings.idle_downtime_seconds:
        return "downtime"
    return base


def _segments_path(video_id: str) -> Path:
    return get_storage().local_path(f"processed/{video_id}/segments.json")


def _hand_pose_path(video_id: str) -> Path:
    return get_storage().local_path(f"processed/{video_id}/hand_pose.parquet")


def _active_hand_seconds(video_id: str) -> Optional[float]:
    """Seconds of the clip with at least one detected hand (pose-derived signal)."""
    path = _hand_pose_path(video_id)
    if not path.exists():
        return None
    rows = load_hand_pose(path)
    if not rows or "left_hand_landmarks" not in rows[0]:
        return 0.0
    with_hand = sum(1 for r in rows if r.get("left_hand_landmarks") or r.get("right_hand_landmarks"))
    # Each sampled frame represents roughly one sampling interval.
    if len(rows) >= 2 and "timestamp_ms" in rows[0]:
        interval = (rows[-1]["timestamp_ms"] - rows[0]["timestamp_ms"]) / 1000.0 / (len(rows) - 1)
    else:
        interval = 0.1
    return round(with_hand * interval, 2)


def extract_events(video_id: str) -> dict:
    """(Re)build the events for a video from its segments. Idempotent."""
    seg_path = _segments_path(video_id)
    if not seg_path.exists():
        raise FileNotFoundError(f"segments.json missing for {video_id}")
    data = json.loads(seg_path.read_text())
    segments = data.get("segments", [])

    with session_scope() as s:
        video = s.get(Video, video_id)
        property_tag = video.property_tag if video else None
        # Idempotent: clear prior events for this video before re-inserting.
        s.query(Event).filter(Event.video_id == video_id).delete()
        for seg in segments:
            label = seg["task_label"]
            dur = float(seg.get("duration_seconds", seg["end_time"] - seg["start_time"]))
            s.add(Event(
                video_id=video_id,
                type=_event_type(label, dur),
                label=label,
                start_time=float(seg["start_time"]),
                end_time=float(seg["end_time"]),
                duration_seconds=round(dur, 3),
                property_tag=property_tag,
                description=seg.get("description", ""),
            ))

    summary = summarize_video(video_id)
    log.info(
        "events %s: %d events, service=%d downtime=%.0fs contamination=%d",
        video_id, summary["event_count"], summary["service_event_count"],
        summary["downtime_seconds"], summary["contamination_event_count"],
    )
    return summary


def summarize_video(video_id: str) -> dict:
    """Per-video operational summary computed from the events table."""
    with session_scope() as s:
        rows = s.execute(
            select(Event).where(Event.video_id == video_id).order_by(Event.start_time)
        ).scalars().all()
        events = [e.to_dict() for e in rows]

    time_per_task: dict = {}
    time_per_type: dict = {}
    for e in events:
        time_per_task[e["label"]] = round(time_per_task.get(e["label"], 0) + e["duration_seconds"], 2)
        time_per_type[e["type"]] = round(time_per_type.get(e["type"], 0) + e["duration_seconds"], 2)

    return {
        "video_id": video_id,
        "event_count": len(events),
        "total_event_seconds": round(sum(e["duration_seconds"] for e in events), 2),
        "time_per_task": time_per_task,
        "time_per_type": time_per_type,
        "service_event_count": sum(1 for e in events if e["type"] == "service"),
        "idle_seconds": round(sum(e["duration_seconds"] for e in events if e["type"] in ("idle", "downtime")), 2),
        "downtime_seconds": round(sum(e["duration_seconds"] for e in events if e["type"] == "downtime"), 2),
        "downtime_event_count": sum(1 for e in events if e["type"] == "downtime"),
        "contamination_event_count": sum(1 for e in events if e["type"] == "contamination"),
        "active_hand_seconds": _active_hand_seconds(video_id),
        "events": events,
    }


def operator_overview(operator_id: Optional[str] = None) -> dict:
    """Operator-wide rollup across all processed videos (dashboard home page)."""
    with session_scope() as s:
        vq = select(Video)
        if operator_id:
            vq = vq.where(Video.operator_id == operator_id)
        videos = s.execute(vq).scalars().all()
        video_ids = [v.id for v in videos]
        total_seconds = sum(v.duration_seconds or 0 for v in videos)

        eq = select(Event)
        if video_ids:
            eq = eq.where(Event.video_id.in_(video_ids))
        events = s.execute(eq).scalars().all()

        per_property: dict = {}
        service = downtime_s = contamination = 0
        for e in events:
            tag = e.property_tag or "(untagged)"
            p = per_property.setdefault(tag, {"service_events": 0, "downtime_seconds": 0.0,
                                              "contamination_flags": 0, "total_seconds": 0.0})
            p["total_seconds"] = round(p["total_seconds"] + e.duration_seconds, 2)
            if e.type == "service":
                service += 1; p["service_events"] += 1
            elif e.type == "downtime":
                downtime_s += e.duration_seconds; p["downtime_seconds"] = round(p["downtime_seconds"] + e.duration_seconds, 2)
            elif e.type == "contamination":
                contamination += 1; p["contamination_flags"] += 1

    return {
        "operator_id": operator_id or "all",
        "video_count": len(videos),
        "total_hours": round(total_seconds / 3600.0, 3),
        "total_service_events": service,
        "total_downtime_seconds": round(downtime_s, 2),
        "total_contamination_flags": contamination,
        "per_property": per_property,
    }
