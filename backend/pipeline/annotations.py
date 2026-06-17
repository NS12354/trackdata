"""Human corrections for segments (labels, descriptions, boundaries) with a
full edit log — the labeling loop a data product runs on.

Corrections persist directly into processed/<vid>/segments.json (the single
source of truth every consumer reads: dashboard, chat, events, exports).
Edited segments carry ``human_verified: true`` and confidence 1.0 — verified
subsets are worth more to buyers and feed the frozen eval sets.

Every change is appended to data/annotations/edits.jsonl:
  {ts, video_id, op, segment_index, field, before, after, editor}
That log is the audit trail ("human-verified" must be provable), the eval-set
generator, and future model-improvement pairs (prediction -> correction).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from storage import get_storage


def _segments_path(video_id: str) -> Path:
    return get_storage().local_path(f"processed/{video_id}/segments.json")


def _edits_log_path() -> Path:
    p = get_storage().local_path("annotations/edits.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def validate_segments(segments: List[dict], duration: Optional[float]) -> List[str]:
    """Structural validation; returns problems (empty == OK)."""
    problems = []
    prev_end = 0.0
    for i, s in enumerate(segments):
        try:
            st, en = float(s["start_time"]), float(s["end_time"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"segment {i}: missing/invalid start_time/end_time")
            continue
        if en <= st:
            problems.append(f"segment {i}: end <= start ({st}..{en})")
        if st < prev_end - 0.051:  # tolerate float jitter
            problems.append(f"segment {i}: overlaps previous (starts {st} < {prev_end})")
        prev_end = max(prev_end, en)
        if not str(s.get("task_label", "")).strip():
            problems.append(f"segment {i}: empty task_label")
    if duration and prev_end > duration + 1.0:
        problems.append(f"segments extend past clip duration ({prev_end} > {duration})")
    return problems


def apply_corrections(video_id: str, segments: List[dict],
                      edits: Optional[List[dict]] = None,
                      editor: str = "dashboard") -> dict:
    """Persist corrected segments, log the edits, return the updated doc.

    ``segments`` is the FULL corrected list (the UI sends the whole timeline);
    each entry: start_time, end_time, task_label, confidence?, description?,
    human_verified?. ``edits`` are display-level change records from the UI —
    logged verbatim so the audit trail says what the human actually did.
    """
    path = _segments_path(video_id)
    if not path.exists():
        raise FileNotFoundError(f"segments.json missing for {video_id}")
    doc = json.loads(path.read_text())

    problems = validate_segments(segments, doc.get("duration_seconds"))
    if problems:
        raise ValueError("; ".join(problems))

    now = time.time()
    out_segments = []
    for s in segments:
        seg = {
            "start_time": round(float(s["start_time"]), 3),
            "end_time": round(float(s["end_time"]), 3),
            "task_label": str(s["task_label"]).strip(),
            "confidence": 1.0 if s.get("human_verified") else float(s.get("confidence", 0.0)),
            "description": str(s.get("description", "")).strip(),
            "duration_seconds": round(float(s["end_time"]) - float(s["start_time"]), 3),
        }
        if s.get("human_verified"):
            seg["human_verified"] = True
        out_segments.append(seg)

    doc["segments"] = out_segments
    doc["human_edited_at"] = now
    doc["human_verified_count"] = sum(1 for s in out_segments if s.get("human_verified"))
    path.write_text(json.dumps(doc, indent=2))

    with _edits_log_path().open("a", encoding="utf-8") as f:
        for e in (edits or [{"op": "bulk_update", "note": "full timeline replace"}]):
            f.write(json.dumps({"ts": now, "video_id": video_id,
                                "editor": editor, **e}) + "\n")

    # Corrected segments change the derived metrics — rebuild them.
    from .events import extract_events
    extract_events(video_id)
    return doc
