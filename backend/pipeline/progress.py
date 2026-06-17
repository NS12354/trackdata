"""Per-clip live progress reporting — a sidecar file, not a DB migration.

Stages call ``report()`` as they work; the API serves the file; the dashboard
polls it. Best-effort by design: progress must never break processing, so all
errors are swallowed.

processed/<vid>/progress.json:
  {"stage": "annotating activities", "pct": 44, "detail": "4/9 'pick up the casing'", "ts": ...}
"""
from __future__ import annotations

import json
import time
from typing import Optional


def _path(video_id: str):
    from storage import get_storage
    p = get_storage().local_path(f"processed/{video_id}/progress.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def report(video_id: str, stage: str, pct: Optional[float] = None,
           detail: str = "") -> None:
    try:
        _path(video_id).write_text(json.dumps({
            "stage": stage,
            "pct": None if pct is None else round(float(pct), 1),
            "detail": str(detail)[:160],
            "ts": time.time(),
        }))
    except Exception:  # noqa: BLE001 - progress must never break processing
        pass


def read(video_id: str) -> dict:
    try:
        return json.loads(_path(video_id).read_text())
    except Exception:  # noqa: BLE001
        return {}
