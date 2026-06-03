"""Phase 4 tests: event typing + end-to-end extraction/summary from segments.

Run from backend/:  python tests/test_events.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="revisent_ev_"))
os.environ.update(
    DATABASE_URL=f"sqlite:///{_TMP/'ev.db'}",
    DATA_DIR=str(_TMP / "data"),
    IDLE_DOWNTIME_SECONDS="30",
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import init_db, session_scope  # noqa: E402
from models import Video, VideoStatus, Event  # noqa: E402
from storage import get_storage  # noqa: E402
from pipeline.events import _event_type, extract_events, summarize_video, operator_overview  # noqa: E402

init_db()


def test_event_typing():
    assert _event_type("moving container", 5) == "service"
    assert _event_type("loading/unloading", 5) == "service"
    assert _event_type("handling overflow/contamination", 5) == "contamination"
    assert _event_type("idle/waiting", 10) == "idle"
    assert _event_type("idle/waiting", 45) == "downtime"   # >= 30s threshold
    assert _event_type("transit/walking", 5) == "transit"
    assert _event_type("approaching property", 5) == "transit"


def _seed(video_id, property_tag, segments):
    with session_scope() as s:
        s.add(Video(id=video_id, original_filename="t.mp4", operator_id="op-1",
                    property_tag=property_tag, status=VideoStatus.processed,
                    duration_seconds=segments[-1]["end_time"]))
    out = get_storage().local_path(f"processed/{video_id}/segments.json")
    out.write_text(json.dumps({"segments": segments}))


def test_extract_and_summarize():
    segs = [
        {"start_time": 0, "end_time": 5, "task_label": "transit/walking",
         "confidence": 0.6, "description": "walking", "duration_seconds": 5},
        {"start_time": 5, "end_time": 20, "task_label": "moving container",
         "confidence": 0.7, "description": "moving a bin", "duration_seconds": 15},
        {"start_time": 20, "end_time": 60, "task_label": "idle/waiting",
         "confidence": 0.5, "description": "waiting", "duration_seconds": 40},
        {"start_time": 60, "end_time": 70, "task_label": "handling overflow/contamination",
         "confidence": 0.6, "description": "overflow", "duration_seconds": 10},
    ]
    _seed("vid-A", "Greystar-Maple", segs)
    summary = extract_events("vid-A")

    assert summary["event_count"] == 4
    assert summary["service_event_count"] == 1
    assert summary["downtime_seconds"] == 40   # the 40s idle -> downtime
    assert summary["downtime_event_count"] == 1
    assert summary["contamination_event_count"] == 1
    assert summary["time_per_task"]["moving container"] == 15

    # Idempotent: re-running doesn't duplicate.
    extract_events("vid-A")
    with session_scope() as s:
        assert s.query(Event).filter(Event.video_id == "vid-A").count() == 4

    # Per-property carried onto events.
    with session_scope() as s:
        ev = s.query(Event).filter(Event.video_id == "vid-A").first()
        assert ev.property_tag == "Greystar-Maple"


def test_operator_overview():
    _seed("vid-B", "Greystar-Maple", [
        {"start_time": 0, "end_time": 12, "task_label": "moving container",
         "confidence": 0.7, "description": "", "duration_seconds": 12}])
    extract_events("vid-B")
    ov = operator_overview()
    assert ov["video_count"] >= 2
    assert ov["total_service_events"] >= 2
    assert "Greystar-Maple" in ov["per_property"]
    assert ov["per_property"]["Greystar-Maple"]["service_events"] >= 2
    print(f"ok: overview videos={ov['video_count']} service={ov['total_service_events']} "
          f"downtime={ov['total_downtime_seconds']}s props={list(ov['per_property'])}")


if __name__ == "__main__":
    test_event_typing()
    test_extract_and_summarize()
    test_operator_overview()
    print("ALL TESTS PASSED")
