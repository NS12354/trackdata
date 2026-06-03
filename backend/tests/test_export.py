"""Phase 7 tests: export bundle contains all extracts + a well-formed manifest.

Run from backend/:  python tests/test_export.py
"""
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="revisent_ex_"))
os.environ.update(
    DATABASE_URL=f"sqlite:///{_TMP/'ex.db'}",
    DATA_DIR=str(_TMP / "data"),
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from db import init_db, session_scope  # noqa: E402
from models import Video, VideoStatus  # noqa: E402
from storage import get_storage  # noqa: E402
from pipeline.events import extract_events  # noqa: E402
from pipeline.export import build_export  # noqa: E402

init_db()
VID = "exp-1"


def _seed():
    st = get_storage()
    # Minimal anonymized video file.
    st.local_path(f"anonymized/{VID}.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42fakevideo")
    # Minimal hand_pose.parquet with the embedded metadata the manifest reads.
    table = pa.table({"frame_number": pa.array([0], type=pa.int32())})
    table = table.replace_schema_metadata({
        b"model": b"mediapipe_hands", b"landmark_count": b"21",
        b"sample_fps": b"10.0", b"coord_order": b"x,y,z",
    })
    hp = st.local_path(f"processed/{VID}/hand_pose.parquet")
    pq.write_table(table, hp)
    # segments.json
    st.local_path(f"processed/{VID}/segments.json").write_text(json.dumps({
        "video_id": VID, "provider": "ollama", "model": "moondream",
        "sample_fps": 1.0, "frames_classified": 3, "cost_usd": 0.0,
        "duration_seconds": 20,
        "segments": [
            {"start_time": 0, "end_time": 8, "task_label": "moving container",
             "confidence": 0.6, "description": "moving a bin", "duration_seconds": 8},
            {"start_time": 8, "end_time": 20, "task_label": "transit/walking",
             "confidence": 0.6, "description": "walking", "duration_seconds": 12},
        ],
    }))
    with session_scope() as s:
        s.add(Video(id=VID, original_filename="clip.mov", operator_id="op-1",
                    property_tag="Greystar-Maple", status=VideoStatus.processed,
                    duration_seconds=20, anonymization_method="union+...",
                    anonymization_coverage=1.0,
                    anonymization_meta=json.dumps({"fps": 30, "width": 1080, "height": 1920})))
    extract_events(VID)  # populate events from segments


def test_build_export_bundle():
    _seed()
    zip_path = build_export(VID)
    assert zip_path.exists() and zip_path.stat().st_size > 0

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        assert names == {
            "anonymized.mp4", "hand_pose.parquet", "segments.json",
            "events.json", "manifest.json",
        }, names

        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["export_version"] == "1.0"
        assert manifest["video"]["id"] == VID
        assert manifest["video"]["property_tag"] == "Greystar-Maple"
        assert manifest["capture"]["width"] == 1080
        assert manifest["processing"]["hand_pose"]["model"] == "mediapipe_hands"
        assert manifest["processing"]["segmentation"]["provider"] == "ollama"
        assert manifest["processing"]["segmentation"]["cost_usd"] == 0.0
        assert manifest["processing"]["events"]["service_event_count"] == 1
        assert "consent" in manifest and "Forge" in manifest["future_work"]
        assert len(manifest["contents"]) == 5

        events = json.loads(zf.read("events.json"))
        assert events["event_count"] == 2
        assert events["service_event_count"] == 1

    print(f"ok: bundle {zip_path.name} ({zip_path.stat().st_size} bytes) with all 5 entries")


if __name__ == "__main__":
    test_build_export_bundle()
    print("ALL TESTS PASSED")
