"""Integration tests: API + Celery (eager) + DB, end-to-end on a synthetic clip.

Runs the real app via FastAPI TestClient against a temp SQLite DB, an isolated
data dir, and Celery in eager mode (tasks run inline) — so an upload flows all
the way through anonymize -> hand-pose and the DB/files reflect it.

Run from backend/:  python -m pytest tests/test_integration.py -v
                or: python tests/test_integration.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Configure the environment BEFORE importing the app, so settings pick it up.
_TMP = Path(tempfile.mkdtemp(prefix="revisent_it_"))
os.environ.update(
    DATABASE_URL=f"sqlite:///{_TMP/'it.db'}",
    DATA_DIR=str(_TMP / "data"),
    CELERY_TASK_ALWAYS_EAGER="true",
    ENVIRONMENT="development",
    API_KEY="test-secret-key",          # exercise auth
    VALIDATE_UPLOAD_CONTENT="true",
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402
from db import init_db  # noqa: E402

# Create schema up front (TestClient without a context manager does not fire the
# app's startup event).
init_db()

# Make segmentation deterministic and Ollama-independent for CI: swap the VLM
# provider for a fake. The real Ollama path is exercised manually / in dev.
import pipeline.segmentation as _seg  # noqa: E402
from pipeline.segmentation_providers import FrameLabel  # noqa: E402


class _FakeSegProvider:
    name = "fake"
    model = "fake-vlm"

    def classify(self, jpeg_bytes):
        return FrameLabel(task="transit/walking", confidence=0.6, description="walking"), 0.0


_seg.get_segmentation_provider = lambda: _FakeSegProvider()

client = TestClient(main.app)
AUTH = {"X-API-Key": "test-secret-key"}


def _make_clip(path: Path, seconds=2, fps=15):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"testsrc=duration={seconds}:size=320x240:rate={fps}",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def test_health_and_ready():
    assert client.get("/health").json()["status"] == "ok"
    r = client.get("/ready")
    assert r.json()["database"] is True


def test_auth_required():
    # No key -> 401
    assert client.get("/api/videos").status_code == 401
    # With key -> 200
    assert client.get("/api/videos", headers=AUTH).status_code == 200


def test_rejects_non_video_content():
    # .mp4 extension but text content -> 400 (magic-byte sniff)
    files = {"file": ("fake.mp4", b"this is not a video", "video/mp4")}
    r = client.post("/api/videos", files=files, headers=AUTH)
    assert r.status_code == 400, r.text


def test_upload_processes_end_to_end():
    clip = _TMP / "clip.mp4"
    _make_clip(clip)
    with open(clip, "rb") as fh:
        r = client.post(
            "/api/videos",
            files={"file": ("clip.mp4", fh, "video/mp4")},
            data={"property_tag": "Greystar-Test"},
            headers=AUTH,
        )
    assert r.status_code == 201, r.text
    vid = r.json()["id"]

    # Eager Celery ran anonymize -> hand-pose -> segmentation inline.
    got = client.get(f"/api/videos/{vid}", headers=AUTH).json()
    assert got["status"] == "processed", got
    assert got["hand_pose_extracted"] is True
    assert got["segmented"] is True
    assert got["segmentation_cost_usd"] == 0.0   # local/fake provider is free
    assert got["property_tag"] == "Greystar-Test"

    # Anonymized stream serves with range support.
    s = client.get(f"/api/videos/{vid}/anonymized", headers={**AUTH, "Range": "bytes=0-99"})
    assert s.status_code == 206
    assert s.headers["content-range"].endswith(f"/{int(s.headers['content-range'].split('/')[1])}")

    # Hand-pose JSON is available with sampling metadata.
    hp = client.get(f"/api/videos/{vid}/hand-pose", headers=AUTH).json()
    assert hp["metadata"]["model"] == "mediapipe_hands"
    assert isinstance(hp["frames"], list) and len(hp["frames"]) > 0

    # Segments are available with $0 cost and the task taxonomy.
    seg = client.get(f"/api/videos/{vid}/segments", headers=AUTH).json()
    assert seg["cost_usd"] == 0.0
    assert isinstance(seg["segments"], list) and len(seg["segments"]) > 0
    assert seg["segments"][0]["task_label"] == "transit/walking"

    # Events derived from segments + per-video summary.
    ev = client.get(f"/api/videos/{vid}/events", headers=AUTH).json()
    assert ev["event_count"] > 0
    assert "transit" in ev["time_per_type"]

    # Operator-wide overview rolls up.
    ov = client.get("/api/metrics/overview", headers=AUTH).json()
    assert ov["video_count"] >= 1 and "per_property" in ov


if __name__ == "__main__":
    test_health_and_ready()
    test_auth_required()
    test_rejects_non_video_content()
    test_upload_processes_end_to_end()
    print("ALL INTEGRATION TESTS PASSED")
