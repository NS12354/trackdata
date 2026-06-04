"""Phase 6 tests: chat context-building + answer flow with a mocked LLM (no Ollama).

Run from backend/:  python tests/test_chat.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="revisent_chat_"))
os.environ.update(
    DATABASE_URL=f"sqlite:///{_TMP/'c.db'}",
    DATA_DIR=str(_TMP / "data"),
    SEGMENTATION_MODE="open",
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db import init_db, session_scope  # noqa: E402
from models import Video, VideoStatus  # noqa: E402
from storage import get_storage  # noqa: E402
from pipeline.events import extract_events  # noqa: E402
import pipeline.chat as chatmod  # noqa: E402

init_db()
VID = "chat-1"


def _seed():
    get_storage().local_path(f"processed/{VID}/segments.json").write_text(json.dumps({
        "segments": [
            {"start_time": 0, "end_time": 5, "task_label": "walking hallway",
             "confidence": 0.6, "description": "The person is walking down a hallway carrying a box.",
             "duration_seconds": 5},
            {"start_time": 5, "end_time": 18, "task_label": "opening container",
             "confidence": 0.6, "description": "The person opens a large container and looks inside.",
             "duration_seconds": 13},
        ],
    }))
    with session_scope() as s:
        s.add(Video(id=VID, original_filename="route.mov", operator_id="op",
                    property_tag="Maple", status=VideoStatus.processed, duration_seconds=18))
    extract_events(VID)


def test_build_context_contains_timeline():
    _seed()
    ctx = chatmod.build_context(VID)
    assert "walking down a hallway" in ctx
    assert "opens a large container" in ctx
    assert "00:00" in ctx and "00:05" in ctx          # timestamps present
    assert "route.mov" in ctx                          # video metadata present


def test_answer_uses_context():
    captured = {}

    def fake_llm(question, context):
        captured["q"] = question
        captured["ctx"] = context
        return "The worker walked with a box, then opened a container."

    # Mock the LLM call so the test needs no Ollama.
    chatmod._ask_ollama = fake_llm  # type: ignore

    out = chatmod.answer_question("What happened?", VID)
    assert out["answer"].startswith("The worker walked")
    assert out["provider"] == "ollama"
    # The model was actually given the grounding commentary.
    assert "container" in captured["ctx"]
    assert "What happened?" == captured["q"]
    print(f"ok: answer grounded in context ({len(captured['ctx'])} chars)")


if __name__ == "__main__":
    test_build_context_contains_timeline()
    test_answer_uses_context()
    print("ALL TESTS PASSED")
