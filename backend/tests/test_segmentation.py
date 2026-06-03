"""Phase 3 tests: taxonomy mapping, aggregation, short-segment merge, and the
end-to-end segment_video flow with a mocked VLM provider (no Ollama needed).

Run from backend/:  python tests/test_segmentation.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.segmentation_providers import _match_taxonomy, _parse_label, FrameLabel  # noqa: E402
from pipeline.segmentation import segment_video, _aggregate, _merge_short, Segment  # noqa: E402


def test_taxonomy_mapping():
    assert _match_taxonomy("idle/waiting") == "idle/waiting"
    assert _match_taxonomy("the worker is moving a container") == "moving container"
    assert _match_taxonomy("opening a gate") == "opening gate/enclosure"
    assert _match_taxonomy("manipulating a latch") == "manipulating lock or latch"
    assert _match_taxonomy("walking down the street") == "transit/walking"
    assert _match_taxonomy("completely unrelated text xyz") is None


def test_parse_json_label():
    lab = _parse_label('{"task": "loading/unloading", "confidence": 0.8, "description": "lifting a bin"}')
    assert lab.task == "loading/unloading"
    assert lab.confidence == 0.8
    assert "lifting" in lab.description


def test_aggregate_and_merge():
    L = lambda t: FrameLabel(task=t, confidence=0.6, description=f"{t} desc")
    # transit(0,1,2) idle(3) transit(4,5)  at 1s spacing
    per_frame = [(0.0, L("transit/walking")), (1.0, L("transit/walking")),
                 (2.0, L("transit/walking")), (3.0, L("idle/waiting")),
                 (4.0, L("transit/walking")), (5.0, L("transit/walking"))]
    segs = _aggregate(per_frame, duration=6.0, sample_interval=1.0)
    assert [s.task_label for s in segs] == ["transit/walking", "idle/waiting", "transit/walking"]
    # The 1s idle blip is below a 1.5s threshold -> merged away.
    merged = _merge_short([Segment(**{**s.__dict__}) for s in segs], min_seconds=1.5)
    assert all(s.duration >= 1.5 or len(merged) == 1 for s in merged)
    assert "idle/waiting" not in [s.task_label for s in merged] or len(merged) == 1


class _FakeProvider:
    name = "fake"
    model = "fake-vlm"

    def __init__(self, labels):
        self.labels = labels
        self.i = 0

    def classify(self, jpeg_bytes):
        lab = self.labels[min(self.i, len(self.labels) - 1)]
        self.i += 1
        return FrameLabel(task=lab, confidence=0.6, description=f"{lab}"), 0.0


def _make_clip(path, seconds=4, fps=30):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"testsrc=duration={seconds}:size=320x240:rate={fps}",
                    "-pix_fmt", "yuv420p", str(path)], check=True)


def test_segment_video_end_to_end_mocked():
    tmp = Path(tempfile.mkdtemp())
    clip = tmp / "clip.mp4"
    _make_clip(clip, seconds=4, fps=30)
    # 4s @ 1fps -> ~4 samples. Provider returns transit, transit, loading, loading.
    provider = _FakeProvider(["transit/walking", "transit/walking",
                              "loading/unloading", "loading/unloading"])
    result = segment_video(clip, "testvid", provider=provider, sample_fps=1.0)

    assert result.cost_usd == 0.0, "local/mock provider must be free"
    assert result.provider == "fake"
    assert result.frames_classified >= 3
    labels = [s.task_label for s in result.segments]
    assert "transit/walking" in labels and "loading/unloading" in labels
    # Times are monotonic and within the clip.
    for s in result.segments:
        assert 0 <= s.start_time <= s.end_time <= result.duration_seconds + 1
    print(f"ok: {result.frames_classified} frames -> {len(result.segments)} segments, "
          f"cost ${result.cost_usd}, labels={labels}")


if __name__ == "__main__":
    test_taxonomy_mapping()
    test_parse_json_label()
    test_aggregate_and_merge()
    test_segment_video_end_to_end_mocked()
    print("ALL TESTS PASSED")
