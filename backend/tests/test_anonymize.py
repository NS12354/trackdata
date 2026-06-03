"""Phase 1 tests: blur correctness + end-to-end anonymization on a synthetic clip.

Run from backend/:  python -m pytest tests/ -v   (or: python tests/test_anonymize.py)
"""
import sys
from pathlib import Path

import numpy as np

# Make backend/ importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.anonymize import _blur_box, _build_filled_boxes, anonymize_video  # noqa: E402
from pipeline.video_meta import probe  # noqa: E402


def test_gap_filling_bridges_dropouts():
    """A face detected at frames 0 and 4 but missed at 1-3 must still be blurred
    on those middle frames (interpolated), and held a couple frames after."""
    box = (10.0, 10.0, 50.0, 50.0)
    per_frame = [[(box, True)], [], [], [], [(box, True)]] + [[] for _ in range(5)]
    filled, stats = _build_filled_boxes(
        per_frame, frames_total=10, max_gap=8, hold=2, dilation=0.06, match_iou=0.1,
    )
    assert stats["confirmed_tracks"] == 1, f"expected one confirmed track, got {stats}"
    for f in range(0, 5):
        assert filled[f], f"frame {f} should be blurred (detected or interpolated)"
    # tail hold of 2 frames after the last detection (frame 4)
    assert filled[5] and filled[6], "tail-hold frames should be blurred"
    assert not filled[9], "frames well past the track should not be blurred"


def test_large_gap_not_bridged():
    """A gap longer than max_gap is a genuine absence and must NOT be bridged."""
    box = (10.0, 10.0, 50.0, 50.0)
    per_frame = [[(box, True)]] + [[] for _ in range(20)] + [[(box, True)]]
    filled, _ = _build_filled_boxes(
        per_frame, frames_total=22, max_gap=5, hold=1, dilation=0.06, match_iou=0.1,
    )
    # Middle of the long gap must be clear.
    assert not filled[11], "long gap should not be bridged"


def test_weak_track_rejected_as_false_positive():
    """A track whose detections are never strong (e.g. a hand mis-detected below
    every detector's confirm threshold) must be dropped entirely — no blur."""
    box = (10.0, 10.0, 50.0, 50.0)
    per_frame = [[(box, False)], [(box, False)], [(box, False)]] + [[] for _ in range(5)]
    filled, stats = _build_filled_boxes(
        per_frame, frames_total=8, max_gap=8, hold=2, dilation=0.06, match_iou=0.1,
    )
    assert stats["confirmed_tracks"] == 0, "weak track should not be confirmed"
    assert stats["rejected_tracks"] == 1
    assert all(not b for b in filled), "no frame should be blurred for a rejected track"


def test_strong_frame_confirms_whole_track():
    """One strong detection rescues the track's weak frames (real face)."""
    box = (10.0, 10.0, 50.0, 50.0)
    per_frame = [[(box, False)], [(box, True)], [(box, False)]] + [[] for _ in range(5)]
    filled, stats = _build_filled_boxes(
        per_frame, frames_total=8, max_gap=8, hold=2, dilation=0.06, match_iou=0.1,
    )
    assert stats["confirmed_tracks"] == 1
    for f in range(0, 3):
        assert filled[f], f"frame {f} of a confirmed track should be blurred"


def test_blur_reduces_local_variance():
    """A blurred high-frequency region must lose detail (lower variance)."""
    rng = np.random.RandomState(0)
    frame = (rng.rand(200, 200, 3) * 255).astype(np.uint8)
    before = float(frame[50:150, 50:150].var())
    _blur_box(frame, (50, 50, 150, 150), strength=0.6)
    after = float(frame[50:150, 50:150].var())
    assert after < before * 0.6, f"blur did not reduce variance enough: {before:.1f} -> {after:.1f}"


def test_blur_box_out_of_bounds_is_safe():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    _blur_box(frame, (-10, -10, 5, 5), strength=0.6)    # off top-left
    _blur_box(frame, (95, 95, 200, 200), strength=0.6)  # off bottom-right
    _blur_box(frame, (50, 50, 40, 40), strength=0.6)    # inverted -> no-op


def _make_clip(path: Path, seconds=2, fps=15):
    import cv2
    w, h = 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(seconds * fps):
        frame = np.full((h, w, 3), (i * 5 % 255), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_anonymize_end_to_end(tmp_path=None):
    import tempfile
    tmp = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    src = tmp / "in.mp4"
    out = tmp / "out.mp4"
    _make_clip(src)

    result = anonymize_video(src, out)

    assert out.exists() and out.stat().st_size > 0, "no anonymized output produced"
    assert result.frames_total > 0, "no frames read"
    assert 0.0 <= result.coverage <= 1.0
    # Output should be re-readable.
    meta = probe(out)
    assert meta.frame_count > 0 or meta.duration_seconds > 0
    print(f"e2e ok: frames={result.frames_total} codec={result.output_codec} "
          f"coverage={result.coverage:.2%}")


if __name__ == "__main__":
    test_blur_reduces_local_variance()
    test_blur_box_out_of_bounds_is_safe()
    test_gap_filling_bridges_dropouts()
    test_large_gap_not_bridged()
    test_weak_track_rejected_as_false_positive()
    test_strong_frame_confirms_whole_track()
    test_anonymize_end_to_end()
    print("ALL TESTS PASSED")
