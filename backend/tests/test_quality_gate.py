"""Quality-gate tests: rotation decision logic (pure) + a synthetic-clip probe
(structure and no-hands warning; real-hand rotation detection is validated on
real footage via scripts/process_clip.py).

Run from backend/:  python tests/test_quality_gate.py
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.quality_gate import decide_rotation, probe_clip, tally_votes  # noqa: E402


def test_tally_votes():
    # Unanimous (>=2) decides — including "confidently upright".
    assert tally_votes([180, 180, 180]) == (180, True)
    assert tally_votes([0, 0, 0]) == (0, True)
    # Majority suggests but does NOT decide (a wrong auto-rotation corrupts
    # every downstream coordinate).
    assert tally_votes([180, 180, 0]) == (180, False)
    # Single vote or none: never decide.
    assert tally_votes([90]) == (90, False)
    assert tally_votes([]) == (0, False)
    print("ok: VLM vote tallying")


def test_decide_rotation():
    # Clear winner -> decided.
    rot, ok = decide_rotation({0: 0.2, 90: 0.1, 180: 6.4, 270: 0.3})
    assert (rot, ok) == (180, True)
    # Upright winning is also a decision (extra rotation 0).
    rot, ok = decide_rotation({0: 5.0, 90: 0.4, 180: 1.1, 270: 0.2})
    assert (rot, ok) == (0, True)
    # Ambiguous (close scores) -> not decided.
    rot, ok = decide_rotation({0: 2.0, 90: 0.0, 180: 2.4, 270: 0.1})
    assert ok is False
    # Weak evidence (below min score) -> not decided.
    rot, ok = decide_rotation({0: 0.0, 90: 0.0, 180: 0.9, 270: 0.0})
    assert ok is False
    # No evidence -> rotation 0, not decided.
    rot, ok = decide_rotation({0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0})
    assert (rot, ok) == (0, False)
    print("ok: rotation decision logic")


def _make_clip(path: Path, seconds=2, fps=10, luma=170):
    w, h = 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)
    for _ in range(seconds * fps):
        frame = np.full((h, w, 3), luma, dtype=np.uint8)
        # texture so blur metric is sane
        noise = rng.integers(0, 60, size=(h, w, 1), dtype=np.uint8)
        frame = cv2.add(frame, np.repeat(noise, 3, axis=2))
        writer.write(frame)
    writer.release()


def test_probe_clip_structure_and_warnings():
    tmp = Path(tempfile.mkdtemp())
    clip = tmp / "clip.mp4"
    _make_clip(clip)
    rep = probe_clip(clip, sample_frames=4, use_vlm=False)
    assert rep.frames_sampled >= 3
    assert set(rep.hand_scores_by_rotation) == {0, 90, 180, 270}
    assert rep.vlm_votes == []
    assert rep.rotation_decided is False, "no VLM votes -> must not decide a rotation"
    assert rep.suggested_rotation == 0
    assert any("no hands" in w for w in rep.warnings)
    assert rep.mean_luma > 100  # bright synthetic clip
    d = rep.as_dict()
    assert "warnings" in d and "hand_scores_by_rotation" in d and "vlm_votes" in d
    print(f"ok: probe report (luma={rep.mean_luma}, warnings={len(rep.warnings)})")


def test_probe_dark_clip_warns():
    tmp = Path(tempfile.mkdtemp())
    clip = tmp / "dark.mp4"
    _make_clip(clip, luma=10)
    rep = probe_clip(clip, sample_frames=3, use_vlm=False)
    assert any("dark" in w for w in rep.warnings), rep.warnings
    print("ok: dark-clip warning")


if __name__ == "__main__":
    test_decide_rotation()
    test_tally_votes()
    test_probe_clip_structure_and_warnings()
    test_probe_dark_clip_warns()
    print("ALL TESTS PASSED")
