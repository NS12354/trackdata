"""Orientation auto-correction tests: rotation helper + metadata fallback when no
faces are present (the content detector should not fire on a faceless clip).

Run from backend/:  python tests/test_orientation.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.orientation import _rotate, detect_orientation_by_faces, resolve_video_meta  # noqa: E402


def test_rotate_dims():
    f = np.zeros((100, 200, 3), dtype=np.uint8)  # h=100, w=200
    assert _rotate(f, 0).shape[:2] == (100, 200)
    assert _rotate(f, 180).shape[:2] == (100, 200)       # 180 keeps dims
    assert _rotate(f, 90).shape[:2] == (200, 100)        # quarter-turn swaps
    assert _rotate(f, 270).shape[:2] == (200, 100)


def _make_faceless_clip(path: Path, seconds=2, fps=15):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"testsrc=duration={seconds}:size=320x240:rate={fps}",
                    "-pix_fmt", "yuv420p", str(path)], check=True)


def test_no_faces_falls_back_to_metadata():
    tmp = Path(tempfile.mkdtemp())
    clip = tmp / "noface.mp4"
    _make_faceless_clip(clip)
    # No faces -> content detector declines (returns None).
    assert detect_orientation_by_faces(clip) is None
    # resolve_video_meta then uses the metadata rotation (0 here) and real dims.
    meta = resolve_video_meta(clip)
    assert meta.rotation == 0
    assert meta.width == 320 and meta.height == 240
    print(f"ok: faceless clip -> rotation {meta.rotation}, dims {meta.width}x{meta.height}")


if __name__ == "__main__":
    test_rotate_dims()
    test_no_faces_falls_back_to_metadata()
    print("ALL TESTS PASSED")
