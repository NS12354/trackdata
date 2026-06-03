"""Phase 2 tests: hand pose extraction produces a loadable Parquet with the
expected schema and embedded sampling metadata.

Run from backend/:  python tests/test_hand_pose.py
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.hand_pose import (  # noqa: E402
    extract_hand_pose, load_hand_pose, read_hand_pose_metadata,
)


def _make_clip(path: Path, seconds=2, fps=30):
    """A synthetic clip with no real hands — extraction should still run and
    produce a well-formed file with ~0% hand coverage (honest)."""
    w, h = 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(seconds * fps):
        frame = np.full((h, w, 3), (i * 4 % 255), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_extract_and_load():
    tmp = Path(tempfile.mkdtemp())
    clip = tmp / "in.mp4"
    out = tmp / "hand_pose.parquet"
    _make_clip(clip, seconds=2, fps=30)

    result = extract_hand_pose(clip, out, video_id="testvid", sample_fps=10)

    assert out.exists() and out.stat().st_size > 0, "no parquet produced"
    # 2s @ 30fps sampled at 10fps -> stride 3 -> ~20 sampled frames
    assert result.sample_stride == 3, f"unexpected stride {result.sample_stride}"
    assert 15 <= result.frames_sampled <= 22, f"sampled {result.frames_sampled}"
    assert 0.0 <= result.coverage <= 1.0

    rows = load_hand_pose(out)
    assert len(rows) == result.frames_sampled
    row = rows[0]
    assert set(row) >= {
        "frame_number", "timestamp_ms",
        "left_hand_landmarks", "right_hand_landmarks",
        "left_confidence", "right_confidence",
    }
    # No hands in synthetic clip -> landmark fields are None.
    assert row["left_hand_landmarks"] is None
    assert row["right_hand_landmarks"] is None

    meta = read_hand_pose_metadata(out)
    assert meta["model"] == "mediapipe_hands"
    assert meta["video_id"] == "testvid"
    assert float(meta["sample_fps"]) > 0
    print(f"ok: sampled={result.frames_sampled} stride={result.sample_stride} "
          f"coverage={result.coverage:.0%} sample_fps={meta['sample_fps']}")


def test_landmark_shape_when_present():
    """If landmarks are present they must be 21x3. Validate the writer round-trips
    a synthetic 21x3 set correctly via the parquet schema."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pipeline.hand_pose import _LANDMARK_TYPE

    tmp = Path(tempfile.mkdtemp())
    out = tmp / "shape.parquet"
    fake = [[float(i), float(i) + 0.1, float(i) - 0.1] for i in range(21)]
    table = pa.table({"left_hand_landmarks": pa.array([fake, None], type=_LANDMARK_TYPE)})
    pq.write_table(table, out)
    rows = pq.read_table(out).to_pylist()
    assert rows[0]["left_hand_landmarks"] is not None
    assert len(rows[0]["left_hand_landmarks"]) == 21
    assert all(len(pt) == 3 for pt in rows[0]["left_hand_landmarks"])
    assert rows[1]["left_hand_landmarks"] is None
    print("ok: 21x3 landmark round-trip + null handling")


if __name__ == "__main__":
    test_extract_and_load()
    test_landmark_shape_when_present()
    print("ALL TESTS PASSED")
