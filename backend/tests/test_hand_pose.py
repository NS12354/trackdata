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
        "left_world_landmarks", "right_world_landmarks",
        "left_confidence", "right_confidence",
    }
    # No hands in synthetic clip -> landmark fields are None.
    assert row["left_hand_landmarks"] is None
    assert row["right_hand_landmarks"] is None
    assert row["left_world_landmarks"] is None
    assert row["right_world_landmarks"] is None

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


def test_handedness_continuity_fix():
    """A lone hand that flips L<->R against spatial continuity gets relabeled;
    a genuinely distinct left hand far away is left alone."""
    from pipeline.hand_pose import _fix_handedness_flips

    def hand(x, y):
        return [[x, y, 0.0]]  # only the wrist landmark is consulted

    times = [i * 100.0 for i in range(5)]
    left =  [None,          None,          hand(0.51, 0.5), None,          hand(0.10, 0.5)]
    right = [hand(0.5, 0.5), hand(0.5, 0.5), None,           hand(0.5, 0.5), None]
    lw = [None, None, hand(0.51, 0.5), None, hand(0.10, 0.5)]
    rw = [hand(0.5, 0.5), hand(0.5, 0.5), None, hand(0.5, 0.5), None]
    lc = [None, None, 0.9, None, 0.9]
    rc = [0.9, 0.9, None, 0.9, None]

    fixes = _fix_handedness_flips(left, right, lw, rw, lc, rc, times)
    assert fixes == 1, f"expected 1 fix, got {fixes}"
    assert right[2] is not None and left[2] is None, "misflagged frame must move to right"
    assert rc[2] == 0.9 and lc[2] is None
    assert left[4] is not None, "genuine far-away left hand must stay left"
    print("ok: handedness continuity (1 flip fixed, real left preserved)")


def test_crossover_swap_and_teleport_fixes():
    """When hands cross, the classifier swaps L/R; continuity assignment swaps
    them back. Surviving teleport-sized jumps get nulled (physics gate)."""
    from pipeline.hand_pose import _fix_two_hand_swaps, _reject_teleports

    def hand(x, y):
        return [[x, y, 0.0]]

    times = [i * 33.3 for i in range(6)]
    # True tracks: L moves .30->.55, R moves .70->.45 (crossing). At frames 3-5
    # the detector's labels are SWAPPED (post-cross misclassification).
    Lx = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    Rx = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45]
    left =  [hand(Lx[i], 0.5) for i in range(3)] + [hand(Rx[i], 0.5) for i in range(3, 6)]
    right = [hand(Rx[i], 0.5) for i in range(3)] + [hand(Lx[i], 0.5) for i in range(3, 6)]
    lw = [None] * 6; rw = [None] * 6
    lc = [0.9] * 6; rc = [0.9] * 6
    swaps = _fix_two_hand_swaps(left, right, lw, rw, lc, rc, times)
    assert swaps >= 1, "crossover swap must be detected"
    # After fixing, each side's track is smooth (no >0.1 jumps).
    for seq, xs in ((left, Lx), (right, Rx)):
        for i in range(6):
            assert abs(seq[i][0][0] - xs[i]) < 1e-6, f"track not restored at {i}"

    # Teleport gate: a 0.5-jump in one 33ms frame gets nulled.
    seq = [hand(0.3, 0.5), hand(0.31, 0.5), hand(0.85, 0.5), hand(0.32, 0.5)]
    world = [None] * 4; conf = [0.9] * 4
    rej = _reject_teleports(seq, world, conf, [i * 33.3 for i in range(4)])
    assert rej == 1 and seq[2] is None and conf[2] is None
    assert seq[3] is not None, "recovery frame near the track must survive"
    print(f"ok: crossover swaps fixed ({swaps}), teleport rejected ({rej})")


def test_kalman_smoother():
    """Kalman: noisy stationary hand converges near truth; moving hand tracks
    without gross lag; dropouts reset cleanly."""
    from pipeline.hand_pose import _smooth_kalman
    rng = np.random.default_rng(0)
    truth = [[0.5, 0.5, 0.0]] * 21
    times = [i * 33.3 for i in range(30)]
    noisy = [(np.array(truth) + rng.normal(0, 0.01, (21, 3))).tolist() for _ in times]
    sm = _smooth_kalman(noisy, times)
    err_raw = np.mean([np.abs(np.array(f) - truth).mean() for f in noisy[10:]])
    err_sm = np.mean([np.abs(np.array(f) - truth).mean() for f in sm[10:]])
    assert err_sm < err_raw * 0.8, f"should reduce noise: {err_sm:.4f} vs {err_raw:.4f}"
    # Moving target: x advances 0.01/frame; a CV-Kalman has ~zero steady-state
    # ramp lag (it estimates velocity) — assert essentially none.
    moving = [(np.array(truth) + [i * 0.01, 0, 0]).tolist() for i in range(30)]
    smm = _smooth_kalman(moving, times)
    lag = abs(np.array(smm[-1])[0][0] - np.array(moving[-1])[0][0])
    assert lag < 0.003, f"kalman ramp lag should be ~0: {lag:.4f}"
    # Dropout reset.
    gappy = moving[:5] + [None] * 3 + moving[8:]
    out = _smooth_kalman(gappy, times)
    assert out[5] is None and out[8] is not None
    print(f"ok: kalman (noise {err_raw:.4f}->{err_sm:.4f}, lag {lag:.4f})")


def test_wilor_frame_mapping():
    """WiLoR detections map onto our channels: normalized 2D, hand-centered
    metric world, absolute wrist_cam = kp3d[0] + cam_t."""
    from pipeline.hand_pose import _wilor_frame_to_sides
    kp2d = [[100.0 + i, 200.0 + i] for i in range(21)]
    kp3d = [[0.01 * i, 0.0, 0.4] for i in range(21)]
    hands = [{"is_right": 1.0, "kp2d": kp2d, "kp3d": kp3d,
              "cam_t": [0.05, -0.02, 0.5], "focal": 600.0}]
    L, R, LW, RW, lc, rc, lcam, rcam = _wilor_frame_to_sides(hands, 1000, 2000)
    assert L is None and R is not None
    assert abs(R[0][0] - 0.1) < 1e-6 and abs(R[0][1] - 0.1) < 1e-6  # 100/1000, 200/2000
    assert abs(np.mean([p[0] for p in RW])) < 1e-6, "world must be centered"
    assert abs(rcam[0] - 0.05) < 1e-4 and abs(rcam[2] - 0.9) < 1e-4  # kp3d[0]+cam_t
    assert rc == 0.95 and lcam is None
    # Focal-convention rescale: WiLoR focal 600 vs true 1200 -> translation x2.
    *_, rcam2 = _wilor_frame_to_sides(hands, 1000, 2000, f_true=1200.0)
    assert abs(rcam2[2] - 1.8) < 1e-3 and abs(rcam2[0] - 0.1) < 1e-3
    print("ok: wilor frame mapping (2D norm, centered world, focal-rescaled wrist_cam)")


if __name__ == "__main__":
    test_extract_and_load()
    test_landmark_shape_when_present()
    test_handedness_continuity_fix()
    test_crossover_swap_and_teleport_fixes()
    test_kalman_smoother()
    test_wilor_frame_mapping()
    print("ALL TESTS PASSED")
