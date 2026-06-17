"""Tests for signal-fusion boundary detection (Phase 3, fused mode).

Pure numeric helpers are tested directly (no video needed). Two integration
tests synthesize a clip with a mid-point scene change and exercise
``detect_boundaries`` + the fused ``segment_video`` path; both skip cleanly if
ffmpeg is unavailable.

Run from backend/:  python tests/test_boundary.py
"""
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.boundary import (  # noqa: E402
    _robust_norm, _resample_to_grid, _novelty_score, _pick_peaks,
    _quaternion_angle, hand_signals, ego_signals, detect_boundaries,
    _local_zscore, _signal_candidates, _nms_candidates,
)
from pipeline.segmentation_providers import FrameLabel  # noqa: E402


def test_robust_norm():
    out = _robust_norm(np.array([0.0, 1, 2, 3, 4, 100]))
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(_robust_norm(np.zeros(5)) == 0.0)  # degenerate -> zeros


def test_resample_to_grid():
    g = np.array([0.0, 1.0, 2.0])
    assert np.allclose(_resample_to_grid([0.0, 2.0], [0.0, 2.0], g), [0.0, 1.0, 2.0])
    assert np.all(_resample_to_grid([], [], g) == 0.0)  # empty -> zeros


def test_novelty_score_peaks_at_step():
    T = 20
    f = np.zeros((T, 1))
    f[10:] = 1.0
    scores = _novelty_score(f, half_window=3)
    assert 9 <= int(scores.argmax()) <= 11, f"peak at {scores.argmax()}, expected ~10"


def test_pick_peaks_respects_spacing():
    grid = np.arange(0.0, 10.0, 0.25)  # 0.25 s grid
    scores = np.zeros(len(grid))
    scores[10] = 1.0   # t=2.5  (strongest)
    scores[12] = 0.9   # t=3.0  (within 2 s of 2.5 -> suppressed)
    scores[20] = 0.8   # t=5.0  (kept)
    peaks = _pick_peaks(scores, grid, min_spacing_s=2.0, threshold=0.5)
    assert set(peaks) == {10, 20}


def test_local_zscore_flags_local_outlier():
    # A modest local bump and a huge one far away: BOTH should stand out locally,
    # because each is judged against its own neighborhood (the late-fusion fix).
    x = np.zeros(40)
    x[10] = 0.5
    x[30] = 5.0
    z = _local_zscore(x, half_win=4)
    assert z[10] > 1.5, "subtle local peak must still score high locally"
    assert z[30] > 1.5


def test_signal_candidates_keeps_subtle_local_peak():
    s = np.zeros(40)
    s[10] = 0.5   # subtle
    s[30] = 5.0   # loud, far away
    idxs = [i for i, _ in _signal_candidates(s, weight=1.0, k=1.5, local_half_win=6, nms_radius=2)]
    assert 10 in idxs and 30 in idxs, "loud peak must not suppress the distant subtle one"


def test_late_fusion_unions_signals():
    # Regression for the demo miss: signal A has a loud cut at t=2s, signal B a
    # modest cut at t=7s. A single global threshold (set by A) would bury B; late
    # fusion nominates each against its own baseline, so both survive.
    grid = np.linspace(0.0, 10.0, 41)  # 0.25 s grid
    a = np.zeros(41); a[8] = 1.0     # t=2.0
    b = np.zeros(41); b[28] = 0.4    # t=7.0
    pooled = []
    for name, s in (("A", a), ("B", b)):
        for idx, strength in _signal_candidates(s, weight=1.0, k=1.5, local_half_win=8, nms_radius=4):
            pooled.append((idx, strength, name))
    chosen = sorted(i for i, _, _ in _nms_candidates(pooled, grid, min_spacing_s=2.0))
    assert 8 in chosen and 28 in chosen


def test_nms_candidates_enforces_spacing():
    grid = np.arange(0.0, 10.0, 0.25)
    # two strong candidates 0.5 s apart -> only the stronger survives a 2 s min.
    pooled = [(10, 0.9, "A"), (12, 1.0, "B"), (30, 0.7, "A")]
    chosen = sorted(i for i, _, _ in _nms_candidates(pooled, grid, min_spacing_s=2.0))
    assert chosen == [12, 30]


def test_quaternion_angle():
    assert abs(_quaternion_angle([1, 0, 0, 0], [1, 0, 0, 0])) < 1e-6
    # [0,1,0,0] is a 180-degree rotation about x.
    assert abs(_quaternion_angle([1, 0, 0, 0], [0, 1, 0, 0]) - math.pi) < 1e-4


def test_hand_signals():
    rows = [
        {"timestamp_ms": 0, "left_hand_landmarks": [[0.1, 0.1, 0.0]] * 21, "right_hand_landmarks": None},
        {"timestamp_ms": 100, "left_hand_landmarks": [[0.2, 0.1, 0.0]] * 21, "right_hand_landmarks": None},
        {"timestamp_ms": 200, "left_hand_landmarks": None, "right_hand_landmarks": None},
    ]
    times, counts, speeds = hand_signals(rows)
    assert times == [0.0, 0.1, 0.2]
    assert counts == [1.0, 1.0, 0.0]           # hand leaves frame on the last row
    assert abs(speeds[1] - 1.0) < 1e-6          # moved 0.1 over 0.1 s


def test_ego_signals():
    frames = [
        {"timestamp_ms": 0, "position": [0, 0, 0], "quaternion": [1, 0, 0, 0], "tracked": True},
        {"timestamp_ms": 100, "position": [1, 0, 0], "quaternion": [1, 0, 0, 0], "tracked": True},
        {"timestamp_ms": 200, "position": [1, 0, 0], "quaternion": [0.7071, 0, 0.7071, 0], "tracked": True},
        {"timestamp_ms": 300, "position": [9, 9, 9], "quaternion": [1, 0, 0, 0], "tracked": False},
    ]
    times, trans, rot = ego_signals(frames)
    assert len(times) == 3                       # untracked frame dropped
    assert trans[0] == 0.0 and rot[0] == 0.0
    assert trans[1] > 0.0                         # translated
    assert rot[2] > 0.0                           # rotated ~90 deg


# --- ffmpeg-backed integration tests (skip if ffmpeg missing) ----------------

def _make_scene_change_clip(path: Path, seconds_each: int = 3, fps: int = 30):
    """Animated testsrc (high motion) followed by a static red field (no motion):
    a strong appearance change at t=seconds_each."""
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=duration={seconds_each}:size=320x240:rate={fps}",
        "-f", "lavfi", "-i", f"color=c=red:duration={seconds_each}:size=320x240:rate={fps}",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1[outv]",
        "-map", "[outv]", "-pix_fmt", "yuv420p", str(path),
    ], check=True)


def test_detect_boundaries_finds_scene_change():
    if shutil.which("ffmpeg") is None:
        print("skip: ffmpeg not available")
        return
    from pipeline.video_meta import probe
    tmp = Path(tempfile.mkdtemp())
    clip = tmp / "scene.mp4"
    _make_scene_change_clip(clip, seconds_each=3)
    meta = probe(clip)
    res = detect_boundaries(clip, meta, hand_rows=[], head_frames=[],
                            grid_fps=4.0, min_segment_seconds=1.5)
    assert "frame_diff" in res.signals_used
    interior = [b for b in res.boundaries if 0.1 < b < meta.duration_seconds - 0.1]
    assert interior, "expected at least one interior cut"
    assert any(abs(b - 3.0) <= 1.2 for b in interior), f"no cut near 3.0 s: {res.boundaries}"
    print(f"ok: boundaries={res.boundaries} signals={res.signals_used}")


class _CyclingProvider:
    """Returns alternating labels so distinct segments don't all coalesce away."""
    name = "fake"
    model = "fake-vlm"

    def __init__(self):
        self.i = 0

    def classify(self, jpeg_bytes):
        lab = ["transit/walking", "loading/unloading"][self.i % 2]
        self.i += 1
        return FrameLabel(task=lab, confidence=0.7, description=f"{lab} commentary"), 0.0


def test_segment_video_fused_end_to_end():
    if shutil.which("ffmpeg") is None:
        print("skip: ffmpeg not available")
        return
    from config import settings
    from pipeline.segmentation import segment_video
    tmp = Path(tempfile.mkdtemp())
    clip = tmp / "scene.mp4"
    _make_scene_change_clip(clip, seconds_each=3)

    prev_mode = settings.segmentation_boundary_mode
    settings.segmentation_boundary_mode = "fused"
    try:
        # Pass empty pose products to bypass storage/DB.
        result = segment_video(clip, "fusedvid", provider=_CyclingProvider(),
                               hand_rows=[], head_frames=[])
    finally:
        settings.segmentation_boundary_mode = prev_mode

    assert result.cost_usd == 0.0, "fake provider must be free"
    assert result.boundary_meta and result.boundary_meta["mode"] == "fused"
    assert result.segments, "expected at least one segment"
    # Far fewer classifications than a 1 fps per-frame pass over a 6 s clip.
    assert result.frames_classified <= 8
    for s in result.segments:
        assert 0 <= s.start_time <= s.end_time <= result.duration_seconds + 1
    print(f"ok: {len(result.segments)} segments, frames_classified={result.frames_classified}, "
          f"signals={result.boundary_meta['signals_used']}")


if __name__ == "__main__":
    test_robust_norm()
    test_resample_to_grid()
    test_novelty_score_peaks_at_step()
    test_pick_peaks_respects_spacing()
    test_local_zscore_flags_local_outlier()
    test_signal_candidates_keeps_subtle_local_peak()
    test_late_fusion_unions_signals()
    test_nms_candidates_enforces_spacing()
    test_quaternion_angle()
    test_hand_signals()
    test_ego_signals()
    test_detect_boundaries_finds_scene_change()
    test_segment_video_fused_end_to_end()
    print("ALL TESTS PASSED")
