"""Signal-fusion boundary detection (decouples *where the cuts are* from *what
each clip is*).

The per-frame VLM path (``segmentation._aggregate*``) makes a cut wherever two
adjacent 1 fps labels happen to differ — so boundaries are quantized to the
sample grid, every frame costs a VLM call, and a single bad caption invents a
boundary. This module instead *finds* the cuts with cheap, dense signals the
pipeline already computes, at their native temporal resolution, for ~free:

  * frame-diff   — appearance change between consecutive frames (location/scene
                   change, large motion). Computed here from the video.
  * hand-pose    — hands entering/leaving frame + centroid velocity. Loaded from
                   the Phase-2 ``hand_pose.parquet`` (10 fps) when present.
  * ego-motion   — camera translation speed + rotation rate from the Phase-2
                   visual-odometry ``head_pose.json`` (6 fps) when present. The
                   transit<->stationary-work boundary lives here.

Each signal is resampled onto a common time grid, robustly normalized, and
stacked into a per-grid feature vector. A boundary is where the feature
*statistics* shift: we score each grid point by the L2 distance between the mean
feature vector just before it and just after it (a step / "novelty" score), then
greedily pick peaks that are above an adaptive threshold and at least
``min_segment_seconds`` apart.

The output is a set of segment intervals covering ``[0, duration]``. The caller
(``segmentation``) then asks the VLM to label each segment *once*, instead of
once per frame — far cheaper and a cleaner question for the model.

Pure-ish by design: the numeric helpers operate on plain arrays/lists so they
unit-test without a video, storage, or DB (see ``tests/test_boundary.py``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .video_meta import VideoMeta, apply_rotation

log = logging.getLogger("revisent.boundary")


@dataclass
class BoundaryResult:
    # Cut times in seconds, INCLUDING the outer 0.0 and duration endpoints.
    boundaries: List[float]
    # Consecutive (start, end) intervals derived from ``boundaries``.
    segments: List[Tuple[float, float]]
    # Which signals actually contributed (present + enabled + non-degenerate).
    signals_used: List[str] = field(default_factory=list)
    grid_fps: float = 0.0
    # For each interior cut, the signal that nominated it (late-fusion provenance);
    # parallel to the interior boundaries. Handy for tuning ("which signal found
    # this cut?") and for the eval harness.
    cut_sources: List[str] = field(default_factory=list)

    def to_meta(self) -> dict:
        return {
            "method": "late_fusion_local_threshold",
            "signals_used": self.signals_used,
            "cut_sources": self.cut_sources,
            "grid_fps": round(self.grid_fps, 3),
            "num_segments": len(self.segments),
        }


# --------------------------------------------------------------------------- #
# Numeric helpers (pure — unit-tested directly, no video/storage needed).
# --------------------------------------------------------------------------- #

def _robust_norm(values: np.ndarray, pct: float = 95.0) -> np.ndarray:
    """Scale to ~[0,1] by a high percentile so a single spike can't dominate the
    fusion. Returns zeros for an empty/constant signal."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return v
    v = np.clip(v, 0.0, None)
    scale = np.percentile(v, pct)
    if not np.isfinite(scale) or scale <= 1e-9:
        return np.zeros_like(v)
    return np.clip(v / scale, 0.0, 1.0)


def _resample_to_grid(times: Sequence[float], values: Sequence[float], grid: np.ndarray) -> np.ndarray:
    """Linear-interpolate an irregularly-sampled signal onto ``grid`` (seconds).

    Empty input -> zeros. ``np.interp`` holds the endpoints flat outside range,
    which is the behavior we want (no spurious edge transitions)."""
    grid = np.asarray(grid, dtype=np.float64)
    if len(times) == 0:
        return np.zeros_like(grid)
    t = np.asarray(times, dtype=np.float64)
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(t)
    return np.interp(grid, t[order], x[order])


def _moving_average(arr: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average with edge padding (smooths per-frame noise)."""
    arr = np.asarray(arr, dtype=np.float64)
    if win <= 1 or arr.size == 0:
        return arr
    win = min(win, arr.size)
    k = np.ones(win) / win
    pad = win // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.convolve(padded, k, mode="same")[pad: pad + arr.size]


def _novelty_score(features: np.ndarray, half_window: int) -> np.ndarray:
    """Per-row step score: L2 distance between the mean feature vector in the
    ``half_window`` rows before a point and the ``half_window`` rows after it.

    ``features`` is (T, D). Returns length-T scores; peaks mark regime changes.
    """
    if features.ndim != 2:
        raise ValueError("features must be 2-D (T, D)")
    T = features.shape[0]
    hw = max(1, int(half_window))
    scores = np.zeros(T, dtype=np.float64)
    if T < 2:
        return scores
    for i in range(T):
        lo = max(0, i - hw)
        hi = min(T, i + hw)
        if i - lo < 1 or hi - i < 1:
            continue
        before = features[lo:i].mean(axis=0)
        after = features[i:hi].mean(axis=0)
        scores[i] = float(np.linalg.norm(after - before))
    return scores


def _pick_peaks(
    scores: np.ndarray, grid: np.ndarray, *, min_spacing_s: float, threshold: float
) -> List[int]:
    """Greedily select peak indices: take the highest-scoring point above
    ``threshold``, suppress everything within ``min_spacing_s`` of it, repeat.

    Greedy non-max suppression guarantees no two cuts closer than the minimum
    segment length, and prefers the strongest transitions first."""
    scores = np.asarray(scores, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    candidates = [i for i in range(len(scores)) if scores[i] >= threshold]
    candidates.sort(key=lambda i: scores[i], reverse=True)
    chosen: List[int] = []
    for i in candidates:
        if all(abs(grid[i] - grid[j]) >= min_spacing_s for j in chosen):
            chosen.append(i)
    return sorted(chosen)


def _quaternion_angle(q1: Sequence[float], q2: Sequence[float]) -> float:
    """Smallest rotation angle (radians) between two [w,x,y,z] quaternions."""
    a = np.asarray(q1, dtype=np.float64)
    b = np.asarray(q2, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    dot = abs(float(np.dot(a / na, b / nb)))
    return 2.0 * np.arccos(min(1.0, dot))


# --------------------------------------------------------------------------- #
# Per-signal feature extraction.
# --------------------------------------------------------------------------- #

def _hand_centroid(landmarks) -> Optional[np.ndarray]:
    """Mean (x, y) of a 21x3 normalized landmark list, or None."""
    if not landmarks:
        return None
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return None
    return pts[:, :2].mean(axis=0)


def hand_signals(hand_rows: Sequence[dict]) -> Tuple[List[float], List[float], List[float]]:
    """From hand-pose rows -> (times_s, hand_count, centroid_speed).

    ``hand_count`` (0/1/2) jumps when hands enter or leave the work area;
    ``centroid_speed`` separates active manipulation from a held/idle hand."""
    times: List[float] = []
    counts: List[float] = []
    speeds: List[float] = []
    prev_c: Optional[np.ndarray] = None
    prev_t: Optional[float] = None
    for r in hand_rows:
        t = float(r.get("timestamp_ms", 0.0)) / 1000.0
        left = _hand_centroid(r.get("left_hand_landmarks"))
        right = _hand_centroid(r.get("right_hand_landmarks"))
        present = [c for c in (left, right) if c is not None]
        count = float(len(present))
        centroid = np.mean(present, axis=0) if present else None
        speed = 0.0
        if centroid is not None and prev_c is not None and prev_t is not None:
            dt = max(t - prev_t, 1e-3)
            speed = float(np.linalg.norm(centroid - prev_c) / dt)  # normalized units / s
        times.append(t)
        counts.append(count)
        speeds.append(speed)
        if centroid is not None:
            prev_c, prev_t = centroid, t
    return times, counts, speeds


def ego_signals(head_frames: Sequence[dict]) -> Tuple[List[float], List[float], List[float]]:
    """From head-pose VO frames -> (times_s, translation_speed, rotation_rate).

    VO position is up-to-scale and drifts, so we use *frame-to-frame* deltas
    (relative motion), not absolute position. Rotation rate is the more reliable
    of the two per the VO module's own note."""
    times: List[float] = []
    trans: List[float] = []
    rot: List[float] = []
    prev_p: Optional[np.ndarray] = None
    prev_q: Optional[Sequence[float]] = None
    prev_t: Optional[float] = None
    for f in head_frames:
        if not f.get("tracked", True):
            continue
        t = float(f.get("timestamp_ms", 0.0)) / 1000.0
        p = np.asarray(f.get("position", [0, 0, 0]), dtype=np.float64)
        q = f.get("quaternion", [1, 0, 0, 0])
        ts = rs = 0.0
        if prev_p is not None and prev_t is not None:
            dt = max(t - prev_t, 1e-3)
            ts = float(np.linalg.norm(p - prev_p) / dt)
            rs = _quaternion_angle(prev_q, q) / dt
        times.append(t)
        trans.append(ts)
        rot.append(rs)
        prev_p, prev_q, prev_t = p, q, t
    return times, trans, rot


def frame_diff_signals(
    video_path: Path, meta: VideoMeta, sample_fps: float, *, thumb: int = 64
) -> Tuple[List[float], List[float], List[float]]:
    """Decode the video at ~``sample_fps`` and measure appearance change between
    consecutive sampled frames -> (times_s, pixel_diff, hist_diff).

      * pixel_diff: mean abs difference of small grayscale thumbnails (motion /
        local change).
      * hist_diff : 1 - histogram correlation (global appearance / location
        change; robust to small motion).
    """
    src_fps = meta.fps or 30.0
    stride = max(1, int(round(src_fps / max(sample_fps, 0.1))))
    times: List[float] = []
    pix: List[float] = []
    hist: List[float] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    prev_small: Optional[np.ndarray] = None
    prev_hist: Optional[np.ndarray] = None
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                frame = apply_rotation(frame, meta.rotation)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (thumb, thumb), interpolation=cv2.INTER_AREA)
                h = cv2.calcHist([gray], [0], None, [64], [0, 256])
                cv2.normalize(h, h, 0.0, 1.0, cv2.NORM_MINMAX)
                t = round(idx / src_fps, 3)
                if prev_small is not None:
                    pd = float(np.abs(small.astype(np.float64) - prev_small).mean() / 255.0)
                    corr = float(cv2.compareHist(prev_hist, h, cv2.HISTCMP_CORREL))
                    hd = max(0.0, 1.0 - corr)
                else:
                    pd = hd = 0.0
                times.append(t)
                pix.append(pd)
                hist.append(hd)
                prev_small, prev_hist = small.astype(np.float64), h
            idx += 1
    finally:
        cap.release()
    return times, pix, hist


# --------------------------------------------------------------------------- #
# Late fusion: each signal nominates cuts against its OWN local baseline.
#
# Early fusion (stack all signals -> one novelty score -> one global threshold)
# has a failure mode: a clip with one dramatic change and several subtle ones
# lets the dramatic change raise the global bar so high the subtle cuts are
# missed, and it dilutes a sharp single-signal cue (e.g. a histogram jump) by
# averaging it with quieter signals. Late fusion fixes both — every signal scores
# itself, is thresholded LOCALLY (so a loud event elsewhere doesn't hide a quiet
# one here), and the candidates are unioned. A cut only one signal sees survives.
# --------------------------------------------------------------------------- #

def _step_score(series: np.ndarray, half_window: int) -> np.ndarray:
    """Change score for a LEVEL signal (motion, hand presence, ego speed): the
    before/after-mean step magnitude. Peaks where the level shifts."""
    return _novelty_score(np.asarray(series, dtype=np.float64).reshape(-1, 1), half_window)


def _local_zscore(x: np.ndarray, half_win: int) -> np.ndarray:
    """(value - local_mean) / local_std over a centered sliding window. Lets a
    point be judged against its NEIGHBORHOOD, not the whole clip."""
    x = np.asarray(x, dtype=np.float64)
    T = x.size
    z = np.zeros(T)
    hw = max(1, int(half_win))
    for i in range(T):
        w = x[max(0, i - hw): min(T, i + hw + 1)]
        sd = w.std()
        z[i] = (x[i] - w.mean()) / sd if sd > 1e-9 else 0.0
    return z


def _signal_candidates(
    change_score: np.ndarray, *, weight: float, k: float, local_half_win: int, nms_radius: int,
) -> List[Tuple[int, float]]:
    """Candidate cut indices for one signal: local maxima whose LOCAL z-score
    clears ``k``. Strength = z * weight, so candidates are comparable across
    signals of different magnitudes when they're pooled."""
    z = _local_zscore(change_score, local_half_win)
    T = change_score.size
    out: List[Tuple[int, float]] = []
    r = max(1, int(nms_radius))
    for i in range(T):
        if change_score[i] <= 1e-9 or z[i] < k:
            continue
        lo, hi = max(0, i - r), min(T, i + r + 1)
        if change_score[i] >= change_score[lo:hi].max():  # local maximum
            out.append((i, float(z[i] * max(weight, 0.0))))
    return out


def _nms_candidates(
    cands: List[Tuple[int, float, str]], grid: np.ndarray, min_spacing_s: float,
) -> List[Tuple[int, float, str]]:
    """Greedy non-max suppression over POOLED candidates: take the strongest, drop
    anything within ``min_spacing_s`` of it, repeat. Guarantees the minimum
    segment length and keeps the strongest nomination when signals agree."""
    chosen: List[Tuple[int, float, str]] = []
    for idx, strength, name in sorted(cands, key=lambda c: c[1], reverse=True):
        if all(abs(grid[idx] - grid[j]) >= min_spacing_s for j, _, _ in chosen):
            chosen.append((idx, strength, name))
    return chosen


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #

def _segments_from_boundaries(boundaries: Sequence[float]) -> List[Tuple[float, float]]:
    b = sorted(set(round(x, 3) for x in boundaries))
    return [(b[i], b[i + 1]) for i in range(len(b) - 1) if b[i + 1] - b[i] > 1e-3]


# ===========================================================================
# FUTURE IMPROVEMENTS (Tier 2/3) — raise the accuracy ceiling once a labeled
# eval set exists (scripts/eval_boundaries.py). Tune/validate against that F1,
# NOT against synthetic clips (downscaled noise behaves unlike real footage).
#
#   Tier 2 — better signals:
#     * Learned-embedding distance (CLIP/DINOv2) between consecutive frames as a
#       signal — catches SEMANTIC scene/activity changes that look pixel-similar.
#       Single biggest lever; cheap on a local GPU. Add as another series here.
#     * Color (HSV) histograms + edge-change ratio instead of grayscale only.
#     * Optical-flow magnitude as the motion signal (more robust than pixel-diff
#       to global camera motion vs. content change).
#     * Ensure the Phase-2 hand/ego products are actually present at runtime —
#       on real footage they mark manipulation/transit cuts pixels can't see.
#
#   Tier 3 — better decision + precision:
#     * Replace local-threshold peak picking with change-point detection
#       (ruptures PELT/KernelCPD) or kernel temporal segmentation (KTS) on the
#       stacked signals — principled, fewer hand-tuned knobs.
#     * Sub-grid refinement: binary-search the exact cut frame within the chosen
#       grid cell using the cheap frame-diff signal.
#     * Emit a per-cut confidence (z-strength is already computed) and expose it
#       so downstream can filter weak cuts.
# ===========================================================================

def detect_boundaries(
    video_path: Path,
    meta: VideoMeta,
    *,
    hand_rows: Optional[Sequence[dict]] = None,
    head_frames: Optional[Sequence[dict]] = None,
    grid_fps: float = 4.0,
    min_segment_seconds: float = 2.0,
    smooth_seconds: float = 0.75,
    window_seconds: float = 1.5,
    threshold_k: float = 1.5,
    local_window_seconds: float = 4.0,
    max_segments: int = 60,
    use_frame_diff: bool = True,
    use_hand_pose: bool = True,
    use_ego_motion: bool = True,
    weight_frame_diff: float = 1.0,
    weight_hand_pose: float = 1.0,
    weight_ego_motion: float = 1.0,
) -> BoundaryResult:
    """Late-fusion boundary detection: each cheap signal nominates cut candidates
    against its own local baseline; the union (greedy-NMS'd) becomes the cuts.

    Falls back gracefully: any missing/disabled/degenerate signal is simply not
    a nominator. If *no* signal is usable the whole clip is one segment.

    ``threshold_k`` is now a LOCAL z-score threshold (k std above the sliding
    local baseline), not a global one — ~1.5-2.5 is typical.
    """
    duration = meta.duration_seconds or 0.0
    if duration <= 0:
        # Derive from frame count when ffprobe didn't give us a duration.
        duration = (meta.frame_count / (meta.fps or 30.0)) if meta.frame_count else 0.0

    grid_fps = max(0.5, float(grid_fps))
    n = int(round(duration * grid_fps)) + 1 if duration > 0 else 0
    if n < 4 or duration <= min_segment_seconds:
        return BoundaryResult([0.0, duration], _segments_from_boundaries([0.0, duration]),
                              signals_used=[], grid_fps=grid_fps)
    grid = np.linspace(0.0, duration, n)
    smooth_win = max(1, int(round(smooth_seconds * grid_fps)))
    half_window = max(1, int(round(window_seconds * grid_fps)))
    local_half = max(1, int(round(local_window_seconds * grid_fps)))
    nms_radius = max(1, int(round(min_segment_seconds * grid_fps / 2)))

    # Each entry: (signal_name, change_score_series, weight).
    series: List[Tuple[str, np.ndarray, float]] = []
    used: List[str] = []

    if use_frame_diff:
        try:
            ft, pix, hist = frame_diff_signals(video_path, meta, grid_fps)
            if any(v > 0 for v in pix) or any(v > 0 for v in hist):
                # Motion is a LEVEL -> smooth to de-noise, then step score.
                pix_g = _moving_average(_robust_norm(_resample_to_grid(ft, pix, grid)), smooth_win)
                series.append(("frame_diff:motion", _step_score(pix_g, half_window), weight_frame_diff))
                # Histogram discontinuity is a SPIKE that already peaks AT the cut
                # (the canonical shot-boundary cue) -> use it RAW. Smoothing would
                # spread a sharp single-frame jump and sink it below threshold.
                hist_g = _robust_norm(_resample_to_grid(ft, hist, grid))
                series.append(("frame_diff:appearance", hist_g, weight_frame_diff))
                used.append("frame_diff")
        except Exception as exc:  # noqa: BLE001 — never let a signal failure abort
            log.warning("frame-diff signal failed: %s", exc)

    if use_hand_pose and hand_rows:
        ht, counts, speeds = hand_signals(hand_rows)
        if ht:
            cg = _moving_average(_robust_norm(_resample_to_grid(ht, counts, grid)), smooth_win)
            sg = _moving_average(_robust_norm(_resample_to_grid(ht, speeds, grid)), smooth_win)
            series.append(("hand:presence", _step_score(cg, half_window), weight_hand_pose))
            series.append(("hand:velocity", _step_score(sg, half_window), weight_hand_pose))
            used.append("hand_pose")

    if use_ego_motion and head_frames:
        et, trans, rot = ego_signals(head_frames)
        if et:
            tg = _moving_average(_robust_norm(_resample_to_grid(et, trans, grid)), smooth_win)
            rg = _moving_average(_robust_norm(_resample_to_grid(et, rot, grid)), smooth_win)
            series.append(("ego:translation", _step_score(tg, half_window), weight_ego_motion))
            series.append(("ego:rotation", _step_score(rg, half_window), weight_ego_motion))
            used.append("ego_motion")

    if not series:
        log.warning("no usable boundary signals; returning a single segment")
        return BoundaryResult([0.0, duration], _segments_from_boundaries([0.0, duration]),
                              signals_used=[], grid_fps=grid_fps)

    # Late fusion: pool every signal's local-threshold candidates, then NMS.
    pooled: List[Tuple[int, float, str]] = []
    for name, score, weight in series:
        for idx, strength in _signal_candidates(
            score, weight=weight, k=threshold_k, local_half_win=local_half, nms_radius=nms_radius
        ):
            pooled.append((idx, strength, name))
    chosen = _nms_candidates(pooled, grid, min_segment_seconds)

    # Cap the number of cuts: keep the strongest if we overshoot.
    if max_segments and len(chosen) > max_segments - 1:
        chosen = sorted(chosen, key=lambda c: c[1], reverse=True)[: max_segments - 1]
    chosen.sort(key=lambda c: c[0])

    interior = [round(float(grid[idx]), 3) for idx, _, _ in chosen]
    cut_sources = [name for _, _, name in chosen]
    boundaries = [0.0] + interior + [round(duration, 3)]
    segments = _segments_from_boundaries(boundaries)
    log.info("boundaries: %d segment(s) from signals=%s via %s (grid %.1f fps)",
             len(segments), ",".join(used) or "none", ",".join(cut_sources) or "none", grid_fps)
    return BoundaryResult(sorted(set(boundaries)), segments, signals_used=used,
                          grid_fps=grid_fps, cut_sources=cut_sources)
