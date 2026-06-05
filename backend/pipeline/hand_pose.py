"""Hand pose extraction via MediaPipe Hands (Phase 2).

Runs on the *anonymized* video (faces blurred, hands untouched) and produces
per-frame 21-point hand keypoints. Output is a Parquet file:

    data/processed/{video_id}/hand_pose.parquet

Columns:
    frame_number          int   — source frame index that was sampled
    timestamp_ms          float — frame_number / source_fps * 1000
    left_hand_landmarks   list<list<float>>(21x3) | null  — normalized x,y,z
    right_hand_landmarks  list<list<float>>(21x3) | null
    left_confidence       float | null  — handedness classification score
    right_confidence      float | null

Landmarks are normalized: x,y in [0,1] relative to frame width/height, z is a
relative depth (smaller = closer to camera). Storing normalized coords keeps the
data resolution-independent for the dashboard canvas overlay.

Sampling rate and provenance are embedded in the Parquet schema metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from config import settings
from .video_meta import probe, apply_rotation

mp_hands = mp.solutions.hands

# 21 landmarks, each [x, y, z]. The outer list is variable-length (rather than
# fixed at 21) because fixed-size lists mishandle null rows in pyarrow — a null
# hand must round-trip as null, not an empty list. The 21-point shape is enforced
# in code (MediaPipe always returns 21) and documented in the schema metadata.
_LANDMARK_TYPE = pa.list_(pa.list_(pa.float32(), 3))


@dataclass
class HandPoseResult:
    video_id: str
    output_path: str
    source_fps: float
    sample_fps: float
    sample_stride: int
    frames_total: int
    frames_sampled: int
    frames_with_any_hand: int
    left_hand_frames: int
    right_hand_frames: int
    coverage: float  # fraction of sampled frames with >=1 hand

    def as_meta(self) -> dict:
        return asdict(self)


def _landmarks_to_list(landmark_list) -> List[List[float]]:
    return [[lm.x, lm.y, lm.z] for lm in landmark_list.landmark]


def _temporal_fill(seq, times, max_gap_s):
    """Bridge short None-gaps (blurred frames a detector missed) by linear interp
    between the surrounding detections. Only gaps <= max_gap_s are filled; longer
    dropouts are left genuine. Returns (new_seq, filled_mask)."""
    out = list(seq)
    filled = [False] * len(seq)
    valid = [i for i, s in enumerate(seq) if s is not None]
    for a, b in zip(valid, valid[1:]):
        if b - a <= 1 or (times[b] - times[a]) / 1000.0 > max_gap_s:
            continue
        A = np.asarray(seq[a], dtype=np.float32)
        B = np.asarray(seq[b], dtype=np.float32)
        span = times[b] - times[a] or 1.0
        for k in range(a + 1, b):
            t = (times[k] - times[a]) / span
            out[k] = ((1 - t) * A + t * B).tolist()
            filled[k] = True
    return out, filled


def _smooth_oneeuro(seq, times, min_cutoff=1.5, beta=0.04):
    """One-Euro smoothing: de-jitters slow/noisy detections but stays responsive to
    fast motion (cutoff rises with speed). Resets across genuine dropouts."""
    out = list(seq)
    x_prev = dx_prev = None
    prev_t = None

    def alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    for i, s in enumerate(seq):
        if s is None:
            x_prev = dx_prev = prev_t = None
            continue
        x = np.asarray(s, dtype=np.float32)
        t = times[i] / 1000.0
        if x_prev is None:
            x_prev, dx_prev, prev_t = x, np.zeros_like(x), t
            continue
        dt = max(1e-3, t - prev_t)
        dx = (x - x_prev) / dt
        dx_hat = alpha(1.0, dt) * dx + (1 - alpha(1.0, dt)) * dx_prev
        cutoff = min_cutoff + beta * float(np.linalg.norm(dx_hat))
        a = alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * x_prev
        out[i] = x_hat.tolist()
        x_prev, dx_prev, prev_t = x_hat, dx_hat, t
    return out


def extract_hand_pose(
    video_path: Path,
    output_path: Path,
    video_id: str,
    sample_fps: Optional[float] = None,
) -> HandPoseResult:
    """Extract per-frame hand keypoints and write a Parquet file."""
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = probe(video_path)
    src_fps = meta.fps or 30.0
    sample_fps = sample_fps or settings.hand_pose_sample_fps
    stride = max(1, int(round(src_fps / sample_fps)))
    effective_sample_fps = src_fps / stride

    frame_numbers: List[int] = []
    timestamps: List[float] = []
    left_lms: List[Optional[List[List[float]]]] = []
    right_lms: List[Optional[List[List[float]]]] = []
    left_conf: List[Optional[float]] = []
    right_conf: List[Optional[float]] = []

    frames_total = 0
    frames_sampled = 0
    frames_with_any = 0
    left_count = 0
    right_count = 0

    swap = settings.hand_pose_swap_handedness

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=settings.hand_pose_max_hands,
            model_complexity=settings.hand_pose_model_complexity,
            min_detection_confidence=settings.hand_pose_min_detection_confidence,
            min_tracking_confidence=settings.hand_pose_min_tracking_confidence,
        ) as hands:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames_total += 1
                if idx % stride != 0:
                    idx += 1
                    continue

                frame = apply_rotation(frame, meta.rotation)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                left: Optional[List[List[float]]] = None
                right: Optional[List[List[float]]] = None
                lc: Optional[float] = None
                rc: Optional[float] = None

                if results.multi_hand_landmarks and results.multi_handedness:
                    for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                        cls = handed.classification[0]
                        label = cls.label  # "Left" / "Right"
                        if swap:
                            label = "Right" if label == "Left" else "Left"
                        coords = _landmarks_to_list(lm)
                        if label == "Left":
                            left, lc = coords, float(cls.score)
                        else:
                            right, rc = coords, float(cls.score)

                if left is not None or right is not None:
                    frames_with_any += 1
                if left is not None:
                    left_count += 1
                if right is not None:
                    right_count += 1

                frame_numbers.append(idx)
                timestamps.append(round(idx / src_fps * 1000.0, 2))
                left_lms.append(left)
                right_lms.append(right)
                left_conf.append(lc)
                right_conf.append(rc)

                frames_sampled += 1
                idx += 1
    finally:
        cap.release()

    # --- Temporal robustness for blurry/fast footage ---
    # Bridge short blur-dropouts, then de-jitter without lagging fast motion.
    max_gap = settings.hand_pose_gap_fill_seconds
    left_lms, left_filled = _temporal_fill(left_lms, timestamps, max_gap)
    right_lms, right_filled = _temporal_fill(right_lms, timestamps, max_gap)
    for i, f in enumerate(left_filled):
        if f and left_conf[i] is None:
            left_conf[i] = 0.2   # mark interpolated frames (low confidence)
    for i, f in enumerate(right_filled):
        if f and right_conf[i] is None:
            right_conf[i] = 0.2
    if settings.hand_pose_smooth:
        left_lms = _smooth_oneeuro(left_lms, timestamps)
        right_lms = _smooth_oneeuro(right_lms, timestamps)
    # Recompute coverage/counts to include the bridged frames.
    frames_with_any = sum(1 for l, r in zip(left_lms, right_lms) if l is not None or r is not None)
    left_count = sum(1 for l in left_lms if l is not None)
    right_count = sum(1 for r in right_lms if r is not None)

    table = pa.table(
        {
            "frame_number": pa.array(frame_numbers, type=pa.int32()),
            "timestamp_ms": pa.array(timestamps, type=pa.float64()),
            "left_hand_landmarks": pa.array(left_lms, type=_LANDMARK_TYPE),
            "right_hand_landmarks": pa.array(right_lms, type=_LANDMARK_TYPE),
            "left_confidence": pa.array(left_conf, type=pa.float32()),
            "right_confidence": pa.array(right_conf, type=pa.float32()),
        }
    )

    coverage = (frames_with_any / frames_sampled) if frames_sampled else 0.0

    # Embed provenance/sampling info in the parquet schema metadata so it travels
    # with the file.
    schema_meta = {
        b"video_id": video_id.encode(),
        b"model": b"mediapipe_hands",
        b"landmark_count": b"21",
        b"coord_order": b"x,y,z (normalized; x,y in [0,1], z relative depth)",
        b"source_fps": str(src_fps).encode(),
        b"sample_fps": str(round(effective_sample_fps, 4)).encode(),
        b"sample_stride": str(stride).encode(),
        b"handedness_swapped": str(swap).encode(),
        b"handedness_note": b"MediaPipe assumes a mirrored image; swapped for forward-facing chest cam",
    }
    table = table.replace_schema_metadata(schema_meta)
    pq.write_table(table, output_path)

    return HandPoseResult(
        video_id=video_id,
        output_path=str(output_path),
        source_fps=src_fps,
        sample_fps=round(effective_sample_fps, 4),
        sample_stride=stride,
        frames_total=frames_total,
        frames_sampled=frames_sampled,
        frames_with_any_hand=frames_with_any,
        left_hand_frames=left_count,
        right_hand_frames=right_count,
        coverage=round(coverage, 4),
    )


def load_hand_pose(parquet_path: Path) -> List[dict]:
    """Load hand pose parquet into a list of per-frame dicts (for API / tests)."""
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    return rows


def read_hand_pose_metadata(parquet_path: Path) -> dict:
    """Read the embedded schema metadata (sampling rate, model, etc.)."""
    schema = pq.read_schema(parquet_path)
    meta = schema.metadata or {}
    return {k.decode(): v.decode() for k, v in meta.items()}
