"""Hand pose extraction via MediaPipe Hands (Phase 2).

Runs on the *anonymized* video (faces blurred, hands untouched) and produces
per-frame 21-point hand keypoints. Output is a Parquet file:

    data/processed/{video_id}/hand_pose.parquet

Columns:
    frame_number          int   — source frame index that was sampled
    timestamp_ms          float — frame_number / source_fps * 1000
    left_hand_landmarks   list<list<float>>(21x3) | null  — normalized x,y,z
    right_hand_landmarks  list<list<float>>(21x3) | null
    left_world_landmarks  list<list<float>>(21x3) | null  — METRIC x,y,z (meters)
    right_world_landmarks list<list<float>>(21x3) | null
    left_confidence       float | null  — handedness classification score
    right_confidence      float | null

Image landmarks are normalized: x,y in [0,1] relative to frame width/height, z a
relative depth (smaller = closer to camera) — resolution-independent for the
dashboard canvas overlay. World landmarks are MediaPipe's metric 3D estimate in
METERS, origin at the hand's geometric center: real hand shape/size, which gives
metric grasp aperture and lets depth be recovered from apparent hand scale.

Sampling rate and provenance are embedded in the Parquet schema metadata.
"""
from __future__ import annotations

import json
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


def _fix_two_hand_swaps(left, right, left_w, right_w, l_conf, r_conf, times,
                        max_gap_s=0.5):
    """Crossover identity fix: in TWO-hand frames, assign detections to sides by
    PREDICTED track positions (constant-velocity), not the classifier's
    per-frame vote — which flips exactly when hands cross/overlap. Velocity
    matters: at the crossing instant both hands occupy the same spot, and only
    direction of travel disambiguates them. In place; returns swap count."""
    # side -> (pos xy, vel xy, t)
    trk = {"L": None, "R": None}
    swaps = 0

    def predict(side, t):
        s = trk[side]
        if s is None or (t - s[2]) > max_gap_s:
            return None
        dt = t - s[2]
        return (s[0][0] + s[1][0] * dt, s[0][1] + s[1][1] * dt)

    def update(side, pos, t):
        s = trk[side]
        if s is not None and (t - s[2]) <= max_gap_s and (t - s[2]) > 1e-4:
            dt = t - s[2]
            vel = ((pos[0] - s[0][0]) / dt, (pos[1] - s[0][1]) / dt)
        else:
            vel = (0.0, 0.0)
        trk[side] = (pos, vel, t)

    for i in range(len(times)):
        t = times[i] / 1000.0
        L, R = left[i], right[i]
        if L is not None and R is not None:
            pl, pr = predict("L", t), predict("R", t)
            if pl is not None and pr is not None:
                wl, wr = L[0], R[0]
                d = lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1]))  # noqa: E731
                keep = d(wl, pl) + d(wr, pr)
                swap = d(wl, pr) + d(wr, pl)
                if swap < keep * 0.8 and keep - swap > 0.02:
                    left[i], right[i] = right[i], left[i]
                    left_w[i], right_w[i] = right_w[i], left_w[i]
                    l_conf[i], r_conf[i] = r_conf[i], l_conf[i]
                    swaps += 1
        if left[i] is not None:
            update("L", (left[i][0][0], left[i][0][1]), t)
        if right[i] is not None:
            update("R", (right[i][0][0], right[i][0][1]), t)
    return swaps


def _reject_teleports(seq, world, conf, times, max_speed=6.0):
    """Physics gate: a wrist cannot cross several image-widths in one sample.
    Jumps faster than ``max_speed`` (normalized units/second) are detector
    identity errors — null the frame and let gap-fill bridge it. In place;
    returns rejection count."""
    last_p, last_t = None, None
    rejected = 0
    for i in range(len(times)):
        s = seq[i]
        if s is None:
            continue
        t = times[i] / 1000.0
        if last_p is not None:
            dt = max(1e-3, t - last_t)
            speed = float(np.hypot(s[0][0] - last_p[0], s[0][1] - last_p[1])) / dt
            if speed > max_speed:
                seq[i] = None
                world[i] = None
                conf[i] = None
                rejected += 1
                continue  # keep last_p anchored at the trusted position
        last_p, last_t = (s[0][0], s[0][1]), t
    return rejected


def _fix_handedness_flips(left, right, left_w, right_w, l_conf, r_conf, times,
                          max_gap_s=0.6, max_jump=0.10, min_majority=0.65):
    """Relabel single-hand frames whose handedness flipped mid-track.

    MediaPipe classifies L/R per frame with no memory, so a lone hand flickers
    sides. We group consecutive SINGLE-hand frames into spatial-continuity runs
    (wrist moves < max_jump between frames, gap <= max_gap_s; two-hand frames
    break runs and are never touched), then assign each whole run the side the
    classifier voted for by confidence-weighted MAJORITY. Per-run voting cannot
    cascade (an early misflag can't capture the track, unlike follow-the-last-
    seen schemes — measured to corrupt a real clip). Ties/weak majorities are
    left alone. Operates in place; returns the number of relabeled frames."""
    n = len(times)

    def rec(i):
        if (left[i] is None) == (right[i] is None):
            return None  # zero or two hands: not a single-hand frame
        side = "L" if left[i] is not None else "R"
        h = left[i] if side == "L" else right[i]
        conf = (l_conf[i] if side == "L" else r_conf[i]) or 0.5
        return side, (float(h[0][0]), float(h[0][1])), float(conf)

    runs, cur = [], []
    for i in range(n):
        r = rec(i)
        if r is None:
            if cur:
                runs.append(cur); cur = []
            continue
        if cur:
            pj, pt = cur[-1][2], times[cur[-1][0]] / 1000.0
            d = float(np.hypot(r[1][0] - pj[0], r[1][1] - pj[1]))
            if d > max_jump or (times[i] / 1000.0 - pt) > max_gap_s:
                runs.append(cur); cur = []
        cur.append((i, r[0], r[1], r[2]))
    if cur:
        runs.append(cur)

    fixes = 0
    for run in runs:
        if len(run) < 2:
            continue
        wl = sum(c for _, s, _, c in run if s == "L")
        wr = sum(c for _, s, _, c in run if s == "R")
        total = wl + wr
        if total <= 0 or max(wl, wr) / total < min_majority:
            continue  # ambiguous run: leave the classifier's labels alone
        win = "L" if wl > wr else "R"
        for i, s, _, _ in run:
            if s == win:
                continue
            if win == "R":
                right[i], left[i] = left[i], None
                right_w[i], left_w[i] = left_w[i], None
                r_conf[i], l_conf[i] = l_conf[i], None
            else:
                left[i], right[i] = right[i], None
                left_w[i], right_w[i] = right_w[i], None
                l_conf[i], r_conf[i] = r_conf[i], None
            fixes += 1
    return fixes


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


def _smooth_kalman(seq, times, process_var=0.5, meas_var=1e-4):
    """Constant-velocity Kalman filter per landmark coordinate: predicts motion
    between samples, so it smooths jitter without the low-speed lag of a plain
    low-pass. Resets across genuine dropouts. seq entries: None | 21x3 lists."""
    out = list(seq)
    x = v = Ppp = Ppv = Pvv = None
    prev_t = None
    for i, s in enumerate(seq):
        if s is None:
            x = None
            continue
        z = np.asarray(s, dtype=np.float64).reshape(-1)
        t = times[i] / 1000.0
        if x is None:
            x, v = z.copy(), np.zeros_like(z)
            Ppp = np.full_like(z, 1e-2)
            Ppv = np.zeros_like(z)
            Pvv = np.full_like(z, 1e-1)
            prev_t = t
            continue
        dt = max(1e-3, t - prev_t)
        prev_t = t
        # Predict (per-dimension 2x2 covariance, white-noise acceleration model).
        x = x + v * dt
        q = process_var
        Ppp_n = Ppp + 2 * dt * Ppv + dt * dt * Pvv + q * dt ** 3 / 3.0
        Ppv_n = Ppv + dt * Pvv + q * dt * dt / 2.0
        Pvv_n = Pvv + q * dt
        # Update.
        S = Ppp_n + meas_var
        K1, K2 = Ppp_n / S, Ppv_n / S
        innov = z - x
        x = x + K1 * innov
        v = v + K2 * innov
        Ppp = (1.0 - K1) * Ppp_n
        Ppv = (1.0 - K1) * Ppv_n
        Pvv = Pvv_n - K2 * Ppv_n
        out[i] = x.reshape(21, 3).tolist()
    return out


def _smooth(seq, times):
    """Configured temporal smoother (oneeuro | kalman | none)."""
    kind = str(getattr(settings, "hand_pose_smoother", "oneeuro"))
    if kind == "kalman":
        return _smooth_kalman(seq, times)
    if kind == "none":
        return seq
    mc = getattr(settings, "hand_pose_smooth_min_cutoff", 2.0)
    bt = getattr(settings, "hand_pose_smooth_beta", 0.06)
    return _smooth_oneeuro(seq, times, mc, bt)


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
    left_world: List[Optional[List[List[float]]]] = []
    right_world: List[Optional[List[List[float]]]] = []
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
            from .progress import report as _progress
            total_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames_total += 1
                if total_est and idx % 90 == 0:
                    _progress(video_id, "hand tracking (MediaPipe)",
                              pct=100.0 * idx / total_est,
                              detail=f"frame {idx}/{total_est}")
                if idx % stride != 0:
                    idx += 1
                    continue

                frame = apply_rotation(frame, meta.rotation)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                left: Optional[List[List[float]]] = None
                right: Optional[List[List[float]]] = None
                lw: Optional[List[List[float]]] = None
                rw: Optional[List[List[float]]] = None
                lc: Optional[float] = None
                rc: Optional[float] = None

                if results.multi_hand_landmarks and results.multi_handedness:
                    world_lms = results.multi_hand_world_landmarks or []
                    for i, (lm, handed) in enumerate(
                        zip(results.multi_hand_landmarks, results.multi_handedness)
                    ):
                        cls = handed.classification[0]
                        label = cls.label  # "Left" / "Right"
                        if swap:
                            label = "Right" if label == "Left" else "Left"
                        coords = _landmarks_to_list(lm)
                        world = _landmarks_to_list(world_lms[i]) if i < len(world_lms) else None
                        if label == "Left":
                            left, lw, lc = coords, world, float(cls.score)
                        else:
                            right, rw, rc = coords, world, float(cls.score)

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
                left_world.append(lw)
                right_world.append(rw)
                left_conf.append(lc)
                right_conf.append(rc)

                frames_sampled += 1
                idx += 1
    finally:
        cap.release()

    # --- Temporal robustness for blurry/fast footage ---
    # Identity hygiene BEFORE filling/smoothing (bridges must use clean tracks):
    # 1) crossover pair-swaps (two-hand frames assigned by continuity),
    # 2) single-hand run-vote relabeling, 3) physics gate on teleporting wrists.
    handedness_fixes = swap_fixes = teleports = 0
    if getattr(settings, "hand_pose_handedness_continuity", True):
        swap_fixes = _fix_two_hand_swaps(
            left_lms, right_lms, left_world, right_world,
            left_conf, right_conf, timestamps)
        handedness_fixes = _fix_handedness_flips(
            left_lms, right_lms, left_world, right_world,
            left_conf, right_conf, timestamps)
        teleports = (_reject_teleports(left_lms, left_world, left_conf, timestamps)
                     + _reject_teleports(right_lms, right_world, right_conf, timestamps))

    max_gap = settings.hand_pose_gap_fill_seconds
    left_lms, left_filled = _temporal_fill(left_lms, timestamps, max_gap)
    right_lms, right_filled = _temporal_fill(right_lms, timestamps, max_gap)
    left_world, _ = _temporal_fill(left_world, timestamps, max_gap)
    right_world, _ = _temporal_fill(right_world, timestamps, max_gap)
    for i, f in enumerate(left_filled):
        if f and left_conf[i] is None:
            left_conf[i] = 0.2   # mark interpolated frames (low confidence)
    for i, f in enumerate(right_filled):
        if f and right_conf[i] is None:
            right_conf[i] = 0.2
    if settings.hand_pose_smooth:
        left_lms = _smooth(left_lms, timestamps)
        right_lms = _smooth(right_lms, timestamps)
        left_world = _smooth(left_world, timestamps)
        right_world = _smooth(right_world, timestamps)

    # Shape plausibility: score each frame's hand against the clip's median
    # bone lengths. Low scores mark occlusion-hallucinated frames (fingers
    # wrapped around an object) so consumers can mask or down-weight them.
    from .grasp import median_bone_lengths, shape_plausibility
    ref_l = median_bone_lengths(left_world)
    ref_r = median_bone_lengths(right_world)
    left_plaus = [shape_plausibility(w, ref_l) for w in left_world]
    right_plaus = [shape_plausibility(w, ref_r) for w in right_world]
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
            "left_world_landmarks": pa.array(left_world, type=_LANDMARK_TYPE),
            "right_world_landmarks": pa.array(right_world, type=_LANDMARK_TYPE),
            "left_confidence": pa.array(left_conf, type=pa.float32()),
            "right_confidence": pa.array(right_conf, type=pa.float32()),
            "left_shape_plausibility": pa.array(left_plaus, type=pa.float32()),
            "right_shape_plausibility": pa.array(right_plaus, type=pa.float32()),
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
        b"world_coord_order": b"x,y,z METERS, origin at the hand's geometric center "
                              b"(MediaPipe world landmarks; metric shape, not camera depth)",
        b"shape_plausibility": b"per-frame bone-length consistency vs the clip's median "
                               b"skeleton (1=consistent, low=occlusion-hallucinated joints)",
        b"source_fps": str(src_fps).encode(),
        b"sample_fps": str(round(effective_sample_fps, 4)).encode(),
        b"sample_stride": str(stride).encode(),
        b"handedness_swapped": str(swap).encode(),
        b"handedness_note": b"MediaPipe assumes a mirrored image; swapped for a forward-facing wearable cam",
        b"handedness_continuity_fixes": str(handedness_fixes).encode(),
        b"crossover_swap_fixes": str(swap_fixes).encode(),
        b"teleport_rejections": str(teleports).encode(),
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


# --------------------------------------------------------------------------- #
# WiLoR backend — transformer hand-mesh regression (MANO prior) running in the
# dedicated GPU env. Occlusion-robust (bone-consistency 0.994 vs MediaPipe's
# 0.711 on grasp frames of real footage) and, critically, it predicts the
# hand's ABSOLUTE metric position in camera space -> measured wrist depth.
# --------------------------------------------------------------------------- #

def _wilor_frame_to_sides(hands: List[dict], width: int, height: int,
                          f_true: Optional[float] = None):
    """Map one frame's WiLoR detections onto our (left, right) channels.

    Returns (left, right, left_world, right_world, lc, rc, l_cam, r_cam):
    image landmarks normalized (z=0 — WiLoR 2D is pixel-space; depth lives in
    the metric channels), world landmarks centered like MediaPipe's, wrist_cam
    the ABSOLUTE camera-frame wrist position (kp3d[0] + cam_t, meters).

    WiLoR's translation uses HaMeR's FIXED focal convention (5000/256 * long
    side ~= 37500px on 1080x1920) — depth scales linearly with focal, so we
    rescale by f_true/f_wilor to land in real meters (measured ~9m -> ~0.35m
    at the true 68-degree focal)."""
    out = {"L": (None, None, None, None), "R": (None, None, None, None)}
    for h in hands:
        side = "R" if float(h.get("is_right", 1.0)) >= 0.5 else "L"
        if out[side][0] is not None:
            continue  # keep the first detection per side
        kp2d = np.asarray(h["kp2d"], dtype=np.float64)
        kp3d = np.asarray(h["kp3d"], dtype=np.float64)
        if kp2d.shape != (21, 2) or kp3d.shape != (21, 3):
            continue
        img = np.zeros((21, 3))
        img[:, 0] = kp2d[:, 0] / max(1, width)
        img[:, 1] = kp2d[:, 1] / max(1, height)
        world = kp3d - kp3d.mean(axis=0)
        cam_t = np.asarray(h.get("cam_t", [0, 0, 0]), dtype=np.float64).reshape(3)
        wrist_cam = (kp3d[0] + cam_t)
        f_wilor = float(h.get("focal") or 0.0)
        if f_true and f_wilor > 1.0:
            wrist_cam = wrist_cam * (float(f_true) / f_wilor)
        out[side] = (img.round(5).tolist(), world.round(5).tolist(),
                     0.95, wrist_cam.round(4).tolist())
    L, LW, lc, lcam = out["L"]
    R, RW, rc, rcam = out["R"]
    return L, R, LW, RW, lc, rc, lcam, rcam


def extract_hand_pose_wilor(
    video_path: Path,
    output_path: Path,
    video_id: str,
    sample_fps: Optional[float] = None,
) -> HandPoseResult:
    """WiLoR-backed extraction: run the GPU env subprocess, convert to the
    standard parquet schema (+ wrist_cam columns), share the temporal pipeline
    (handedness run-vote, gap-fill, smoothing, plausibility)."""
    import subprocess
    import tempfile

    from .grasp import median_bone_lengths, shape_plausibility

    video_path, output_path = Path(video_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    wpy = Path(getattr(settings, "wilor_python", "") or
               repo / "data" / "tmp_wilor_env" / "Scripts" / "python.exe")
    runner = repo / "scripts" / "wilor_extract.py"
    if not wpy.exists():
        raise FileNotFoundError(f"WiLoR env python not found: {wpy}")
    fps = sample_fps or settings.hand_pose_sample_fps

    import os
    fd, tmp_name = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)  # Windows: an open fd blocks unlink with a sharing violation
    tmp = Path(tmp_name)
    # The GPU subprocess writes live progress straight into the clip's sidecar.
    from .progress import report as _report
    _report(video_id, "hand tracking (WiLoR, GPU)", pct=0, detail="starting GPU worker")
    progress_path = output_path.parent / "progress.json"
    try:
        r = subprocess.run([str(wpy), str(runner), str(video_path), str(tmp),
                            "--fps", str(fps),
                            "--progress", str(progress_path)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"wilor_extract failed: {r.stderr[-400:]}")
        lines = [json.loads(l) for l in tmp.read_text(encoding="utf-8").splitlines() if l.strip()]
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass  # leftover temp file is harmless; never mask the real result

    meta = lines[0]["meta"]
    width, height = int(meta["width"]), int(meta["height"])
    src_fps, stride = float(meta["src_fps"]), int(meta["stride"])

    frame_numbers, timestamps = [], []
    left_lms, right_lms, left_world, right_world = [], [], [], []
    left_conf, right_conf, l_cam, r_cam = [], [], [], []
    fov = float(getattr(settings, "ego_camera_fov_deg", 90.0))
    f_true = 0.5 * max(width, height) / np.tan(np.radians(fov) / 2.0)
    for row in lines[1:]:
        L, R, LW, RW, lc, rc, lcam, rcam = _wilor_frame_to_sides(
            row.get("hands", []), width, height, f_true=f_true)
        frame_numbers.append(int(row["frame"]))
        timestamps.append(float(row["ts"]))
        left_lms.append(L); right_lms.append(R)
        left_world.append(LW); right_world.append(RW)
        left_conf.append(lc); right_conf.append(rc)
        l_cam.append(lcam); r_cam.append(rcam)

    handedness_fixes = swap_fixes = teleports = 0
    if getattr(settings, "hand_pose_handedness_continuity", True):
        swap_fixes = _fix_two_hand_swaps(
            left_lms, right_lms, left_world, right_world,
            left_conf, right_conf, timestamps)
        handedness_fixes = _fix_handedness_flips(
            left_lms, right_lms, left_world, right_world,
            left_conf, right_conf, timestamps)
        teleports = (_reject_teleports(left_lms, left_world, left_conf, timestamps)
                     + _reject_teleports(right_lms, right_world, right_conf, timestamps))

    max_gap = settings.hand_pose_gap_fill_seconds
    left_lms, lf = _temporal_fill(left_lms, timestamps, max_gap)
    right_lms, rf = _temporal_fill(right_lms, timestamps, max_gap)
    left_world, _ = _temporal_fill(left_world, timestamps, max_gap)
    right_world, _ = _temporal_fill(right_world, timestamps, max_gap)
    for i, f in enumerate(lf):
        if f and left_conf[i] is None:
            left_conf[i] = 0.2
    for i, f in enumerate(rf):
        if f and right_conf[i] is None:
            right_conf[i] = 0.2
    if settings.hand_pose_smooth:
        left_lms = _smooth(left_lms, timestamps)
        right_lms = _smooth(right_lms, timestamps)
        left_world = _smooth(left_world, timestamps)
        right_world = _smooth(right_world, timestamps)

    ref_l = median_bone_lengths(left_world)
    ref_r = median_bone_lengths(right_world)
    left_plaus = [shape_plausibility(w, ref_l) for w in left_world]
    right_plaus = [shape_plausibility(w, ref_r) for w in right_world]

    frames_sampled = len(timestamps)
    frames_with_any = sum(1 for l, r in zip(left_lms, right_lms)
                          if l is not None or r is not None)
    left_count = sum(1 for l in left_lms if l is not None)
    right_count = sum(1 for r in right_lms if r is not None)

    _CAM_TYPE = pa.list_(pa.float32())
    table = pa.table({
        "frame_number": pa.array(frame_numbers, type=pa.int32()),
        "timestamp_ms": pa.array(timestamps, type=pa.float64()),
        "left_hand_landmarks": pa.array(left_lms, type=_LANDMARK_TYPE),
        "right_hand_landmarks": pa.array(right_lms, type=_LANDMARK_TYPE),
        "left_world_landmarks": pa.array(left_world, type=_LANDMARK_TYPE),
        "right_world_landmarks": pa.array(right_world, type=_LANDMARK_TYPE),
        "left_confidence": pa.array(left_conf, type=pa.float32()),
        "right_confidence": pa.array(right_conf, type=pa.float32()),
        "left_shape_plausibility": pa.array(left_plaus, type=pa.float32()),
        "right_shape_plausibility": pa.array(right_plaus, type=pa.float32()),
        "left_wrist_cam": pa.array(l_cam, type=_CAM_TYPE),
        "right_wrist_cam": pa.array(r_cam, type=_CAM_TYPE),
    })
    table = table.replace_schema_metadata({
        b"video_id": video_id.encode(),
        b"model": b"wilor",
        b"landmark_count": b"21",
        b"coord_order": b"x,y normalized image (z unused); world: METERS hand-centered",
        b"wrist_cam_note": b"ABSOLUTE camera-frame wrist xyz in METERS "
                           b"(WiLoR kp3d[0]+cam_t): measured depth",
        b"source_fps": str(src_fps).encode(),
        b"sample_fps": str(round(src_fps / stride, 4)).encode(),
        b"sample_stride": str(stride).encode(),
        b"handedness_continuity_fixes": str(handedness_fixes).encode(),
        b"crossover_swap_fixes": str(swap_fixes).encode(),
        b"teleport_rejections": str(teleports).encode(),
    })
    pq.write_table(table, output_path)

    return HandPoseResult(
        video_id=video_id, output_path=str(output_path),
        source_fps=src_fps, sample_fps=round(src_fps / stride, 4),
        sample_stride=stride, frames_total=frame_numbers[-1] + 1 if frame_numbers else 0,
        frames_sampled=frames_sampled, frames_with_any_hand=frames_with_any,
        left_hand_frames=left_count, right_hand_frames=right_count,
        coverage=round(frames_with_any / frames_sampled, 4) if frames_sampled else 0.0,
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
