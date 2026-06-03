"""Video anonymization via face blurring.

v1 strategy (in the spec's order of preference):
  1. EgoBlur (Meta) — preferred, but its install pulls heavy PyTorch + detectron2
     dependencies and large model weights; deferred (see README "future work").
  2. *** MediaPipe Face Detection + OpenCV Gaussian blur — implemented here. ***
  3. Third-party API — not used; we keep processing local for privacy.

Anti-flicker: per-frame detection alone leaves a face un-blurred whenever the
detector misses (motion blur, profile, partially out of frame). Because this is
offline batch processing we run two passes:
  Pass 1 — detect faces on every frame and associate detections into tracks.
  Fill   — bridge short gaps within a track (interpolate) and hold the blur a few
           frames before/after the face is seen, dilating boxes to cover motion.
  Pass 2 — blur the gap-filled boxes and encode browser-friendly H.264.

This means a face momentarily turned away or at the frame edge stays blurred.

Output is re-encoded to H.264 (yuv420p, +faststart) via an ffmpeg pipe. Audio is
intentionally dropped in v1 (voices are PII we do not yet redact).
"""
from __future__ import annotations

import shutil  
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from config import settings
from .video_meta import probe, VideoMeta, apply_rotation
from .face_detector import get_face_detector

Box = Tuple[float, float, float, float]  # (x0, y0, x1, y1) in pixels


@dataclass
class AnonymizationResult:
    method: str
    frames_total: int
    frames_with_faces: int          # frames with a CONFIRMED face detection
    frames_blurred: int             # frames with >=1 box after gap-filling
    total_face_detections: int      # confirmed detections summed over frames
    tracks: int                     # confirmed face tracks
    rejected_tracks: int            # tracks dropped as false positives
    candidate_detections: int       # all raw detections before confirmation
    coverage: float                 # effective: fraction of frames blurred
    raw_detection_coverage: float   # fraction of frames with a confirmed detection
    mean_faces_per_frame: float
    fps: float
    width: int
    height: int
    duration_seconds: float
    output_codec: str

    def as_meta(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _center(b: Box) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _diag(b: Box) -> float:
    return ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5


def _dilate(b: Box, frac: float) -> Box:
    w, h = b[2] - b[0], b[3] - b[1]
    return (b[0] - w * frac, b[1] - h * frac, b[2] + w * frac, b[3] + h * frac)


def _lerp(a: Box, b: Box, t: float) -> Box:
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Blur
# --------------------------------------------------------------------------- #
def _blur_box(frame: np.ndarray, box: Box, strength: float) -> None:
    """In-place Gaussian blur of a (possibly fractional) box region."""
    h, w = frame.shape[:2]
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(w, int(box[2])), min(h, int(box[3]))
    if x1 <= x0 or y1 <= y0:
        return
    region = frame[y0:y1, x0:x1]
    k = int(max(x1 - x0, y1 - y0) * strength)
    k = max(9, k | 1)
    frame[y0:y1, x0:x1] = cv2.GaussianBlur(region, (k, k), 0)


# --------------------------------------------------------------------------- #
# Track building + gap filling
# --------------------------------------------------------------------------- #
def _build_filled_boxes(
    per_frame: List[List[Tuple[Box, bool]]],
    frames_total: int,
    max_gap: int,
    hold: int,
    dilation: float,
    match_iou: float,
) -> Tuple[List[List[Box]], dict]:
    """Associate detections into tracks, drop tracks that never produce a strong
    detection (false positives), then interpolate gaps and hold lead/tail frames
    for the confirmed tracks.

    Each detection is ``(box, strong)`` where ``strong`` means it cleared its
    detector's confirmation threshold. Returns (boxes_per_frame, stats).
    """
    # Greedy association across frames. Each track records whether it ever saw a
    # strong detection.
    tracks: List[Dict] = []  # {"boxes": {f: box}, "last_f", "last_box", "strong"}
    for f, dets in enumerate(per_frame):
        open_tracks = [t for t in tracks if f - t["last_f"] <= max_gap]
        used = set()
        for box, strong in dets:
            best, best_iou = None, 0.0
            for t in open_tracks:
                if id(t) in used:
                    continue
                iou = _iou(box, t["last_box"])
                if iou > best_iou:
                    best, best_iou = t, iou
            # Fall back to center proximity when boxes moved too far for IoU.
            if best is None or best_iou < match_iou:
                for t in open_tracks:
                    if id(t) in used:
                        continue
                    c1, c2 = _center(box), _center(t["last_box"])
                    dist = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
                    if dist < 0.6 * max(_diag(box), _diag(t["last_box"])):
                        best, best_iou = t, match_iou
                        break
            if best is not None and best_iou >= match_iou:
                best["boxes"][f] = box
                best["last_f"] = f
                best["last_box"] = box
                best["strong"] = best["strong"] or strong
                used.add(id(best))
            else:
                nt = {"boxes": {f: box}, "last_f": f, "last_box": box, "strong": strong}
                tracks.append(nt)
                used.add(id(nt))

    # Confirm tracks: a real face fires at least one strong detection; sporadic
    # false positives on hands/objects never clear the bar and are discarded.
    confirmed = [t for t in tracks if t["strong"]]
    rejected = len(tracks) - len(confirmed)

    out: List[List[Box]] = [[] for _ in range(frames_total)]
    confirmed_detection_frames = set()
    confirmed_detections = 0

    for t in confirmed:
        keyed = sorted(t["boxes"].items())
        confirmed_detections += len(keyed)
        for f, _b in keyed:
            confirmed_detection_frames.add(f)
        # Interpolate gaps between consecutive detected frames.
        for (fa, ba), (fb, bb) in zip(keyed, keyed[1:]):
            gap = fb - fa
            if gap <= 1 or gap > max_gap:
                if gap > max_gap:
                    continue  # genuine absence — do not bridge
            for f in range(fa + 1, fb):
                tt = (f - fa) / (fb - fa)
                box = _lerp(ba, bb, tt)
                # Dilate most in the middle of the gap (least certain there).
                grow = dilation * min(f - fa, fb - f)
                out[f].append(_dilate(box, min(grow, 0.5)))
        # Place the actual detections.
        for f, box in keyed:
            if 0 <= f < frames_total:
                out[f].append(box)
        # Lead/tail hold beyond the track's first/last detection.
        first_f, first_b = keyed[0]
        last_f, last_b = keyed[-1]
        for k in range(1, hold + 1):
            grow = min(dilation * k, 0.5)
            if first_f - k >= 0:
                out[first_f - k].append(_dilate(first_b, grow))
            if last_f + k < frames_total:
                out[last_f + k].append(_dilate(last_b, grow))

    stats = {
        "confirmed_tracks": len(confirmed),
        "rejected_tracks": rejected,
        "total_tracks": len(tracks),
        "confirmed_detections": confirmed_detections,
        "frames_with_confirmed_detection": len(confirmed_detection_frames),
    }
    return out, stats


# --------------------------------------------------------------------------- #
# ffmpeg writer
# --------------------------------------------------------------------------- #
def _open_ffmpeg_writer(out_path: Path, meta: VideoMeta) -> subprocess.Popen | None:
    if shutil.which("ffmpeg") is None:
        return None
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{meta.width}x{meta.height}",
        "-r", f"{meta.fps}",
        "-i", "-",
        "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def anonymize_video(input_path: Path, output_path: Path) -> AnonymizationResult:
    """Blur faces in ``input_path`` (temporally stable), writing ``output_path``."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = probe(input_path)
    pad = settings.face_box_padding
    strength = settings.blur_strength
    max_gap = max(1, int(round(settings.temporal_max_gap_seconds * meta.fps)))
    hold = max(0, int(round(settings.temporal_hold_seconds * meta.fps)))

    # ---- Pass 1: detect faces (configured detector / union) ----
    # Detect every Nth frame; skipped frames advance via the cheaper grab() and
    # are bridged by the temporal gap-fill. per_frame keeps one slot per source
    # frame so indices align with the blur pass.
    per_frame: List[List[Tuple[Box, bool]]] = []
    candidate_detections = 0
    stride = max(1, settings.face_detection_stride)
    detector = get_face_detector()
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        detector.close()
        raise ValueError(f"could not open input video: {input_path}")
    try:
        idx = 0
        while True:
            if idx % stride == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = apply_rotation(frame, meta.rotation)
                dets = detector.detect(frame)
                candidate_detections += len(dets)
                # Pad each raw face box outward, carry the per-detector strong flag.
                per_frame.append([(_dilate(d.box, pad), d.strong) for d in dets])
            else:
                if not cap.grab():  # advance without full decode/convert
                    break
                per_frame.append([])
            idx += 1
    finally:
        cap.release()
        detector.close()

    frames_total = len(per_frame)

    # ---- Confirm tracks + fill gaps / build per-frame blur boxes ----
    filled, stats = _build_filled_boxes(
        per_frame, frames_total, max_gap, hold,
        settings.track_dilation_per_frame, settings.track_match_iou,
    )
    frames_blurred = sum(1 for b in filled if b)
    frames_with_faces = stats["frames_with_confirmed_detection"]
    total_detections = stats["confirmed_detections"]

    # ---- Pass 2: blur + encode ----
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"could not reopen input video: {input_path}")
    ff = _open_ffmpeg_writer(output_path, meta)
    cv_writer = None
    output_codec = "h264"
    if ff is None:
        cv_writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
            meta.fps, (meta.width, meta.height),
        )
        output_codec = "mp4v"
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = apply_rotation(frame, meta.rotation)
            if idx < len(filled):
                for box in filled[idx]:
                    _blur_box(frame, box, strength)
            if ff is not None:
                ff.stdin.write(frame.tobytes())
            else:
                cv_writer.write(frame)
            idx += 1
    finally:
        cap.release()
        if ff is not None:
            ff.stdin.close()
            err = ff.stderr.read().decode("utf-8", "ignore") if ff.stderr else ""
            ret = ff.wait()
            if ret != 0:
                raise RuntimeError(f"ffmpeg encoding failed (code {ret}): {err.strip()[:500]}")
        if cv_writer is not None:
            cv_writer.release()

    coverage = (frames_blurred / frames_total) if frames_total else 0.0
    raw_cov = (frames_with_faces / frames_total) if frames_total else 0.0
    mean_faces = (total_detections / frames_total) if frames_total else 0.0

    return AnonymizationResult(
        method=f"{settings.face_detector}+opencv_blur+temporal_tracking+confirmation",
        frames_total=frames_total,
        frames_with_faces=frames_with_faces,
        frames_blurred=frames_blurred,
        total_face_detections=total_detections,
        tracks=stats["confirmed_tracks"],
        rejected_tracks=stats["rejected_tracks"],
        candidate_detections=candidate_detections,
        coverage=round(coverage, 4),
        raw_detection_coverage=round(raw_cov, 4),
        mean_faces_per_frame=round(mean_faces, 4),
        fps=meta.fps,
        width=meta.width,
        height=meta.height,
        duration_seconds=meta.duration_seconds,
        output_codec=output_codec,
    )
