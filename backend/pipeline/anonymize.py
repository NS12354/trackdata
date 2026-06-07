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

import functools
import logging
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import settings
from .video_meta import VideoMeta, apply_rotation
from .orientation import resolve_video_meta
from .face_detector import get_face_detector

log = logging.getLogger("revisent.anonymize")

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
# Hardware (Apple VideoToolbox) vs software video I/O
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _hw_video_available() -> bool:
    """True if ffmpeg can do VideoToolbox hardware decode + H.264 encode."""
    mode = settings.use_hardware_video.lower()
    if mode == "off" or shutil.which("ffmpeg") is None:
        return False
    if mode == "on":
        return True
    try:
        enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=10).stdout
        hw = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"],
                            capture_output=True, text=True, timeout=10).stdout
        return "h264_videotoolbox" in enc and "videotoolbox" in hw
    except Exception:
        return False


def _rotation_vf(deg: int) -> Optional[str]:
    """ffmpeg filter to rotate raw frames upright — matches ``apply_rotation``."""
    if deg == 90:
        return "transpose=1"                 # 90° clockwise
    if deg == 180:
        return "transpose=2,transpose=2"     # 180°
    if deg == 270:
        return "transpose=2"                 # 90° counter-clockwise
    return None


class _SwReader:
    """OpenCV software decode; rotation applied in Python."""

    def __init__(self, path: Path, meta: VideoMeta):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise ValueError(f"could not open video: {path}")
        self.rot = meta.rotation

    def read(self):
        ok, f = self.cap.read()
        return apply_rotation(f, self.rot) if ok else None

    def grab(self) -> bool:
        return self.cap.grab()

    def close(self):
        self.cap.release()


class _HwReader:
    """VideoToolbox hardware decode via an ffmpeg pipe; rotation baked into the
    filter graph (``-noautorotate`` so it matches the software path)."""

    def __init__(self, path: Path, meta: VideoMeta):
        self.w, self.h = meta.width, meta.height
        self.nbytes = self.w * self.h * 3
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-noautorotate", "-hwaccel", "videotoolbox", "-i", str(path)]
        vf = _rotation_vf(meta.rotation)
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL)

    def _read_exact(self):
        buf = bytearray()
        while len(buf) < self.nbytes:
            chunk = self.proc.stdout.read(self.nbytes - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def read(self):
        buf = self._read_exact()
        if buf is None:
            return None
        # Writable copy so the blur pass can modify it in place.
        return np.frombuffer(bytes(buf), np.uint8).reshape(self.h, self.w, 3).copy()

    def grab(self) -> bool:
        return self._read_exact() is not None

    def close(self):
        try:
            self.proc.stdout.close()
        except Exception:
            pass
        self.proc.wait()


def _make_reader(path: Path, meta: VideoMeta):
    if _hw_video_available():
        try:
            return _HwReader(path, meta)
        except Exception:
            pass
    return _SwReader(path, meta)


def _target_bitrate(meta: VideoMeta) -> str:
    bps = int(meta.width * meta.height * (meta.fps or 30.0) * 0.1)
    bps = max(2_000_000, min(bps, 12_000_000))
    return f"{bps // 1000}k"


def _open_ffmpeg_writer(out_path: Path, meta: VideoMeta):
    """Return (Popen, codec_name) or (None, 'mp4v'). Hardware H.264 when available."""
    if shutil.which("ffmpeg") is None:
        return None, "mp4v"
    if _hw_video_available():
        vcodec = ["-c:v", "h264_videotoolbox", "-b:v", _target_bitrate(meta), "-pix_fmt", "yuv420p"]
        codec = "h264_videotoolbox"
    else:
        vcodec = ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
        codec = "h264"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{meta.width}x{meta.height}", "-r", f"{meta.fps}", "-i", "-",
        "-an", *vcodec, "-movflags", "+faststart", str(out_path),
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE), codec
    except Exception:
        return None, "mp4v"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def anonymize_video(input_path: Path, output_path: Path) -> AnonymizationResult:
    """Blur faces in ``input_path`` (temporally stable), writing ``output_path``."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Content-aware orientation: corrects sideways/upside-down clips whose rotation
    # metadata is missing or wrong, falling back to metadata when no faces.
    meta = resolve_video_meta(input_path)
    pad = settings.face_box_padding
    strength = settings.blur_strength
    max_gap = max(1, int(round(settings.temporal_max_gap_seconds * meta.fps)))
    hold = max(0, int(round(settings.temporal_hold_seconds * meta.fps)))

    # ---- Pass 1: detect faces (configured detector / union) ----
    # Detect every Nth frame; skipped frames advance via the cheaper grab() and
    # are bridged by the temporal gap-fill. per_frame keeps one slot per source
    # frame so indices align with the blur pass.
    per_frame: List[List[Tuple[Box, bool]]] = []
    text_per_frame: List[List[Box]] = []
    candidate_detections = 0
    text_box_total = 0
    stride = max(1, settings.face_detection_stride)
    detector = get_face_detector()
    # Optional EAST text/PII detector (unit numbers, mail/labels, plates, screens,
    # whiteboards). High-recall: we locate text and blur it generously.
    text_detector = None
    if settings.enable_text_blur:
        try:
            from .text_detector import get_text_detector
            text_detector = get_text_detector()
        except Exception as exc:  # noqa: BLE001
            log.warning("text/PII blur disabled (EAST unavailable): %s", exc)
    text_pad = settings.text_box_padding
    text_every = max(1, round(settings.text_detection_stride / stride))  # decoded frames
    current_text: List[Box] = []
    decoded = 0
    reader = _make_reader(input_path, meta)  # hardware decode when available
    try:
        idx = 0
        while True:
            if idx % stride == 0:
                frame = reader.read()  # upright BGR
                if frame is None:
                    break
                dets = detector.detect(frame)
                candidate_detections += len(dets)
                # Pad each raw face box outward, carry the per-detector strong flag.
                per_frame.append([(_dilate(d.box, pad), d.strong) for d in dets])
                # Text is near-static frame-to-frame: detect every Nth decoded frame
                # and carry the boxes forward across the skipped frames.
                if text_detector is not None and decoded % text_every == 0:
                    current_text = [_dilate(b, text_pad) for b in text_detector.detect(frame)]
                    text_box_total += len(current_text)
                decoded += 1
            else:
                if not reader.grab():  # advance without running detection
                    break
                per_frame.append([])
            text_per_frame.append(list(current_text))
            idx += 1
    finally:
        reader.close()
        detector.close()
        if text_detector is not None:
            text_detector.close()

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
    reader = _make_reader(input_path, meta)
    ff, output_codec = _open_ffmpeg_writer(output_path, meta)
    cv_writer = None
    if ff is None:
        cv_writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
            meta.fps, (meta.width, meta.height),
        )
        output_codec = "mp4v"
    try:
        idx = 0
        while True:
            frame = reader.read()
            if frame is None:
                break
            if idx < len(filled):
                for box in filled[idx]:
                    _blur_box(frame, box, strength)
            if idx < len(text_per_frame):
                for box in text_per_frame[idx]:
                    _blur_box(frame, box, strength)
            if ff is not None:
                ff.stdin.write(frame.tobytes())
            else:
                cv_writer.write(frame)
            idx += 1
    finally:
        reader.close()
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
        method=(f"{settings.face_detector}+opencv_blur+temporal_tracking+confirmation"
                + ("+east_text_pii" if settings.enable_text_blur else "")),
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
