"""Content-based orientation auto-correction.

Rotation metadata is sometimes missing or wrong, so a clip can end up sideways or
upside-down. Real footage with people has *upright faces*, so we sample frames,
try all four rotations, and pick the one where face detection is strongest. When
no faces are found (e.g. a chest-cam clip with none in frame), we fall back to the
metadata rotation — which already handles those.

The returned rotation is the degrees to rotate the decoded buffer to make it
upright (matching ``apply_rotation``).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2

from config import settings
from .video_meta import probe, VideoMeta, _ffprobe_rotation, _ffprobe_duration
from .face_detector import YuNetDetector

log = logging.getLogger("revisent.orientation")


def _rotate(frame, deg: int):
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def detect_orientation_by_faces(path: Path) -> Optional[int]:
    """Return the rotation (0/90/180/270) whose upright faces score highest, or
    None if no orientation has confident faces."""
    if not settings.auto_orient:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        n = max(1, settings.orientation_sample_frames)
        if total > 0:
            idxs = [int(i * total / (n + 1)) for i in range(1, n + 1)]
        else:
            idxs = list(range(n))
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if ok:
                frames.append(f)
    finally:
        cap.release()

    if not frames:
        return None

    detector = YuNetDetector()
    scores = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
    try:
        for f in frames:
            for r in (0, 90, 180, 270):
                for d in detector.detect(_rotate(f, r)):
                    scores[r] += d.score
    finally:
        detector.close()

    best = max(scores, key=scores.get)
    if scores[best] >= settings.orientation_min_score:
        log.info("orientation by faces: %s° (scores=%s)",
                 best, {k: round(v, 2) for k, v in scores.items()})
        return best
    return None


def resolve_video_meta(path: Path) -> VideoMeta:
    """Like ``probe`` but with content-based orientation auto-correction.

    Returns a VideoMeta whose ``rotation`` is the effective (corrected) rotation
    and whose width/height are the resulting display dimensions.
    """
    path = Path(path)
    # Raw buffer dimensions + fps/frame_count (probe with metadata rotation gives
    # display dims; re-derive raw to recompute for the effective rotation).
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    try:
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if fps <= 0:
        fps = 30.0

    meta_rotation = _ffprobe_rotation(path)
    content_rotation = detect_orientation_by_faces(path)
    effective = content_rotation if content_rotation is not None else meta_rotation
    if content_rotation is not None and content_rotation != meta_rotation:
        log.info("orientation: content=%s overrides metadata=%s for %s",
                 content_rotation, meta_rotation, path.name)

    width, height = (raw_h, raw_w) if effective in (90, 270) else (raw_w, raw_h)

    duration = _ffprobe_duration(path)
    if duration is None:
        duration = frame_count / fps if frame_count > 0 else 0.0
    elif frame_count <= 0:
        frame_count = int(round(duration * fps))

    return VideoMeta(
        width=width, height=height, fps=fps, frame_count=frame_count,
        duration_seconds=round(duration, 3), rotation=effective,
    )
