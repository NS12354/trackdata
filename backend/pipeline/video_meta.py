"""Probe basic video metadata (fps, dimensions, frame count, duration)."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoMeta:
    # width/height are the *display* dimensions (already swapped if the clip is
    # rotated 90/270°), so downstream code can decode -> rotate -> use these.
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    rotation: int = 0  # display rotation in degrees, normalized to {0,90,180,270}


def _ffprobe_duration(path: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return None


def _ffprobe_rotation(path: Path) -> int:
    """Return the clip's display rotation in degrees ({0,90,180,270}).

    Phone videos store an orientation flag (legacy `rotate` tag and/or a Display
    Matrix side-data `rotation`) that players honor but OpenCV ignores. We read it
    here so the pipeline can rotate decoded frames to be upright.
    """
    if shutil.which("ffprobe") is None:
        return 0
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream_tags=rotate:stream_side_data=rotation",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(out.stdout).get("streams", [])
        if not streams:
            return 0
        st = streams[0]
        rot = 0
        tag = st.get("tags", {}).get("rotate")
        if tag is not None:
            rot = int(tag)
        for sd in st.get("side_data_list", []):
            if "rotation" in sd:
                rot = int(sd["rotation"])
        return rot % 360
    except Exception:
        return 0


def probe(path: Path) -> VideoMeta:
    """Read metadata via OpenCV, cross-checking duration with ffprobe.

    OpenCV's frame count can be unreliable for some codecs, so we prefer
    ffprobe's duration when available and derive frame_count from it. Rotation is
    read from ffprobe and the returned width/height are the *display* dimensions.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    if fps <= 0:
        fps = 30.0  # sane default if codec doesn't report it

    rotation = _ffprobe_rotation(path)
    # For a quarter-turn the display dimensions are swapped relative to the buffer.
    if rotation in (90, 270):
        width, height = height, width

    duration = _ffprobe_duration(path)
    if duration is None:
        duration = frame_count / fps if frame_count > 0 else 0.0
    elif frame_count <= 0:
        frame_count = int(round(duration * fps))

    return VideoMeta(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=round(duration, 3),
        rotation=rotation,
    )


def apply_rotation(frame, rotation: int):
    """Rotate a decoded frame to upright per the clip's display rotation."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame
