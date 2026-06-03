#!/usr/bin/env python3
"""Generate a short synthetic test clip for verifying the pipeline.

Usage:
    python scripts/make_test_video.py [out.mp4] [seconds] [fps]

Note: the synthetic clip contains no real human faces, so anonymization will
honestly report ~0% face coverage on it. To see face blurring in action, upload
a real chest-worn clip. Use this clip to verify ingestion, status transitions,
encoding, and that the pipeline runs end-to-end.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def make_with_ffmpeg(out: Path, seconds: int, fps: int) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi",
        "-i", f"testsrc=duration={seconds}:size=640x480:rate={fps}",
        "-pix_fmt", "yuv420p", str(out),
    ]
    return subprocess.run(cmd).returncode == 0


def make_with_opencv(out: Path, seconds: int, fps: int) -> bool:
    w, h = 640, 480
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    total = seconds * fps
    for i in range(total):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (i % 255, (i * 2) % 255, (i * 3) % 255)
        x = int((i / total) * (w - 80))
        cv2.rectangle(frame, (x, 200), (x + 80, 280), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return True


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/uploads/_sample.mp4")
    seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    out.parent.mkdir(parents=True, exist_ok=True)
    if not make_with_ffmpeg(out, seconds, fps):
        make_with_opencv(out, seconds, fps)
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
