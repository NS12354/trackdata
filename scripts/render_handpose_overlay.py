#!/usr/bin/env python3
"""Render the hand-pose skeleton over an anonymized video, for demo/verification.

Usage:
    python scripts/render_handpose_overlay.py <video_id> [out.mp4]

Reads data/anonymized/<id>.mp4 + data/processed/<id>/hand_pose.parquet and writes
a video with the 21-point hand skeleton drawn on each frame (using the nearest
sampled keypoints, since pose is sampled at ~10fps and video is ~30fps).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import mediapipe as mp

# Make backend/ importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from pipeline.hand_pose import load_hand_pose, read_hand_pose_metadata  # noqa: E402

CONN = mp.solutions.hands.HAND_CONNECTIONS
COLORS = {"left": ((0, 255, 0), (0, 0, 255)), "right": ((255, 200, 0), (0, 0, 255))}


def _draw(frame, lms, bone, joint):
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _z in lms]
    for a, b in CONN:
        cv2.line(frame, pts[a], pts[b], bone, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 4, joint, -1, cv2.LINE_AA)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    vid = sys.argv[1]
    anon = ROOT / "data" / "anonymized" / f"{vid}.mp4"
    pq = ROOT / "data" / "processed" / vid / "hand_pose.parquet"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else (ROOT / "data" / "processed" / vid / "handpose_overlay.mp4")
    if not anon.exists() or not pq.exists():
        print(f"missing inputs: {anon.exists()=} {pq.exists()=}")
        sys.exit(1)

    rows = load_hand_pose(pq)
    meta = read_hand_pose_metadata(pq)
    stride = int(meta.get("sample_stride", "3"))
    by_frame = {r["frame_number"]: r for r in rows}

    cap = cv2.VideoCapture(str(anon))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Encode browser/QuickTime-friendly H.264 via ffmpeg if available.
    ff = None
    if shutil.which("ffmpeg"):
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-", "-an",
             "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(out)],
            stdin=subprocess.PIPE,
        )
        writer = None
    else:
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    idx = 0
    drawn = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Nearest sampled frame within half a stride.
        row = by_frame.get(idx)
        if row is None:
            best = None
            for d in range(1, stride):
                if idx - d in by_frame:
                    best = by_frame[idx - d]; break
                if idx + d in by_frame:
                    best = by_frame[idx + d]; break
            row = best
        if row:
            if row["left_hand_landmarks"]:
                _draw(frame, row["left_hand_landmarks"], *COLORS["left"])
                drawn += 1
            if row["right_hand_landmarks"]:
                _draw(frame, row["right_hand_landmarks"], *COLORS["right"])
                drawn += 1
        if ff:
            ff.stdin.write(frame.tobytes())
        else:
            writer.write(frame)
        idx += 1

    cap.release()
    if ff:
        ff.stdin.close(); ff.wait()
    else:
        writer.release()
    print(f"wrote {out} ({idx} frames, {drawn} hand draws)")


if __name__ == "__main__":
    main()
