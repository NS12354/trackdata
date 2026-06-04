"""Head-pose estimation via monocular visual odometry (Stage 1 of ego-body pose).

The head camera is rigidly attached to the head, so tracking the camera's motion
through the scene *measures* head pose. This is genuine measurement — but
monocular, so it is up-to-scale and accumulates drift over time. It is the
foundation the ego-body model builds on (head pose + hands -> inferred body).

Method: ORB features matched between sampled frames -> essential matrix ->
recoverPose -> accumulate a global rotation + (unit-scale) translation.

Output: per-frame {timestamp_ms, position[x,y,z], quaternion[w,x,y,z], tracked}.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from config import settings
from .orientation import resolve_video_meta, _rotate  # _rotate to apply orientation

log = logging.getLogger("revisent.ego_pose")


@dataclass
class HeadPose:
    timestamp_ms: float
    position: list   # [x, y, z], up-to-scale (drifts)
    quaternion: list  # [w, x, y, z] world orientation
    tracked: bool     # whether VO got a valid estimate for this step


def _intrinsics(w: int, h: int, fov_deg: float) -> np.ndarray:
    f = 0.5 * w / np.tan(np.radians(fov_deg) / 2.0)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)


def _rotmat_to_quat(R: np.ndarray) -> list:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return [float(w), float(x), float(y), float(z)]


def estimate_head_trajectory(video_path: Path, sample_fps: Optional[float] = None) -> List[HeadPose]:
    video_path = Path(video_path)
    meta = resolve_video_meta(video_path)
    src_fps = meta.fps or 30.0
    sample_fps = sample_fps or settings.ego_vo_sample_fps
    stride = max(1, int(round(src_fps / sample_fps)))
    K = _intrinsics(meta.width, meta.height, settings.ego_camera_fov_deg)

    orb = cv2.ORB_create(settings.ego_vo_orb_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    R_w = np.eye(3)
    t_w = np.zeros((3, 1))
    prev_kp = prev_des = None

    poses: List[HeadPose] = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                frame = _rotate(frame, meta.rotation)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kp, des = orb.detectAndCompute(gray, None)
                tracked = False
                if prev_des is not None and des is not None and len(kp) >= 8:
                    matches = bf.knnMatch(prev_des, des, k=2)
                    good = [m for pair in matches if len(pair) == 2
                            for m, n in [pair] if m.distance < 0.75 * n.distance]
                    if len(good) >= 12:
                        p1 = np.float32([prev_kp[m.queryIdx].pt for m in good])
                        p2 = np.float32([kp[m.trainIdx].pt for m in good])
                        E, mask = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
                        if E is not None and E.shape == (3, 3):
                            _, R, t, _ = cv2.recoverPose(E, p1, p2, K)
                            # Accumulate global pose (unit translation scale).
                            t_w = t_w + R_w @ t
                            R_w = R @ R_w
                            tracked = True
                poses.append(HeadPose(
                    timestamp_ms=round(idx / src_fps * 1000.0, 2),
                    position=[round(float(v), 4) for v in t_w.flatten()],
                    quaternion=[round(v, 5) for v in _rotmat_to_quat(R_w)],
                    tracked=tracked,
                ))
                prev_kp, prev_des = kp, des
            idx += 1
    finally:
        cap.release()

    n_tracked = sum(1 for p in poses if p.tracked)
    log.info("head VO %s: %d poses, %d tracked (%.0f%%)",
             video_path.name, len(poses), n_tracked,
             100 * n_tracked / max(1, len(poses)))
    return poses


def trajectory_to_json(video_id: str, poses: List[HeadPose]) -> dict:
    return {
        "video_id": video_id,
        "method": "monocular_visual_odometry_orb",
        "note": "up-to-scale, drifts over time; head orientation is most reliable",
        "frames": [asdict(p) for p in poses],
    }
