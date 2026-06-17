"""Hand-pose providers — pluggable per-frame hand detectors.

Both return the SAME normalized 21-landmark schema (x,y in [0,1], z relative), so
the rest of the pipeline (temporal fill, smoothing, Parquet, overlay, body-pose
fusion, LeRobot export) is identical regardless of model.

  * MediaPipeHandProvider — local MediaPipe Hands (default, free).
  * WiLoRReplicateProvider — lab-grade egocentric model via a Replicate GPU
    endpoint. Paid; sends frames to Replicate; per-frame API calls (slow), so use
    a short clip / low fps / hand_pose_eval_max_frames for evaluation.

Each is a context manager: open() on __enter__, release on __exit__, detect(rgb)
per frame returning FrameHands.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import requests

from config import settings


@dataclass
class FrameHands:
    left: Optional[List[List[float]]] = None        # 21x3 normalized
    left_conf: Optional[float] = None
    right: Optional[List[List[float]]] = None
    right_conf: Optional[float] = None


class MediaPipeHandProvider:
    name = "mediapipe_hands"
    handedness_note = "MediaPipe assumes a mirrored image; swapped for forward-facing chest cam"

    def __init__(self):
        import mediapipe as mp
        self._mp = mp.solutions.hands
        self._hands = None
        self.swap = settings.hand_pose_swap_handedness

    def __enter__(self):
        self._hands = self._mp.Hands(
            static_image_mode=False,
            max_num_hands=settings.hand_pose_max_hands,
            model_complexity=settings.hand_pose_model_complexity,
            min_detection_confidence=settings.hand_pose_min_detection_confidence,
            min_tracking_confidence=settings.hand_pose_min_tracking_confidence,
        )
        return self

    def __exit__(self, *exc):
        if self._hands is not None:
            self._hands.close()

    def detect(self, rgb: np.ndarray) -> FrameHands:
        results = self._hands.process(rgb)
        fh = FrameHands()
        if results.multi_hand_landmarks and results.multi_handedness:
            for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                cls = handed.classification[0]
                label = cls.label  # "Left" / "Right"
                if self.swap:
                    label = "Right" if label == "Left" else "Left"
                coords = [[p.x, p.y, p.z] for p in lm.landmark]
                if label == "Left":
                    fh.left, fh.left_conf = coords, float(cls.score)
                else:
                    fh.right, fh.right_conf = coords, float(cls.score)
        return fh


# WiLoR/MANO 21-keypoint order -> MediaPipe order. OpenPose/MANO hand keypoint
# order matches MediaPipe (0 wrist; 1-4 thumb; 5-8 index; 9-12 middle; 13-16 ring;
# 17-20 pinky), so identity is the right default — but VERIFY against the first
# real Replicate response (overlay will look scrambled if the order differs).
WILOR_TO_MP = list(range(21))


def _as_hand_list(output) -> list:
    """Normalize Replicate output into a list of per-hand records, defensively."""
    if output is None:
        return []
    if isinstance(output, dict):
        for key in ("hands", "predictions", "detections", "results"):
            if key in output and isinstance(output[key], list):
                return output[key]
        return [output]
    if isinstance(output, list):
        return output
    return []


def _extract_keypoints_2d(hand) -> Optional[list]:
    """Pull a 21x2 pixel-keypoint list out of a hand record, trying common keys."""
    if isinstance(hand, list):
        kps = hand
    elif isinstance(hand, dict):
        kps = None
        for key in ("keypoints_2d", "joints_2d", "keypoints", "landmarks_2d",
                    "keypoints2d", "pred_keypoints_2d"):
            if key in hand and hand[key]:
                kps = hand[key]
                break
        if kps is None:
            return None
    else:
        return None
    out = []
    for p in kps:
        if isinstance(p, dict):
            out.append([p.get("x"), p.get("y")])
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append([p[0], p[1]])
    return out if len(out) >= 21 else None


def _is_right(hand) -> bool:
    if not isinstance(hand, dict):
        return True
    for key in ("is_right", "right"):
        if key in hand:
            return bool(hand[key])
    side = str(hand.get("hand_side", hand.get("handedness", hand.get("side", "right")))).lower()
    return "right" in side or side in ("r", "1")


class WiLoRReplicateProvider:
    name = "wilor_replicate"
    handedness_note = "WiLoR returns anatomical left/right (no mirror swap)"
    _CREATE = "https://api.replicate.com/v1/predictions"

    def __init__(self):
        if not settings.replicate_api_token:
            raise RuntimeError("hand_pose_provider=wilor requires REPLICATE_API_TOKEN")
        if not settings.wilor_replicate_version:
            raise RuntimeError("hand_pose_provider=wilor requires WILOR_REPLICATE_VERSION "
                               "(the model version hash from its Replicate page)")
        self.token = settings.replicate_api_token
        self.version = settings.wilor_replicate_version

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def detect(self, rgb: np.ndarray) -> FrameHands:
        h, w = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        data_uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        output = self._predict(data_uri)
        return self._parse(output, w, h)

    def _predict(self, image: str):
        headers = {"Authorization": f"Token {self.token}", "Content-Type": "application/json"}
        r = requests.post(self._CREATE, headers=headers,
                          json={"version": self.version, "input": {"image": image}}, timeout=60)
        r.raise_for_status()
        pred = r.json()
        get_url = pred.get("urls", {}).get("get")
        for _ in range(180):
            status = pred.get("status")
            if status == "succeeded":
                return pred.get("output")
            if status in ("failed", "canceled"):
                raise RuntimeError(f"replicate {status}: {pred.get('error')}")
            time.sleep(1)
            pred = requests.get(get_url, headers=headers, timeout=30).json()
        raise RuntimeError("replicate prediction timed out")

    def _parse(self, output, w: int, h: int) -> FrameHands:
        fh = FrameHands()
        for hand in _as_hand_list(output):
            kps = _extract_keypoints_2d(hand)
            if not kps:
                continue
            if len(kps) >= 21:
                kps = [kps[i] for i in WILOR_TO_MP]
            # Normalize to [0,1]; z=0 (relative depth not recovered from 2D kps).
            coords = [[float(x) / w, float(y) / h, 0.0] for x, y in kps[:21]]
            score = 0.9
            if isinstance(hand, dict):
                score = float(hand.get("score", hand.get("confidence", 0.9)))
            if _is_right(hand):
                fh.right, fh.right_conf = coords, score
            else:
                fh.left, fh.left_conf = coords, score
        return fh


def get_hand_pose_provider():
    p = settings.hand_pose_provider.lower()
    if p == "mediapipe":
        return MediaPipeHandProvider()
    if p == "wilor":
        return WiLoRReplicateProvider()
    raise ValueError(f"unknown hand_pose_provider {settings.hand_pose_provider!r}")
