"""Face detection backends for anonymization.

Privacy is non-negotiable, so we support running two independent detectors and
taking the union of their detections — a face missed by one may be caught by the
other:

  * YuNet (cv2.FaceDetectorYN) — a small, fast, accurate CNN detector. Primary.
  * MediaPipe Face Detection — secondary, for redundancy.

Each detector returns ``Detection`` records carrying a per-detector ``strong``
flag (score cleared that detector's confirmation threshold). The anonymizer's
track-confirmation then keeps any track with at least one strong detection from
*either* detector and drops the rest as false positives.

Detectors return RAW face boxes in original-frame pixel coordinates (no padding);
the caller pads. Each handles its own downscaling for speed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import settings

mp_face_detection = mp.solutions.face_detection

Box = Tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels


@dataclass
class Detection:
    box: Box
    score: float
    source: str    # "yunet" | "mediapipe"
    strong: bool   # cleared this detector's confirmation threshold


def _downscale(frame: np.ndarray, max_dim: int):
    h, w = frame.shape[:2]
    longest = max(h, w)
    if max_dim and longest > max_dim:
        scale = max_dim / longest
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return small, scale
    return frame, 1.0


class YuNetDetector:
    def __init__(self):
        self.confirm = settings.yunet_confirm_score
        self.max_dim = settings.face_detection_max_dim
        # Candidate threshold kept below the confirm threshold so weak frames of a
        # real face are still surfaced and rescued by a confirmed track.
        self.det = cv2.FaceDetectorYN_create(
            settings.yunet_model_path, "", (320, 320),
            settings.yunet_score_threshold, settings.yunet_nms_threshold, 5000,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        small, scale = _downscale(frame, self.max_dim)
        h, w = small.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(small)
        out: List[Detection] = []
        if faces is not None:
            for f in faces:
                x, y, bw, bh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                score = float(f[-1])
                box = (x / scale, y / scale, (x + bw) / scale, (y + bh) / scale)
                out.append(Detection(box, score, "yunet", score >= self.confirm))
        return out

    def close(self):
        pass


class MediaPipeDetector:
    def __init__(self):
        self.confirm = settings.face_confirm_confidence
        self.max_dim = settings.face_detection_max_dim
        self.fd = mp_face_detection.FaceDetection(
            model_selection=settings.face_detection_model,
            min_detection_confidence=settings.face_detection_confidence,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        small, _scale = _downscale(frame, self.max_dim)
        h, w = frame.shape[:2]  # map normalized coords onto the ORIGINAL frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self.fd.process(rgb)
        out: List[Detection] = []
        if results.detections:
            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                box = (bb.xmin * w, bb.ymin * h,
                       (bb.xmin + bb.width) * w, (bb.ymin + bb.height) * h)
                score = float(det.score[0]) if det.score else 0.0
                out.append(Detection(box, score, "mediapipe", score >= self.confirm))
        return out

    def close(self):
        self.fd.close()


class UnionDetector:
    """Runs multiple detectors and concatenates their detections."""

    def __init__(self, detectors):
        self.detectors = detectors

    def detect(self, frame: np.ndarray) -> List[Detection]:
        out: List[Detection] = []
        for d in self.detectors:
            out.extend(d.detect(frame))
        return out

    def close(self):
        for d in self.detectors:
            d.close()


def get_face_detector():
    """Build the configured detector(s)."""
    mode = settings.face_detector.lower()
    if mode == "yunet":
        return YuNetDetector()
    if mode == "mediapipe":
        return MediaPipeDetector()
    if mode == "union":
        return UnionDetector([YuNetDetector(), MediaPipeDetector()])
    raise ValueError(f"unknown face_detector {settings.face_detector!r}")
