"""Scene-text detection (EAST) for PII blurring.

Finds text regions in a frame — unit/apartment numbers, mail & package labels,
whiteboards, license plates, on-screen text — so the anonymizer can blur them.
We don't read the text, only locate it; high recall is the goal (blurring a bit
extra is safe; leaking PII is not).

Model: OpenCV's EAST detector (frozen_east_text_detection.pb). CPU-friendly,
integrates like the YuNet face detector.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from config import settings

log = logging.getLogger("revisent.text_detector")

Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2) pixels
_OUTPUTS = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]


def _round32(v: int) -> int:
    return max(32, int(round(v / 32.0)) * 32)


class EASTTextDetector:
    """Detect axis-aligned bounding boxes around text regions."""

    def __init__(self):
        model = Path(settings.east_model_path)
        if not model.exists():
            raise FileNotFoundError(
                f"EAST model missing at {model}. Download frozen_east_text_detection.pb."
            )
        self.net = cv2.dnn.readNet(str(model))
        self.score_thr = settings.text_score_threshold
        self.nms_thr = settings.text_nms_threshold
        self.work = _round32(settings.text_detection_size)

    def detect(self, frame: np.ndarray) -> List[Box]:
        H, W = frame.shape[:2]
        rW, rH = W / float(self.work), H / float(self.work)
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (self.work, self.work),
            (123.68, 116.78, 103.94), swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        scores, geometry = self.net.forward(_OUTPUTS)
        rects, confs = self._decode(scores, geometry)
        if not rects:
            return []
        # NMS on axis-aligned boxes (x, y, w, h)
        idxs = cv2.dnn.NMSBoxes(
            [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)] for (x1, y1, x2, y2) in rects],
            confs, self.score_thr, self.nms_thr,
        )
        out: List[Box] = []
        if len(idxs) > 0:
            for i in np.array(idxs).flatten():
                x1, y1, x2, y2 = rects[i]
                out.append((x1 * rW, y1 * rH, x2 * rW, y2 * rH))  # scale back to full res
        return out

    def _decode(self, scores: np.ndarray, geometry: np.ndarray):
        """Standard EAST decode -> axis-aligned boxes (in `work` resolution)."""
        nrows, ncols = scores.shape[2], scores.shape[3]
        rects: List[Box] = []
        confs: List[float] = []
        for y in range(nrows):
            s = scores[0, 0, y]
            d0, d1, d2, d3 = geometry[0, 0, y], geometry[0, 1, y], geometry[0, 2, y], geometry[0, 3, y]
            angles = geometry[0, 4, y]
            for x in range(ncols):
                if s[x] < self.score_thr:
                    continue
                ox, oy = x * 4.0, y * 4.0
                ang = angles[x]
                cos, sin = np.cos(ang), np.sin(ang)
                h = d0[x] + d2[x]
                w = d1[x] + d3[x]
                ex = ox + cos * d1[x] + sin * d2[x]
                ey = oy - sin * d1[x] + cos * d2[x]
                rects.append((ex - w, ey - h, ex, ey))
                confs.append(float(s[x]))
        return rects, confs

    def close(self):
        pass


_singleton = None


def get_text_detector() -> EASTTextDetector:
    global _singleton
    if _singleton is None:
        _singleton = EASTTextDetector()
    return _singleton
