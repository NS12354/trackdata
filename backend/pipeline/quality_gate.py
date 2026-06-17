"""Intake quality gate (Phase 0) — catch bad capture BEFORE it becomes bad data.

A clip that processes "successfully" can still be silently worthless: mounted
upside down (rotation metadata can't tell — iOS tags both portrait orientations
identically), too dark, or with hands never in frame. This gate samples a few
frames and reports problems up front.

Orientation, measured honestly (we tried the cheap ways first):
  * Faces — never present in ego footage. Useless here.
  * Hand-detection confidence across 4 rotations — MediaPipe's palm detector is
    largely rotation-tolerant; measured on real clips it's pure noise.
  * Hand anatomy (wrist-below-knuckles, lower-half position) — DEFEATED by grip
    style: two real clips showed opposite signatures in their true orientations
    (holding objects up to the camera vs. reaching). Kept as ADVISORY evidence
    in the report, never trusted to act.
  * The local VLM (qwen2.5vl via Ollama) — simply looks at sampled frames and
    says which way is up. Robust, free, ~5s/frame. THIS is the decision signal:
    a rotation is auto-applied only when all VLM votes agree (majority = warn).

The suggested rotation is the EXTRA rotation needed on top of the container's
rotation metadata (i.e. directly usable as process_clip --rotate). Vote
aggregation is pure (unit-testable without video/VLM); ``probe_clip`` touches
ffmpeg/MediaPipe/Ollama.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .video_meta import probe, apply_rotation

try:
    from config import settings
except Exception:  # pragma: no cover - allows import without app config
    settings = None

log = logging.getLogger("revisent.quality_gate")

ROTATIONS = (0, 90, 180, 270)
# Hand-evidence decision thresholds — ADVISORY ONLY (see module docstring):
# hand evidence is reported and warned on, never auto-applied.
MIN_SCORE = 1.5
MARGIN = 1.8
ANATOMY_WEIGHT_GOOD = 1.0
ANATOMY_WEIGHT_BAD = 0.15
_WRIST, _MIDDLE_MCP = 0, 9
# VLM orientation voting: frames asked, and how the answers map to the extra
# rotation that FIXES the frame. ROTATED_90_CW means the content's top points
# right -> rotating a further 90 CCW (=270 in our convention) corrects it.
VLM_VOTE_FRAMES = 3
_VLM_ANSWER_TO_FIX = {"UPRIGHT": 0, "UPSIDE_DOWN": 180,
                      "ROTATED_90_CW": 270, "ROTATED_90_CCW": 90}
_VLM_PROMPT = (
    "Look at this photo's orientation. Reply with EXACTLY one word from this "
    "list and nothing else: UPRIGHT, UPSIDE_DOWN, ROTATED_90_CW, ROTATED_90_CCW. "
    "UPSIDE_DOWN means the scene is flipped (floor/ground at the top, objects "
    "hanging). ROTATED_90_CW means the top of the scene points to the right. "
    "ROTATED_90_CCW means the top of the scene points to the left."
)

DARK_LUMA = 50.0      # mean 8-bit luma below this = "very dark"
BLUR_LAPLACIAN = 40.0  # Laplacian variance below this = "very blurry"


def _rotate_extra(frame: np.ndarray, deg: int) -> np.ndarray:
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def decide_rotation(scores: Dict[int, float],
                    min_score: float = MIN_SCORE,
                    margin: float = MARGIN) -> tuple[int, bool]:
    """Pick the winning extra-rotation from per-rotation hand scores.

    ADVISORY ONLY (hand evidence is grip-style-confounded; see module
    docstring). Returns (rotation_deg, decided). decided=False when evidence is
    weak or ambiguous.
    """
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best_rot, best), (_, second) = ranked[0], ranked[1]
    decided = best >= min_score and best >= margin * max(second, 1e-6)
    # No evidence at all -> keep 0 (don't "decide" a rotation from nothing).
    if best <= 0:
        return 0, False
    return int(best_rot), bool(decided)


def tally_votes(votes: List[int]) -> tuple[int, bool]:
    """Aggregate per-frame VLM orientation votes.

    Unanimous (>= 2 votes) -> (rotation, decided=True): safe to auto-apply.
    Majority -> (rotation, False): warn, don't act. Empty -> (0, False).
    """
    if not votes:
        return 0, False
    from collections import Counter
    rot, cnt = Counter(votes).most_common(1)[0]
    return int(rot), bool(cnt == len(votes) and len(votes) >= 2)


def _vlm_orientation_vote(jpeg: bytes, base_url: str, model: str,
                          timeout: int = 90) -> Optional[int]:
    """Ask the local VLM which way one frame is rotated; None on any failure."""
    import base64

    import requests

    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={"model": model, "prompt": _VLM_PROMPT,
                  "images": [base64.b64encode(jpeg).decode()],
                  "stream": False,
                  "options": {"num_ctx": 2048, "temperature": 0}},
            timeout=timeout,
        )
        r.raise_for_status()
        text = (r.json().get("response") or "").upper()
    except Exception as exc:  # noqa: BLE001 - VLM down: gate degrades to advisory
        log.warning("VLM orientation vote failed: %s", exc)
        return None
    # Check CCW before CW so neither substring shadows the other.
    for key in ("UPSIDE_DOWN", "ROTATED_90_CCW", "ROTATED_90_CW", "UPRIGHT"):
        if key in text:
            return _VLM_ANSWER_TO_FIX[key]
    log.warning("VLM orientation vote unparseable: %r", text[:120])
    return None


@dataclass
class QualityReport:
    duration_seconds: float
    width: int
    height: int
    fps: float
    frames_sampled: int
    mean_luma: float
    blur_laplacian_var: float
    hand_scores_by_rotation: Dict[int, float]  # ADVISORY (grip-style-confounded)
    vlm_votes: List[int]             # per-frame VLM verdicts (fix rotations)
    suggested_rotation: int          # extra deg on top of metadata rotation
    rotation_decided: bool           # unanimous VLM votes: safe to auto-apply
    warnings: List[str]

    def as_dict(self) -> dict:
        return asdict(self)


def probe_clip(video_path: Path, sample_frames: int = 8,
               detection_confidence: float = 0.4,
               use_vlm: bool = True) -> QualityReport:
    """Sample frames (metadata rotation applied) and grade the capture."""
    import mediapipe as mp  # heavy import; keep it function-local

    video_path = Path(video_path)
    meta = probe(video_path)
    duration = meta.duration_seconds or 0.0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = sorted({int(round(i)) for i in np.linspace(0, max(0, total - 1), sample_frames)})

    frames: List[np.ndarray] = []
    try:
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = apply_rotation(frame, meta.rotation)
            # Downscale: detection quality is fine at ~640 and 4x rotations are
            # 4x the work.
            h, w = frame.shape[:2]
            if max(h, w) > 640:
                s = 640 / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            frames.append(frame)
    finally:
        cap.release()

    # Photometrics on the sampled frames.
    lumas, lap_vars = [], []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        lumas.append(float(gray.mean()))
        lap_vars.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    mean_luma = float(np.mean(lumas)) if lumas else 0.0
    blur_var = float(np.median(lap_vars)) if lap_vars else 0.0

    # Hand evidence per rotation: detection score weighted by ANATOMY (wrist
    # below knuckles = hand entering from the wearer's body at the bottom).
    # Raw detection confidence alone is rotation-tolerant and non-discriminating.
    scores: Dict[int, float] = {r: 0.0 for r in ROTATIONS}
    with mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, model_complexity=0,
        min_detection_confidence=detection_confidence,
    ) as hands:
        for f in frames:
            for rot in ROTATIONS:
                rgb = cv2.cvtColor(_rotate_extra(f, rot), cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                if not (res.multi_hand_landmarks and res.multi_handedness):
                    continue
                for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                    conf = handed.classification[0].score
                    wrist_y = lm.landmark[_WRIST].y       # image y: down
                    mcp_y = lm.landmark[_MIDDLE_MCP].y
                    anatomical = wrist_y > mcp_y          # wrist nearer the bottom
                    scores[rot] += conf * (ANATOMY_WEIGHT_GOOD if anatomical
                                           else ANATOMY_WEIGHT_BAD)

    # Orientation DECISION: ask the local VLM to look at a few frames. Hand
    # evidence above is advisory only (see module docstring).
    votes: List[int] = []
    if use_vlm and frames:
        base_url = str(getattr(settings, "ollama_base_url", "http://localhost:11434")) \
            if settings is not None else "http://localhost:11434"
        model = str(getattr(settings, "ollama_vlm_model", "qwen2.5vl:3b")) \
            if settings is not None else "qwen2.5vl:3b"
        step = max(1, len(frames) // VLM_VOTE_FRAMES)
        for f in frames[::step][:VLM_VOTE_FRAMES]:
            ok_enc, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok_enc:
                continue
            v = _vlm_orientation_vote(buf.tobytes(), base_url, model)
            if v is not None:
                votes.append(v)
    suggested, decided = tally_votes(votes)

    hand_suggest, _ = decide_rotation(scores)

    warnings: List[str] = []
    if decided and suggested != 0:
        warnings.append(
            f"clip is rotated {suggested} deg (VLM votes {votes}) — apply --rotate {suggested}"
        )
    elif votes and suggested != 0:
        warnings.append(
            f"orientation suspect: VLM votes {votes} lean {suggested} deg but are "
            "not unanimous — verify visually"
        )
    elif not votes and hand_suggest != 0 and scores[hand_suggest] > scores.get(0, 0.0):
        warnings.append(
            f"orientation unverified (VLM unavailable); hand evidence weakly leans "
            f"{hand_suggest} deg — verify visually"
        )
    if mean_luma < DARK_LUMA:
        warnings.append(f"very dark footage (mean luma {mean_luma:.0f} < {DARK_LUMA:.0f})")
    if blur_var < BLUR_LAPLACIAN and frames:
        warnings.append(f"very soft/blurry footage (Laplacian var {blur_var:.0f})")
    if max(scores.values() or [0]) <= 0:
        warnings.append("no hands detected in sampled frames — manipulation value doubtful")

    return QualityReport(
        duration_seconds=duration,
        width=meta.width, height=meta.height, fps=meta.fps or 0.0,
        frames_sampled=len(frames),
        mean_luma=round(mean_luma, 1),
        blur_laplacian_var=round(blur_var, 1),
        hand_scores_by_rotation={int(k): round(v, 3) for k, v in scores.items()},
        vlm_votes=votes,
        suggested_rotation=suggested,
        rotation_decided=decided,
        warnings=warnings,
    )
