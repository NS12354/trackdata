"""Grasp-state features from 21-point hand landmarks (Phase 2.6).

Manipulation data is only trainable if the action channel includes the hand's
grasp state — a wrist trajectory alone cannot teach a policy when to close the
gripper. This module derives that state from the MediaPipe 21-point landmarks
already captured in hand_pose.parquet:

  aperture       thumb-tip <-> index-tip distance in palm lengths (palm length =
                 wrist <-> middle-MCP). Scale-free: invariant to how far the
                 hand is from the camera.
  aperture_norm  aperture remapped to [0, 1]: 0 = pinched shut, 1 = fully open.
                 This is the gripper command channel in the LeRobot export.
  curl           mean curl of the four fingers, 0 = extended, 1 = fist.
  closed         convenience flag: aperture_norm <= CLOSED_THRESHOLD.

Inputs are image-normalized coordinates (x, y in [0,1], z relative depth). The
features are ratios *within* one hand, so camera-distance scaling cancels; the
residual image-aspect distortion (x and y normalized by different image sides)
is shared by numerator and denominator and largely cancels too.

Pure functions, no settings/IO — unit-testable without video.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

# MediaPipe Hands landmark indices used here.
_WRIST = 0
_THUMB_TIP = 4
_INDEX_TIP = 8
_MIDDLE_MCP = 9
# finger name -> (MCP index, tip index)
_FINGERS = {"index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}

# Calibration constants (in palm lengths / wrist-distance ratios). Chosen from
# MediaPipe geometry: a closed pinch sits ~0.4 palm lengths thumb-to-index, a
# fully spread hand ~1.6; an extended fingertip sits ~1.9x its MCP's wrist
# distance, a fully curled one ~1.0x.
APERTURE_CLOSED = 0.4
APERTURE_OPEN = 1.6
RATIO_EXTENDED = 1.9
RATIO_CURLED = 1.0
CLOSED_THRESHOLD = 0.3


def grasp_features(
    hand: Optional[List[List[float]]],
    world: Optional[List[List[float]]] = None,
) -> Optional[dict]:
    """Grasp state for one hand's 21x3 landmarks; None if absent/degenerate.

    ``hand`` is the image-normalized landmarks (always available when detected).
    ``world`` is the optional METRIC world landmarks; when present the result
    also carries ``aperture_m`` (thumb-tip<->index-tip distance in meters) and
    ``hand_scale_m`` (wrist<->middle-MCP palm length in meters) — real units a
    gripper can be commanded in, and the size cue for depth-from-scale.
    """
    if not hand or len(hand) < 21:
        return None
    p = np.asarray(hand, dtype=np.float64)
    wrist = p[_WRIST]
    scale = float(np.linalg.norm(p[_MIDDLE_MCP] - wrist))
    if scale < 1e-6:
        return None

    aperture = float(np.linalg.norm(p[_THUMB_TIP] - p[_INDEX_TIP])) / scale
    a_norm = (aperture - APERTURE_CLOSED) / (APERTURE_OPEN - APERTURE_CLOSED)
    a_norm = float(min(1.0, max(0.0, a_norm)))

    curls = []
    for mcp, tip in _FINGERS.values():
        mcp_d = float(np.linalg.norm(p[mcp] - wrist))
        if mcp_d < 1e-6:
            continue
        ratio = float(np.linalg.norm(p[tip] - wrist)) / mcp_d
        c = (RATIO_EXTENDED - ratio) / (RATIO_EXTENDED - RATIO_CURLED)
        curls.append(min(1.0, max(0.0, c)))
    curl = float(np.mean(curls)) if curls else 0.0

    out = {
        "aperture": round(aperture, 4),
        "aperture_norm": round(a_norm, 4),
        "curl": round(curl, 4),
        "closed": a_norm <= CLOSED_THRESHOLD,
    }
    if world and len(world) >= 21:
        w = np.asarray(world, dtype=np.float64)
        out["aperture_m"] = round(float(np.linalg.norm(w[_THUMB_TIP] - w[_INDEX_TIP])), 4)
        out["hand_scale_m"] = round(float(np.linalg.norm(w[_MIDDLE_MCP] - w[_WRIST])), 4)
    return out


def hand_scale_m(world: Optional[List[List[float]]]) -> Optional[float]:
    """Palm length (wrist<->middle-MCP) in meters from world landmarks."""
    if not world or len(world) < 21:
        return None
    w = np.asarray(world, dtype=np.float64)
    s = float(np.linalg.norm(w[_MIDDLE_MCP] - w[_WRIST]))
    return s if s > 1e-6 else None


# ---------------------------------------------------------------------------
# Shape plausibility — occlusion/hallucination detection.
#
# Landmark models hallucinate joints they can't see (fingers wrapped around an
# object — i.e. exactly during grasps). Real finger bones don't change length;
# hallucinated ones do. We score each frame's hand against the clip's median
# bone lengths: 1.0 = consistent skeleton, low = the model was guessing.
# ---------------------------------------------------------------------------
HAND_BONES: List[tuple] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
_PLAUSIBILITY_K = 0.4  # deviation scale: d=0.05 -> ~0.88, d=0.3 -> ~0.47


def bone_lengths(world: Optional[List[List[float]]]) -> Optional[np.ndarray]:
    """Per-bone lengths (meters) of one hand's world landmarks."""
    if not world or len(world) < 21:
        return None
    w = np.asarray(world, dtype=np.float64)
    return np.array([np.linalg.norm(w[a] - w[b]) for a, b in HAND_BONES])


def median_bone_lengths(world_seq: List[Optional[List[List[float]]]],
                        min_frames: int = 5) -> Optional[np.ndarray]:
    """The clip's reference skeleton: per-bone median over detected frames."""
    lens = [bl for w in world_seq if (bl := bone_lengths(w)) is not None]
    if len(lens) < min_frames:
        return None
    ref = np.median(np.stack(lens), axis=0)
    return ref if float(ref.min()) > 1e-6 else None


def shape_plausibility(world: Optional[List[List[float]]],
                       ref: Optional[np.ndarray]) -> Optional[float]:
    """[0,1] consistency of this frame's hand with the clip's reference
    skeleton; None when the hand or reference is unavailable."""
    bl = bone_lengths(world)
    if bl is None or ref is None:
        return None
    d = float(np.mean(np.abs(bl - ref) / ref))
    return round(float(np.exp(-d / _PLAUSIBILITY_K)), 4)
