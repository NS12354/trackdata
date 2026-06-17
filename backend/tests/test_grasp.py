"""Grasp-feature tests: synthetic open hand vs fist landmarks produce the
expected aperture / curl / closed values. Pure numeric — no video needed.

Run from backend/:  python tests/test_grasp.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.grasp import grasp_features, CLOSED_THRESHOLD  # noqa: E402


def _hand(points: dict):
    """21x3 landmark list with the indices grasp_features uses set explicitly."""
    lm = [[0.0, 0.0, 0.0] for _ in range(21)]
    for i, (x, y) in points.items():
        lm[i] = [x, y, 0.0]
    return lm


def open_hand():
    """Spread hand: thumb far from index tip, fingertips extended above MCPs."""
    return _hand({
        0: (0.50, 0.80),                    # wrist
        4: (0.32, 0.62),                    # thumb tip
        5: (0.46, 0.65), 8: (0.42, 0.42),   # index mcp / tip
        9: (0.50, 0.65), 12: (0.50, 0.42),  # middle mcp / tip
        13: (0.54, 0.66), 16: (0.55, 0.44),  # ring mcp / tip
        17: (0.58, 0.67), 20: (0.59, 0.48),  # pinky mcp / tip
    })


def fist():
    """Closed fist: thumb tip touching index tip, fingertips curled to the palm."""
    return _hand({
        0: (0.50, 0.80),
        4: (0.50, 0.70),
        5: (0.46, 0.65), 8: (0.49, 0.69),
        9: (0.50, 0.65), 12: (0.50, 0.69),
        13: (0.54, 0.66), 16: (0.52, 0.70),
        17: (0.58, 0.67), 20: (0.54, 0.71),
    })


def test_open_hand():
    g = grasp_features(open_hand())
    assert g is not None
    assert g["aperture_norm"] > 0.6, f"open hand should be open: {g}"
    assert g["curl"] < 0.3, f"open hand should not be curled: {g}"
    assert g["closed"] is False
    print(f"ok: open hand aperture_norm={g['aperture_norm']} curl={g['curl']}")


def test_fist():
    g = grasp_features(fist())
    assert g is not None
    assert g["aperture_norm"] <= CLOSED_THRESHOLD, f"fist should be closed: {g}"
    assert g["curl"] > 0.7, f"fist should be curled: {g}"
    assert g["closed"] is True
    print(f"ok: fist aperture_norm={g['aperture_norm']} curl={g['curl']}")


def test_monotonic():
    assert grasp_features(open_hand())["aperture"] > grasp_features(fist())["aperture"]
    print("ok: aperture orders open > fist")


def test_degenerate_inputs():
    assert grasp_features(None) is None
    assert grasp_features([]) is None
    assert grasp_features([[0.1, 0.2, 0.0]] * 5) is None        # too few points
    assert grasp_features([[0.5, 0.5, 0.0]] * 21) is None        # zero palm scale
    print("ok: degenerate inputs return None")


def _world_hand(aperture_m=0.09, palm_m=0.09):
    """Minimal metric world landmarks: thumb/index tips aperture_m apart,
    middle MCP palm_m from the wrist (meters, hand-centered like MediaPipe)."""
    lm = [[0.0, 0.0, 0.0] for _ in range(21)]
    lm[4] = [-aperture_m / 2, 0.05, 0.0]   # thumb tip
    lm[8] = [aperture_m / 2, 0.05, 0.0]    # index tip
    lm[9] = [0.0, palm_m, 0.0]             # middle MCP (wrist at origin)
    return lm


def test_metric_fields_from_world_landmarks():
    g = grasp_features(open_hand(), world=_world_hand(aperture_m=0.09, palm_m=0.085))
    assert g is not None
    assert abs(g["aperture_m"] - 0.09) < 1e-6
    assert abs(g["hand_scale_m"] - 0.085) < 1e-6
    # Without world landmarks the metric fields are absent (honest).
    g2 = grasp_features(open_hand())
    assert "aperture_m" not in g2 and "hand_scale_m" not in g2
    from pipeline.grasp import hand_scale_m
    assert abs(hand_scale_m(_world_hand(palm_m=0.08)) - 0.08) < 1e-6
    assert hand_scale_m(None) is None
    print("ok: metric aperture/scale from world landmarks")


def test_shape_plausibility():
    """A hand matching the clip's skeleton scores ~1; a frame with hallucinated
    (stretched) joints scores clearly lower; absences return None."""
    from pipeline.grasp import median_bone_lengths, shape_plausibility

    base = [[i * 0.01, (i % 4) * 0.012, (i % 3) * 0.008] for i in range(21)]
    seq = [list(map(list, base)) for _ in range(10)]
    ref = median_bone_lengths(seq)
    assert ref is not None and len(ref) == 21

    good = shape_plausibility(base, ref)
    assert good is not None and good > 0.95, f"consistent hand should score ~1, got {good}"

    distorted = list(map(list, base))
    distorted[8] = [base[8][0] * 4, base[8][1] * 4, base[8][2] * 4]   # stretched index tip
    distorted[12] = [0.0, 0.0, 0.0]                                    # collapsed middle tip
    bad = shape_plausibility(distorted, ref)
    assert bad is not None and bad < good - 0.2, f"hallucinated joints must score lower: {bad} vs {good}"

    assert shape_plausibility(None, ref) is None
    assert shape_plausibility(base, None) is None
    assert median_bone_lengths([None, None]) is None
    print(f"ok: shape plausibility (consistent={good}, hallucinated={bad})")


if __name__ == "__main__":
    test_open_hand()
    test_fist()
    test_monotonic()
    test_degenerate_inputs()
    test_metric_fields_from_world_landmarks()
    test_shape_plausibility()
    print("ALL TESTS PASSED")
