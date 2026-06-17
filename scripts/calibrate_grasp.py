"""Calibrate grasp thresholds from NATURAL footage — no calibration ritual.

Manipulation clips already contain thousands of grasp samples spanning open
reaches and closed grips. This script mines every processed clip's
hand_pose.parquet, builds the aperture distribution, and reports whether the
constants in pipeline/grasp.py (APERTURE_CLOSED=0.4, APERTURE_OPEN=1.6) match
reality — plus the metric (mm) distribution from world landmarks.

Usage (from repo root):
  backend/.venv/Scripts/python scripts/calibrate_grasp.py            # all clips
  backend/.venv/Scripts/python scripts/calibrate_grasp.py img0077    # specific

Report-only: prints suggested constants; apply them in pipeline/grasp.py (or
leave alone if the verdict is "consistent").
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from pipeline.grasp import (  # noqa: E402
    APERTURE_CLOSED, APERTURE_OPEN, CLOSED_THRESHOLD, grasp_features,
)
from pipeline.hand_pose import load_hand_pose  # noqa: E402


def collect(video_ids: list[str]) -> tuple[list[float], list[float], dict]:
    """(normalized apertures, metric apertures in m, per-hand counts)."""
    apertures: list[float] = []
    metric: list[float] = []
    counts = {"left": 0, "right": 0}
    for vid in video_ids:
        pq_path = REPO / "data" / "processed" / vid / "hand_pose.parquet"
        if not pq_path.exists():
            print(f"  (skip {vid}: no hand_pose.parquet)")
            continue
        rows = load_hand_pose(pq_path)
        for r in rows:
            for side in ("left", "right"):
                g = grasp_features(r.get(f"{side}_hand_landmarks"),
                                   r.get(f"{side}_world_landmarks"))
                if not g:
                    continue
                counts[side] += 1
                apertures.append(g["aperture"])
                if "aperture_m" in g:
                    metric.append(g["aperture_m"])
    return apertures, metric, counts


def main() -> None:
    ids = sys.argv[1:]
    if not ids:
        proc = REPO / "data" / "processed"
        ids = sorted(p.name for p in proc.iterdir()
                     if (p / "hand_pose.parquet").exists()) if proc.exists() else []
    if not ids:
        print("no processed clips found"); return
    print(f"calibrating from {len(ids)} clip(s): {', '.join(ids)}")

    apertures, metric, counts = collect(ids)
    if len(apertures) < 50:
        print(f"only {len(apertures)} hand samples — record more footage first")
        return

    a = np.asarray(apertures)
    pct = {p: float(np.percentile(a, p)) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)}
    print(f"\n{len(a)} grasp samples (L {counts['left']} / R {counts['right']})")
    print("normalized aperture percentiles (thumb-index distance in palm lengths):")
    print("  " + "  ".join(f"p{p}={v:.2f}" for p, v in pct.items()))
    if metric:
        m = np.asarray(metric) * 100
        print(f"metric aperture (cm): p5={np.percentile(m,5):.1f} "
              f"p50={np.percentile(m,50):.1f} p95={np.percentile(m,95):.1f}")

    # Suggested mapping: p5 of natural footage ~ a firmly closed grip, p95 ~ a
    # fully open reach. Round to friendly values.
    sug_closed = round(float(np.percentile(a, 5)), 1)
    sug_open = round(float(np.percentile(a, 95)), 1)
    print(f"\ncurrent constants:   APERTURE_CLOSED={APERTURE_CLOSED}  "
          f"APERTURE_OPEN={APERTURE_OPEN}  (closed flag at norm<={CLOSED_THRESHOLD})")
    print(f"data-driven suggest: APERTURE_CLOSED={sug_closed}  APERTURE_OPEN={sug_open}")

    drift_closed = abs(sug_closed - APERTURE_CLOSED)
    drift_open = abs(sug_open - APERTURE_OPEN)
    if drift_closed <= 0.15 and drift_open <= 0.25:
        print("verdict: CONSISTENT — current constants match your footage; no change needed")
    else:
        # How many samples change closed/open classification under suggestion?
        cur_norm = np.clip((a - APERTURE_CLOSED) / (APERTURE_OPEN - APERTURE_CLOSED), 0, 1)
        new_norm = np.clip((a - sug_closed) / (sug_open - sug_closed), 0, 1)
        flips = float(np.mean((cur_norm <= CLOSED_THRESHOLD) != (new_norm <= CLOSED_THRESHOLD)))
        print(f"verdict: DRIFT — updating constants would reclassify "
              f"{flips:.0%} of samples; consider editing pipeline/grasp.py")


if __name__ == "__main__":
    main()
