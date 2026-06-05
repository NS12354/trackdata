#!/usr/bin/env python3
"""Real hand-accuracy benchmark against FreiHAND (public 3D ground truth).

Runs OUR hand pipeline (MediaPipe Hands, metric world landmarks) on FreiHAND RGB
images and reports error vs the dataset's 3D ground truth — a number a robotics
lab recognizes, in their language (mm).

FreiHAND: https://lmb.informatik.uni-freiburg.de/projects/freihand/
The TRAINING split has public GT (the eval split's GT is withheld). Download +
extract FreiHAND_pub_v2.zip, then:

    backend/.venv/bin/python scripts/eval_hand_freihand.py --root /path/FreiHAND_pub_v2 --n 400

Metrics:
  MPJPE      root-relative (wrist-aligned) — valid because MediaPipe world
             landmarks are metric (meters), like the GT.
  PA-MPJPE   Procrustes-aligned (scale+rot+trans) — the FreiHAND-standard number,
             directly comparable to published results (~7mm SOTA).

A per-joint table is printed as a sanity check: if the joint-order mapping were
wrong, every joint would be uniformly huge; correct mapping -> wrist ~0, small spread.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
RUNS_LOG = ROOT / "data" / "eval" / "hand_runs.jsonl"

# MediaPipe and FreiHAND/MANO both order joints: wrist, then thumb->pinky, each
# finger base->tip. So the mapping is identity. (Verified by the per-joint sanity
# check below — wrong order shows up as uniformly large error.)
FINGER_NAMES = ["wrist"] + [f"{f}{j}" for f in ["thumb", "index", "middle", "ring", "pinky"]
                            for j in ["_mcp", "_pip", "_dip", "_tip"]]


def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    mp_, mg = pred.mean(0), gt.mean(0)
    P, G = pred - mp_, gt - mg
    U, S, Vt = np.linalg.svd(P.T @ G)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    scale = (S @ np.array([1, 1, d])) / (P ** 2).sum()
    return (scale * (R @ P.T)).T + mg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="extracted FreiHAND_pub_v2 folder")
    ap.add_argument("--n", type=int, default=400, help="how many images to evaluate")
    ap.add_argument("--conf", type=float, default=0.3, help="MediaPipe min detection confidence")
    args = ap.parse_args()

    root = Path(args.root)
    # FreiHAND may extract into a nested folder; find the one with training_xyz.json
    if not (root / "training_xyz.json").exists():
        cand = list(root.glob("**/training_xyz.json"))
        if cand:
            root = cand[0].parent
    xyz_p = root / "training_xyz.json"
    rgb_dir = root / "training" / "rgb"
    if not xyz_p.exists() or not rgb_dir.exists():
        raise SystemExit(f"FreiHAND not found under {args.root} "
                         f"(need training_xyz.json + training/rgb/). "
                         f"Download FreiHAND_pub_v2.zip and extract.")

    xyz = json.loads(xyz_p.read_text())           # 32560 x 21 x 3 (meters, cam frame)
    n_gt = len(xyz)
    print(f"FreiHAND: {n_gt} GT poses, evaluating up to {args.n} images...")

    import mediapipe as mp
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                     min_detection_confidence=args.conf)

    per_joint_rr, per_joint_pa = [], []
    detected = 0
    attempted = 0
    for idx in range(args.n):
        img_p = rgb_dir / f"{idx:08d}.jpg"
        if not img_p.exists():
            continue
        attempted += 1
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_world_landmarks:
            continue
        pred = np.array([[lm.x, lm.y, lm.z] for lm in res.multi_hand_world_landmarks[0].landmark])  # meters
        gt = np.array(xyz[idx % n_gt])  # meters
        if pred.shape != gt.shape:
            continue
        detected += 1
        # mm
        pred_mm, gt_mm = pred * 1000.0, gt * 1000.0
        rr = np.linalg.norm((pred_mm - pred_mm[0]) - (gt_mm - gt_mm[0]), axis=1)
        pa = np.linalg.norm(procrustes_align(pred_mm, gt_mm) - gt_mm, axis=1)
        per_joint_rr.append(rr)
        per_joint_pa.append(pa)

    if detected == 0:
        raise SystemExit("no hands detected on the sampled FreiHAND images — check the path / images")

    rr = np.array(per_joint_rr)   # (D, 21) mm
    pa = np.array(per_joint_pa)
    mpjpe = float(rr.mean())
    pampjpe = float(pa.mean())

    print("\n" + "=" * 60)
    print(f"HAND ACCURACY vs FreiHAND  ({detected} hands, {detected}/{attempted} detected)")
    print("=" * 60)
    print(f"  MPJPE     (wrist-aligned, metric)   {mpjpe:6.1f} mm")
    print(f"  PA-MPJPE  (Procrustes; FreiHAND std) {pampjpe:6.1f} mm   <- quote this")
    print(f"  detection rate: {detected/max(1,attempted)*100:.0f}%")
    print("\n  Per-joint PA-MPJPE (mm)  [sanity: wrist small, no uniform blowup]:")
    order = np.argsort(pa.mean(0))
    for ji in order:
        print(f"    {FINGER_NAMES[ji]:<12} {pa[:, ji].mean():6.1f}")
    verdict = "STRONG" if pampjpe < 15 else ("COMPETITIVE" if pampjpe < 25 else "WEAK — investigate")
    print(f"\n  Verdict: {pampjpe:.1f} mm PA-MPJPE -> {verdict}")
    print(f"  (FreiHAND SOTA ~7mm; MediaPipe is a fast general detector, not SOTA.)")
    print("=" * 60)

    rec = {"ts": time.time(), "dataset": "FreiHAND", "n": detected,
           "mpjpe_mm": round(mpjpe, 2), "pa_mpjpe_mm": round(pampjpe, 2),
           "detection_rate": round(detected / max(1, attempted), 3)}
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"logged -> {RUNS_LOG}")


if __name__ == "__main__":
    main()
