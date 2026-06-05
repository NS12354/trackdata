#!/usr/bin/env python3
"""EGOCENTRIC hand-accuracy benchmark against AssemblyHands (the on-point number).

Runs OUR hand pipeline (MediaPipe Hands world landmarks) on AssemblyHands ego
(head-mounted camera) images and reports PA-MPJPE vs the dataset's 3D ground
truth. Unlike FreiHAND (third-person, single hand), this measures egocentric
performance — directly relevant to a head/chest-cam product.

Annotations are tiny (download via gdown). Ego images are a 490K-file Google
Drive folder — download ONE camera folder for one sequence to benchmark:
  Drive: https://drive.google.com/drive/folders/1slji-2LSOBo7eCYNqK6O-qpbvS7D2yXm
  path : ego_images_rectified/val/<sequence>/<HMC_*_mono10bit>/

Usage:
  backend/.venv/bin/python scripts/eval_hand_assemblyhands.py \
      --ann /tmp/assemblyhands/annotations/val \
      --img /tmp/assemblyhands/images --n 400
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

# AssemblyHands order (per skeleton.txt): right hand = idx 0..20, left = 21..41,
# each finger tip->base then wrist last. Reorder into MediaPipe order
# (wrist, thumb cmc->tip, index mcp->tip, ...). MP_TO_AH[m] = AssemblyHands index
# (right hand) for MediaPipe joint m.
MP_TO_AH_RIGHT = [20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16]
MP_TO_AH_LEFT = [i + 21 for i in MP_TO_AH_RIGHT]
MP_NAMES = ["wrist"] + [f"{f}{j}" for f in ["thumb", "index", "middle", "ring", "pinky"]
                        for j in ["_cmc", "_pip2", "_pip", "_tip"]]


def procrustes_align(pred, gt):
    mp_, mg = pred.mean(0), gt.mean(0)
    P, G = pred - mp_, gt - mg
    U, S, Vt = np.linalg.svd(P.T @ G)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    scale = (S @ np.array([1, 1, d])) / (P ** 2).sum()
    return (scale * (R @ P.T)).T + mg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", required=True, help="annotations/val folder")
    ap.add_argument("--img", required=True, help="root containing ego image files")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    ann = Path(args.ann)
    j3d = json.loads((ann / "assemblyhands_val_joint_3d_v1-1.json").read_text())["annotations"]
    ego = json.loads((ann / "assemblyhands_val_ego_data_v1-1.json").read_text())["images"]
    images = list(ego.values()) if isinstance(ego, dict) else ego
    img_root = Path(args.img)

    # which image files do we actually have locally?
    def local_path(file_name: str):
        p1 = img_root / file_name
        if p1.exists():
            return p1
        # also accept the camera folder dropped directly
        p2 = img_root / Path(file_name).name
        return p2 if p2.exists() else None

    import mediapipe as mp
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                     min_detection_confidence=args.conf)

    per_joint_pa, per_joint_rr = [], []
    attempted = matched = 0
    unit_mm = None

    for im in images:
        if matched >= args.n:
            break
        seq, fi = im["seq_name"], im["frame_idx"]
        fk = f"{fi:06d}"
        if seq not in j3d or fk not in j3d[seq]:
            continue
        path = local_path(im["file_name"])
        if path is None:
            continue
        attempted += 1
        rec = j3d[seq][fk]
        wc = np.array(rec["world_coord"], dtype=np.float64)   # (42,3)
        valid = np.array(rec["joint_valid"], dtype=bool)
        if unit_mm is None:
            unit_mm = np.abs(wc).max() > 10  # mm vs meters
        gt_hands = {}
        if valid[MP_TO_AH_RIGHT].all():
            gt_hands["R"] = wc[MP_TO_AH_RIGHT]
        if valid[MP_TO_AH_LEFT].all():
            gt_hands["L"] = wc[MP_TO_AH_LEFT]
        if not gt_hands:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_world_landmarks:
            continue
        for hlm in res.multi_hand_world_landmarks:
            pred = np.array([[lm.x, lm.y, lm.z] for lm in hlm.landmark]) * 1000.0  # mm
            # assign to the GT hand giving the lower PA error
            best = None
            for gt in gt_hands.values():
                gt_mm = gt if unit_mm else gt * 1000.0
                pa = np.linalg.norm(procrustes_align(pred, gt_mm) - gt_mm, axis=1)
                rr = np.linalg.norm((pred - pred[0]) - (gt_mm - gt_mm[0]), axis=1)
                if best is None or pa.mean() < best[0].mean():
                    best = (pa, rr)
            if best is not None:
                per_joint_pa.append(best[0]); per_joint_rr.append(best[1]); matched += 1

    if matched == 0:
        print("No matched ego images found. Make sure the downloaded camera folder's")
        print("images live under --img at the path in ego_data 'file_name', e.g.")
        print("  <img>/ego_images_rectified/val/<seq>/<HMC_*_mono10bit>/000900.jpg")
        print(f"(scanned {attempted} candidate annotated frames)")
        sys.exit(2)

    pa = np.array(per_joint_pa); rr = np.array(per_joint_rr)
    pampjpe, mpjpe = float(pa.mean()), float(rr.mean())
    print("\n" + "=" * 60)
    print(f"EGOCENTRIC HAND ACCURACY vs AssemblyHands  ({matched} hands)")
    print("=" * 60)
    print(f"  MPJPE     (wrist-aligned)            {mpjpe:6.1f} mm")
    print(f"  PA-MPJPE  (Procrustes)               {pampjpe:6.1f} mm   <- quote this")
    print(f"  GT units detected: {'mm' if unit_mm else 'meters->mm'}")
    print("\n  Per-joint PA-MPJPE (mm)  [sanity: no uniform blowup -> mapping ok]:")
    for ji in np.argsort(pa.mean(0)):
        print(f"    {MP_NAMES[ji]:<12} {pa[:, ji].mean():6.1f}")
    verdict = "STRONG" if pampjpe < 20 else ("COMPETITIVE" if pampjpe < 35 else "WEAK (ego is hard)")
    print(f"\n  Verdict: {pampjpe:.1f} mm PA-MPJPE -> {verdict}")
    print("  (Egocentric hands are much harder than 3rd-person; expect worse than FreiHAND.)")
    print("=" * 60)
    rec = {"ts": time.time(), "dataset": "AssemblyHands-ego", "n": matched,
           "mpjpe_mm": round(mpjpe, 2), "pa_mpjpe_mm": round(pampjpe, 2)}
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"logged -> {RUNS_LOG}")


if __name__ == "__main__":
    main()
