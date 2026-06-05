#!/usr/bin/env python3
"""Tune the hand detector for blurry/fast/unstructured footage, measured on the
AssemblyHands ego images (which are exactly that: blurry, fast, occluded).

Sweeps MediaPipe settings + input preprocessing and reports, per config:
  detection rate (don't lose the hand — the #1 thing for blur) and
  PA-MPJPE (accuracy on the hands it does find).
Pick the config that lifts detection without wrecking accuracy.

  backend/.venv/bin/python scripts/tune_hand.py --ann /tmp/assemblyhands/annotations/val \
      --img /tmp/assemblyhands/images --n 250
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import cv2

MP_TO_AH_RIGHT = [20, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16]
MP_TO_AH_LEFT = [i + 21 for i in MP_TO_AH_RIGHT]


def procrustes(pred, gt):
    mp_, mg = pred.mean(0), gt.mean(0)
    P, G = pred - mp_, gt - mg
    U, S, Vt = np.linalg.svd(P.T @ G)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    s = (S @ np.array([1, 1, d])) / (P ** 2).sum()
    return (s * (R @ P.T)).T + mg


def sharpen(img):
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def upscale(img):
    return cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)


PREPROC = {"none": lambda x: x, "sharpen": sharpen, "clahe": clahe, "upscale": upscale}


def load_samples(ann, img_root, n):
    j3d = json.loads((ann / "assemblyhands_val_joint_3d_v1-1.json").read_text())["annotations"]
    ego = json.loads((ann / "assemblyhands_val_ego_data_v1-1.json").read_text())["images"]
    images = list(ego.values()) if isinstance(ego, dict) else ego
    out = []
    for im in images:
        if len(out) >= n:
            break
        seq, fi = im["seq_name"], im["frame_idx"]
        fk = f"{fi:06d}"
        if seq not in j3d or fk not in j3d[seq]:
            continue
        p = img_root / im["file_name"]
        if not p.exists():
            continue
        rec = j3d[seq][fk]
        wc = np.array(rec["world_coord"], float)
        valid = np.array(rec["joint_valid"], bool)
        gts = {}
        if valid[MP_TO_AH_RIGHT].all():
            gts["R"] = wc[MP_TO_AH_RIGHT]
        if valid[MP_TO_AH_LEFT].all():
            gts["L"] = wc[MP_TO_AH_LEFT]
        if gts:
            out.append((p, gts))
    return out


def run_config(samples, mp, complexity, conf, prefn):
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                     model_complexity=complexity, min_detection_confidence=conf)
    errs, det = [], 0
    for p, gts in samples:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = prefn(img)
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_world_landmarks:
            continue
        det += 1
        for hlm in res.multi_hand_world_landmarks:
            pred = np.array([[lm.x, lm.y, lm.z] for lm in hlm.landmark]) * 1000.0
            best = min(np.linalg.norm(procrustes(pred, gt) - gt, axis=1).mean() for gt in gts.values())
            errs.append(best)
    hands.close()
    n = len(samples)
    return det / n if n else 0, float(np.mean(errs)) if errs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--n", type=int, default=250)
    args = ap.parse_args()
    import mediapipe as mp

    samples = load_samples(Path(args.ann), Path(args.img), args.n)
    print(f"loaded {len(samples)} GT-matched ego images\n")
    configs = [
        ("baseline      cx1 c.3 none", 1, 0.3, "none"),
        ("lower-conf     cx1 c.2 none", 1, 0.2, "none"),
        ("lowest-conf    cx1 c.1 none", 1, 0.1, "none"),
        ("sharpen        cx1 c.3 sharp", 1, 0.3, "sharpen"),
        ("clahe          cx1 c.3 clahe", 1, 0.3, "clahe"),
        ("upscale        cx1 c.3 up1.5", 1, 0.3, "upscale"),
        ("sharpen+low    cx1 c.2 sharp", 1, 0.2, "sharpen"),
        ("complexity0    cx0 c.3 none", 0, 0.3, "none"),
    ]
    print(f"{'config':<30}{'detect%':>9}{'PA-MPJPE':>11}")
    print("-" * 50)
    results = []
    for name, cx, conf, pre in configs:
        dr, err = run_config(samples, mp, cx, conf, PREPROC[pre])
        results.append((name, dr, err))
        print(f"{name:<30}{dr*100:>8.1f}%{err:>9.1f}mm")
    print("-" * 50)
    # pick: maximize detection, tie-break lower error (detection matters most for blur)
    best = max(results, key=lambda r: (round(r[1], 3), -r[2]))
    print(f"\nBest for blur (max detection, then accuracy): {best[0]}  "
          f"-> {best[1]*100:.1f}% detect, {best[2]:.1f}mm")


if __name__ == "__main__":
    main()
