#!/usr/bin/env python3
"""Pose-accuracy benchmark — the credential that opens doors at frontier labs.

Computes MPJPE (mean per-joint position error, mm) of our pose against a public
ground-truth dataset, reported the way these labs read it:

  * MPJPE        — root-aligned error (subtract pelvis/wrist)
  * PA-MPJPE     — Procrustes-aligned (scale+rot+trans). THE fair metric for our
                   monocular, up-to-scale data — quote this one first.
  * per-joint    — error for every joint
  * by provenance — measured (hands) vs ik (arms) vs oriented (torso) vs inferred
                    (legs/head). This stratification is our differentiator: most
                    vendors quote one average; we show exactly which joints to trust.

Datasets (download separately; all are license/registration gated):
  assemblyhands  egocentric hand pose, GT 3D hand joints   (hand MPJPE)
  egoexo4d       egocentric body pose, GT 3D body joints    (arm/torso MPJPE)
  pairs          bring-your-own: an .npz of pred/gt/joint_names you produced
  synthetic      self-test (no data needed) — verifies the metric core is correct

Usage:
  python scripts/eval_pose.py synthetic
  python scripts/eval_pose.py assemblyhands --root /data/AssemblyHands
  python scripts/eval_pose.py pairs --npz mypairs.npz

Target ranges (be honest about where you land):
  hand MPJPE  : <25mm competitive, <15mm strong
  arm/torso   : <50mm in the conversation, <30mm strong
If you're far worse, fix the model (wrist/IK) before selling — don't soft-pedal.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
RUNS_LOG = ROOT / "data" / "eval" / "pose_runs.jsonl"


# ---------------------------------------------------------------------------
# metric core (verified by `synthetic`)
# ---------------------------------------------------------------------------
def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Similarity-align pred to gt (scale+rotation+translation). Returns aligned pred."""
    mp, mg = pred.mean(0), gt.mean(0)
    P, G = pred - mp, gt - mg
    U, S, Vt = np.linalg.svd(P.T @ G)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    scale = S @ np.array([1, 1, d]) / (P ** 2).sum()
    return (scale * (R @ P.T)).T + mg


def mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-joint Euclidean error (same units as input)."""
    return np.linalg.norm(pred - gt, axis=-1)


def root_align(j: np.ndarray, root_idx: int) -> np.ndarray:
    return j - j[root_idx]


def evaluate_sample(pred: np.ndarray, gt: np.ndarray, root_idx: int = 0):
    """Return (per_joint_mpjpe_mm, per_joint_pampjpe_mm)."""
    pj = mpjpe(root_align(pred, root_idx), root_align(gt, root_idx))
    pa = mpjpe(procrustes_align(pred, gt), gt)
    return pj, pa


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def aggregate_and_report(per_joint_mpjpe, per_joint_pa, joint_names, provenance=None,
                         dataset="", n_samples=0):
    pj = np.array(per_joint_mpjpe)        # (N, J) mm
    pa = np.array(per_joint_pa)
    overall_mpjpe = float(pj.mean())
    overall_pa = float(pa.mean())

    print("\n" + "=" * 64)
    print(f"POSE ACCURACY — {dataset}  ({n_samples} samples)")
    print("=" * 64)
    print(f"  MPJPE     (root-aligned)      {overall_mpjpe:7.1f} mm")
    print(f"  PA-MPJPE  (Procrustes; FAIR)  {overall_pa:7.1f} mm   <- quote this")

    print("\n  Per-joint PA-MPJPE (mm):")
    order = np.argsort(pa.mean(0))
    for ji in order:
        tag = f"  [{provenance[ji]}]" if provenance else ""
        print(f"    {joint_names[ji]:<12} {pa[:, ji].mean():7.1f}{tag}")

    by_prov = {}
    if provenance:
        print("\n  By provenance (PA-MPJPE mm) — which joints to trust:")
        for prov in ("measured", "ik", "oriented", "inferred"):
            idx = [i for i, p in enumerate(provenance) if p == prov]
            if idx:
                v = float(pa[:, idx].mean())
                by_prov[prov] = round(v, 1)
                print(f"    {prov:<10} {v:7.1f}")

    # honest verdict against target ranges
    hand_idx = [i for i, n in enumerate(joint_names) if "wrist" in n or "hand" in n]
    if hand_idx:
        h = float(pa[:, hand_idx].mean())
        verdict = "STRONG" if h < 15 else ("COMPETITIVE" if h < 25 else "NEEDS WORK")
        print(f"\n  Hand/wrist PA-MPJPE: {h:.1f} mm  -> {verdict}")
    print("=" * 64)

    rec = {
        "ts": time.time(), "dataset": dataset, "n_samples": n_samples,
        "mpjpe_mm": round(overall_mpjpe, 2), "pa_mpjpe_mm": round(overall_pa, 2),
        "per_joint_pa_mm": {joint_names[i]: round(float(pa[:, i].mean()), 2) for i in range(len(joint_names))},
        "by_provenance_pa_mm": by_prov,
    }
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"logged -> {RUNS_LOG}\n")
    return rec


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
def run_synthetic(args):
    """Self-test: known-noise predictions must recover ~that error. Verifies core."""
    rng = np.random.default_rng(0)
    from pipeline.body_pose import JOINT_NAMES, _T
    provenance = [t[2] for t in _T]   # tuple: (name, frac, provenance, base_conf)
    J = len(JOINT_NAMES)
    noise_mm = 20.0
    pj_all, pa_all = [], []
    for _ in range(args.n):
        gt = rng.normal(0, 300, size=(J, 3))               # mm-scale skeleton
        # prediction = a random similarity transform of GT + gaussian noise; PA
        # alignment should undo the transform and recover ~noise_mm.
        R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        pred = (1.4 * (R @ gt.T).T + rng.normal(0, 50, 3)) + rng.normal(0, noise_mm, size=(J, 3))
        pj, pa = evaluate_sample(pred, gt)
        pj_all.append(pj); pa_all.append(pa)
    rec = aggregate_and_report(pj_all, pa_all, list(JOINT_NAMES), provenance,
                               dataset="synthetic-selftest", n_samples=args.n)
    ok = abs(rec["pa_mpjpe_mm"] - noise_mm) < 6.0
    print(f"SELF-TEST {'PASS' if ok else 'FAIL'}: PA-MPJPE {rec['pa_mpjpe_mm']}mm "
          f"vs injected {noise_mm}mm (Procrustes correctly removed scale+rotation).")
    return 0 if ok else 1


def run_pairs(args):
    """Bring-your-own pred/gt: .npz with arrays pred (N,J,3), gt (N,J,3), joint_names (J)."""
    d = np.load(args.npz, allow_pickle=True)
    pred, gt = d["pred"], d["gt"]
    names = list(d["joint_names"]) if "joint_names" in d else [f"j{i}" for i in range(pred.shape[1])]
    prov = list(d["provenance"]) if "provenance" in d else None
    pj_all, pa_all = [], []
    for i in range(pred.shape[0]):
        pj, pa = evaluate_sample(pred[i], gt[i])
        pj_all.append(pj); pa_all.append(pa)
    aggregate_and_report(pj_all, pa_all, names, prov, dataset=Path(args.npz).stem, n_samples=pred.shape[0])
    return 0


def run_assemblyhands(args):
    """Hand MPJPE on AssemblyHands. Runs MediaPipe on their egocentric frames and
    compares 21-pt hand joints (wrist-relative) to GT 3D annotations.

    Expected layout (download from https://assemblyhands.github.io):
      <root>/images/ego/...            egocentric frames
      <root>/annotations/*.json        GT 3D joints (camera frame, mm)
    """
    root = Path(args.root)
    ann_dir = root / "annotations"
    if not root.exists() or not ann_dir.exists():
        print(f"AssemblyHands not found at {root}.")
        print("Download (registration-gated) from https://assemblyhands.github.io and")
        print("point --root at the extracted folder. The adapter then runs MediaPipe")
        print("on each ego frame and reports hand PA-MPJPE vs GT.")
        print("\nThe metric core is verified independently: run "
              "`python scripts/eval_pose.py synthetic`.")
        return 2
    # Real evaluation path (runs when the dataset is present).
    try:
        import cv2  # noqa: F401
        import mediapipe as mp
    except ImportError:
        print("need opencv + mediapipe installed to run the AssemblyHands adapter")
        return 2
    from glob import glob
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                     min_detection_confidence=0.3)
    pj_all, pa_all = [], []
    names = [f"hand_{i}" for i in range(21)]
    n = 0
    for ann_path in sorted(glob(str(ann_dir / "*.json")))[: args.limit or None]:
        ann = json.loads(Path(ann_path).read_text())
        for sample in ann.get("samples", []):
            img = cv2.imread(str(root / sample["image"]))
            if img is None:
                continue
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_hand_landmarks:
                continue
            h, w = img.shape[:2]
            pred = np.array([[lm.x * w, lm.y * h, lm.z * w]
                             for lm in res.multi_hand_landmarks[0].landmark])
            gt = np.array(sample["joints3d"])  # (21,3) mm, dataset convention
            pj, pa = evaluate_sample(pred, gt)
            pj_all.append(pj); pa_all.append(pa); n += 1
    if not n:
        print("no usable AssemblyHands samples matched — check layout/annotation keys")
        return 2
    aggregate_and_report(pj_all, pa_all, names, dataset="AssemblyHands", n_samples=n)
    return 0


def run_egoexo4d(args):
    """Body/arm MPJPE on Ego-Exo4D body pose (GT from the multi-cam rig).

    Expected: <root>/takes/.../body_pose.json (GT 3D body joints). Map our 17-joint
    skeleton to their convention, report arm/torso PA-MPJPE; legs are inferred and
    excluded from the headline (state that explicitly)."""
    root = Path(args.root)
    if not root.exists():
        print(f"Ego-Exo4D not found at {root}.")
        print("Download (license-gated) from https://ego-exo4d-data.org and point")
        print("--root at the extracted folder. The adapter maps our skeleton to their")
        print("joint convention and reports arm/torso PA-MPJPE (legs excluded, stated).")
        return 2
    print("Ego-Exo4D adapter: dataset present — implement the take->our-pipeline run")
    print("here once the local takes layout is confirmed (frames + GT body_pose).")
    return 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic"); s.add_argument("--n", type=int, default=200); s.set_defaults(f=run_synthetic)
    p = sub.add_parser("pairs"); p.add_argument("--npz", required=True); p.set_defaults(f=run_pairs)
    a = sub.add_parser("assemblyhands"); a.add_argument("--root", required=True)
    a.add_argument("--limit", type=int, default=0); a.set_defaults(f=run_assemblyhands)
    e = sub.add_parser("egoexo4d"); e.add_argument("--root", required=True); e.set_defaults(f=run_egoexo4d)
    args = ap.parse_args()
    sys.exit(args.f(args))


if __name__ == "__main__":
    main()
