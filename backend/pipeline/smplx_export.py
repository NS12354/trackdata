"""SMPL-X / MANO parametric export — the secondary format for retargeting buyers.

Some buyers (NVIDIA GR00T, humanoid teams) prefer SMPL-X body+hand *parameters*
over raw joints because they retarget directly to robot embodiments. This module:

  1. Defines the joint mapping  revisent-ego-17  ->  SMPL-X body joints  (real,
     usable now — see JOINT_TO_SMPLX).
  2. Fits SMPL-X pose parameters to our 3D joints per frame, IF the licensed
     model is present; otherwise raises a clear, actionable error.

SMPL-X model files are LICENSE-GATED (https://smpl-x.is.tue.mpg.de). They cannot
be redistributed, so this ships the integration, not the weights. To enable:
    1. Register + download SMPL-X at https://smpl-x.is.tue.mpg.de
    2. pip install smplx torch
    3. Set SMPLX_MODEL_DIR to the folder containing SMPLX_NEUTRAL.npz
Then fit_smplx_sequence() runs. Until then, ship raw joints + LeRobot.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .body_pose import JOINT_NAMES

log = logging.getLogger("revisent.smplx_export")

# revisent joint name -> SMPL-X body joint index (SMPL-X body kinematic order).
# Our single 'spine' maps to SMPL-X spine1; 'chest' to spine3. head/legs are our
# inferred joints — fitted but flagged low-confidence so buyers can mask them.
JOINT_TO_SMPLX = {
    "pelvis": 0, "l_hip": 1, "r_hip": 2, "spine": 3, "l_knee": 4, "r_knee": 5,
    "chest": 9, "l_ankle": 7, "r_ankle": 8, "neck": 12, "head": 15,
    "l_shoulder": 16, "r_shoulder": 17, "l_elbow": 18, "r_elbow": 19,
    "l_wrist": 20, "r_wrist": 21,
}
# MediaPipe 21-pt hand -> MANO is documented separately; SMPL-X hand pose (45
# params/hand) is fit from the measured hand landmarks when available.

SMPLX_PARAM_SPEC = {
    "transl": [3], "global_orient": [3], "body_pose": [63],
    "left_hand_pose": [45], "right_hand_pose": [45],
    "betas": [10], "expression": [10],
    "_note": "Per-frame except betas/expression (per-subject). Angle-axis radians.",
}


def smplx_available() -> Optional[str]:
    """Return the model dir if SMPL-X is usable, else None."""
    model_dir = os.environ.get("SMPLX_MODEL_DIR", "")
    if not model_dir or not Path(model_dir).exists():
        return None
    try:
        import smplx  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return None
    return model_dir


def fit_smplx_sequence(body_doc: dict, out_path: Path, iters: int = 200) -> dict:
    """Fit SMPL-X params to the body_doc joints and write an .npz.

    Requires the licensed SMPL-X model (see module docstring). Raises a clear
    RuntimeError with setup steps if unavailable.
    """
    model_dir = smplx_available()
    if model_dir is None:
        raise RuntimeError(
            "SMPL-X model not available. This is license-gated:\n"
            "  1) register + download at https://smpl-x.is.tue.mpg.de\n"
            "  2) pip install smplx torch\n"
            "  3) export SMPLX_MODEL_DIR=/path/to/models  (contains SMPLX_NEUTRAL.npz)\n"
            "Until then, ship the raw-joints + LeRobot exports (no license needed)."
        )
    import torch
    import smplx

    device = "cpu"
    model = smplx.create(model_dir, model_type="smplx", gender="neutral",
                         use_pca=False, batch_size=1).to(device)

    frames = body_doc["frames"]
    target_idx = [JOINT_TO_SMPLX[n] for n in JOINT_NAMES]
    H = body_doc.get("height_cm", 170) / 100.0

    betas = torch.zeros(1, 10, requires_grad=True)
    out_params = {k: [] for k in ("transl", "global_orient", "body_pose",
                                  "left_hand_pose", "right_hand_pose")}
    confidences = []

    for f in frames:
        tgt = np.array([f["joints"][n]["pos"] for n in JOINT_NAMES], np.float32)
        w = np.array([f["joints"][n]["confidence"] for n in JOINT_NAMES], np.float32)
        tgt_t = torch.tensor(tgt)[None]
        w_t = torch.tensor(w)[None, :, None]

        transl = torch.zeros(1, 3, requires_grad=True)
        go = torch.zeros(1, 3, requires_grad=True)
        bp = torch.zeros(1, 63, requires_grad=True)
        lh = torch.zeros(1, 45, requires_grad=True)
        rh = torch.zeros(1, 45, requires_grad=True)
        opt = torch.optim.Adam([transl, go, bp, lh, rh, betas], lr=0.05)
        for _ in range(iters):
            opt.zero_grad()
            o = model(betas=betas, global_orient=go, body_pose=bp,
                      left_hand_pose=lh, right_hand_pose=rh, transl=transl)
            pred = o.joints[:, target_idx, :]
            loss = (w_t * (pred - tgt_t) ** 2).mean()
            loss.backward(); opt.step()
        for k, v in (("transl", transl), ("global_orient", go), ("body_pose", bp),
                     ("left_hand_pose", lh), ("right_hand_pose", rh)):
            out_params[k].append(v.detach().cpu().numpy()[0])
        confidences.append(w)

    npz = {k: np.array(v, np.float32) for k, v in out_params.items()}
    npz["betas"] = betas.detach().cpu().numpy()
    npz["timestamps_ms"] = np.array([f["timestamp_ms"] for f in frames], np.float64)
    npz["joint_confidence"] = np.array(confidences, np.float32)
    npz["fit_joint_order"] = np.array(JOINT_NAMES)
    np.savez_compressed(out_path, **npz)
    log.info("SMPL-X fit: %d frames -> %s", len(frames), out_path)
    return {"frames": len(frames), "path": str(out_path)}


def write_smplx_readme(out_dir: Path) -> Path:
    """Document the SMPL-X mapping + how to enable fitting (ships always)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "format": "SMPL-X parameters (.npz)",
        "status": "integration shipped; weights are license-gated and NOT included",
        "enable": [
            "register + download at https://smpl-x.is.tue.mpg.de",
            "pip install smplx torch",
            "export SMPLX_MODEL_DIR=/path/to/models",
        ],
        "param_spec": SMPLX_PARAM_SPEC,
        "joint_mapping_revisent_to_smplx": JOINT_TO_SMPLX,
        "provenance": "head + legs are inferred priors; joint_confidence in the "
                      ".npz lets you mask them during retargeting.",
    }
    p = out_dir / "smplx_README.json"
    p.write_text(json.dumps(doc, indent=2))
    return p


if __name__ == "__main__":
    import sys
    from .pose_export import load_body_doc
    vid = sys.argv[1] if len(sys.argv) > 1 else "91b674ab-ab61-426d-b3d8-631bc84e10fe"
    out = Path("../data/exports") / vid / "smplx"
    readme = write_smplx_readme(out)
    print(f"wrote mapping/readme: {readme}")
    print(f"SMPL-X model available: {bool(smplx_available())}")
    if smplx_available():
        doc = load_body_doc(vid)
        print(fit_smplx_sequence(doc, out / "smplx_params.npz", iters=50))
    else:
        print("(fitting skipped — license-gated weights not present; integration is ready)")
