#!/usr/bin/env python3
"""Build an OFFICIAL LeRobot dataset from our processed pose data, using lerobot's
own API so it's guaranteed to load in vanilla `lerobot` (v3.0+).

Run in a python>=3.10 venv with lerobot installed (NOT the backend 3.9 venv):
    /tmp/lerobot-venv/bin/python scripts/to_lerobot.py --out data/exports/lerobot_v3
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

JOINT_NAMES = ["pelvis", "spine", "chest", "neck", "head", "l_shoulder", "r_shoulder",
               "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
               "l_knee", "r_knee", "l_ankle", "r_ankle"]
STATE_NAMES = [f"{j}.{a}" for j in JOINT_NAMES for a in ("x", "y", "z")]
ACTION_NAMES = ["l_wrist.x", "l_wrist.y", "l_wrist.z", "r_wrist.x", "r_wrist.y", "r_wrist.z"]
TASK = "egocentric manipulation demonstration"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "exports" / "lerobot_v3"))
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        import shutil; shutil.rmtree(out)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    clips = sorted(p.parent.name for p in (DATA / "processed").glob("*/body_pose.json"))
    docs = [(c, json.loads((DATA / "processed" / c / "body_pose.json").read_text())) for c in clips]
    docs = [(c, d) for c, d in docs if d.get("frames")]
    if not docs:
        raise SystemExit("no processed clips with body_pose.json")

    features = {
        "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        "observation.confidence": {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": list(JOINT_NAMES)},
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
    }
    ds = LeRobotDataset.create(repo_id="revisent/ego-manipulation", fps=10,
                               features=features, root=str(out),
                               robot_type="human_egocentric", use_videos=False)

    total = 0
    for clip, doc in docs:
        for f in doc["frames"]:
            joints = f["joints"]
            state = np.array([v for n in JOINT_NAMES for v in joints[n]["pos"]], dtype=np.float32)
            conf = np.array([joints[n]["confidence"] for n in JOINT_NAMES], dtype=np.float32)
            action = np.array(joints["l_wrist"]["pos"] + joints["r_wrist"]["pos"], dtype=np.float32)
            ds.add_frame({"observation.state": state, "observation.confidence": conf,
                          "action": action, "task": TASK})
            total += 1
        ds.save_episode()
        print(f"  episode {clip[:8]}: {len(doc['frames'])} frames")
    ds.finalize()
    print(f"\nbuilt LeRobot v3 dataset: {len(docs)} episodes, {total} frames -> {out}")


if __name__ == "__main__":
    main()
