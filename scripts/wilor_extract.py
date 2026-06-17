"""WiLoR hand extraction runner — executes inside the GPU env (data/tmp_wilor_env).

Samples a video at the requested fps, runs WiLoR (transformer hand-mesh
regression with a MANO prior: occlusion-robust, metric) per frame, and writes
raw detections as JSON lines for the backend to convert into the standard
hand_pose.parquet. Kept dependency-minimal: this env has torch/cv2/numpy but
not the backend stack.

Usage:
  data/tmp_wilor_env/Scripts/python scripts/wilor_extract.py \
      <video> <out.jsonl> --fps 30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--progress", default="", help="optional progress.json path")
    args = ap.parse_args()

    def progress(pct, detail):
        if not args.progress:
            return
        try:
            with open(args.progress, "w", encoding="utf-8") as pf:
                pf.write(json.dumps({"stage": "hand tracking (WiLoR, GPU)",
                                     "pct": round(pct, 1), "detail": detail,
                                     "ts": time.time()}))
        except Exception:
            pass

    logging.disable(logging.INFO)
    import cv2
    import numpy as np
    import torch
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = WiLorHandPose3dEstimationPipeline(
        device=dev, dtype=torch.float16 if dev == "cuda" else torch.float32)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"could not open {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / args.fps)))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    t0 = time.time()
    n_frames = n_hands = 0
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": {"backend": "wilor", "device": dev,
                                     "src_fps": src_fps, "stride": stride,
                                     "width": w, "height": h}}) + "\n")
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                if n_frames % 60 == 0:
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                    progress(100.0 * idx / total, f"frame {idx}/{total}, {n_hands} hands so far")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands = []
                try:
                    out = pipe.predict(rgb)
                except Exception as exc:  # noqa: BLE001 - one bad frame shouldn't kill the run
                    print(f"frame {idx}: predict failed: {exc}", file=sys.stderr)
                    out = []
                for hd in out:
                    p = hd.get("wilor_preds") or {}
                    try:
                        hands.append({
                            "is_right": float(hd.get("is_right", 1.0)),
                            "kp2d": np.asarray(p["pred_keypoints_2d"])[0].round(2).tolist(),
                            "kp3d": np.asarray(p["pred_keypoints_3d"])[0].round(5).tolist(),
                            "cam_t": np.asarray(p["pred_cam_t_full"]).reshape(-1)[:3].round(5).tolist(),
                            "focal": float(np.asarray(p["scaled_focal_length"]).reshape(-1)[0]),
                        })
                    except Exception as exc:  # noqa: BLE001
                        print(f"frame {idx}: bad hand record: {exc}", file=sys.stderr)
                n_hands += len(hands)
                f.write(json.dumps({"frame": idx,
                                    "ts": round(idx / src_fps * 1000.0, 2),
                                    "hands": hands}) + "\n")
                n_frames += 1
            idx += 1
    cap.release()
    dt = time.time() - t0
    print(f"WILOR_DONE frames={n_frames} hands={n_hands} "
          f"sec={dt:.0f} ms_per_frame={1000*dt/max(1,n_frames):.0f} device={dev}")


if __name__ == "__main__":
    main()
