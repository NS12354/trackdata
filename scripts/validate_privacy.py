#!/usr/bin/env python3
"""Privacy validation — the diligence killer if you can't produce these numbers.

Reports the three metrics a research engineer asks for before licensing footage:

  face-blur recall      — % of labeled face-frames where a face IS blurred (leaks
                          are the inverse; this is the privacy guarantee)
  hand false-blur rate  — % of hand-only frames where a blur WRONGLY fires (your
                          claimed ~0% — measure it)
  PII-miss rate         — % of labeled PII regions (text/plates/addresses) NOT
                          covered by a blur box (optional; needs region labels)

Build a labeled set spanning lighting, motion, occlusion, multiple workers, and
glove/no-glove. Then:

    # scaffold a labels file from your clips (you then fill the arrays):
    python scripts/validate_privacy.py bootstrap clipA.mp4 clipB.mp4 --out labels.json

    # score + log (feeds the dataset card):
    python scripts/validate_privacy.py score --manifest labels.json

Labels manifest = JSON list of:
  {"video": "path.mp4",
   "face_frames": [12,13,...],         # frames containing a real face
   "hand_only_frames": [40,41,...],    # frames with a visible hand and NO face
   "pii_regions": {"88": [[x1,y1,x2,y2], ...]}}   # optional, pixel coords
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from pipeline.video_meta import probe, apply_rotation  # noqa: E402
from pipeline.face_detector import get_face_detector  # noqa: E402
from pipeline.anonymize import _build_filled_boxes, _dilate  # noqa: E402
from config import settings  # noqa: E402

RUNS_LOG = ROOT / "data" / "eval" / "privacy_runs.jsonl"


def detect_blur_boxes(video: Path):
    """Run the real anonymization detector; return per-frame list of blur boxes."""
    meta = probe(video)
    pad = settings.face_box_padding
    max_gap = max(1, round(settings.temporal_max_gap_seconds * meta.fps))
    hold = max(0, round(settings.temporal_hold_seconds * meta.fps))
    detector = get_face_detector()
    per_frame = []
    cap = cv2.VideoCapture(str(video))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = apply_rotation(frame, meta.rotation)
            dets = detector.detect(frame)
            per_frame.append([(_dilate(d.box, pad), d.strong) for d in dets])
    finally:
        cap.release()
        detector.close()
    filled, stats = _build_filled_boxes(
        per_frame, len(per_frame), max_gap, hold,
        settings.track_dilation_per_frame, settings.track_match_iou,
    )
    return filled, stats


def _overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def score_video(item: dict):
    video = Path(item["video"])
    if not video.is_absolute():
        video = ROOT / video
    filled, _ = detect_blur_boxes(video)
    n = len(filled)
    blurred = [bool(b) for b in filled]

    face = [f for f in item.get("face_frames", []) if f < n]
    leaks = [f for f in face if not blurred[f]]
    face_recall = (1 - len(leaks) / len(face)) if face else None

    hand = [f for f in item.get("hand_only_frames", []) if f < n]
    false_blur = [f for f in hand if blurred[f]]
    hand_fbr = (len(false_blur) / len(hand)) if hand else None

    pii = item.get("pii_regions", {})
    miss = total = 0
    for fstr, regions in pii.items():
        fi = int(fstr)
        if fi >= n:
            continue
        boxes = filled[fi] or []
        for r in regions:
            total += 1
            if not any(_overlap(r, b) for b in boxes):
                miss += 1
    pii_miss = (miss / total) if total else None

    return {
        "video": video.name, "frames": n,
        "face_frames": len(face), "leaks": len(leaks), "face_recall": face_recall,
        "hand_only_frames": len(hand), "hand_false_blur": len(false_blur), "hand_false_blur_rate": hand_fbr,
        "pii_regions": total, "pii_miss": miss, "pii_miss_rate": pii_miss,
    }


def cmd_bootstrap(args):
    items = []
    for v in args.videos:
        p = Path(v)
        m = probe(p if p.is_absolute() else ROOT / p)
        items.append({
            "video": v,
            "_frames_available": m.frame_count,
            "face_frames": [], "hand_only_frames": [], "pii_regions": {},
        })
    Path(args.out).write_text(json.dumps(items, indent=2))
    print(f"wrote labels template {args.out} for {len(items)} clip(s).")
    print("Fill face_frames / hand_only_frames (frame indices) + optional pii_regions, "
          "then: validate_privacy.py score --manifest " + args.out)


def cmd_score(args):
    items = json.loads(Path(args.manifest).read_text())
    if isinstance(items, dict):
        items = [items]
    rows = [score_video(it) for it in items]

    def agg(key, num, den):
        ns = sum(r[num] for r in rows)
        ds = sum(r[den] for r in rows)
        return (1 - ns / ds) if (key == "face_recall" and ds) else (ns / ds if ds else None)

    n_face = sum(r["face_frames"] for r in rows)
    n_leak = sum(r["leaks"] for r in rows)
    n_hand = sum(r["hand_only_frames"] for r in rows)
    n_fb = sum(r["hand_false_blur"] for r in rows)
    n_pii = sum(r["pii_regions"] for r in rows)
    n_miss = sum(r["pii_miss"] for r in rows)

    face_recall = round(1 - n_leak / n_face, 4) if n_face else None
    hand_fbr = round(n_fb / n_hand, 4) if n_hand else None
    pii_miss = round(n_miss / n_pii, 4) if n_pii else None

    print("=" * 60)
    print("PRIVACY VALIDATION")
    print("=" * 60)
    for r in rows:
        print(f"  {r['video']:<32} recall={_fmt(r['face_recall'])} "
              f"hand_fbr={_fmt(r['hand_false_blur_rate'])} pii_miss={_fmt(r['pii_miss_rate'])}")
    print("-" * 60)
    print(f"  Face-blur recall:      {_fmt(face_recall)}  ({n_face - n_leak}/{n_face} face-frames blurred)")
    print(f"  Hand false-blur rate:  {_fmt(hand_fbr)}  ({n_fb}/{n_hand} hand-only frames wrongly blurred)")
    print(f"  PII-miss rate:         {_fmt(pii_miss)}  ({n_miss}/{n_pii} regions missed)"
          if n_pii else "  PII-miss rate:         (no region labels provided)")
    print("=" * 60)

    rec = {
        "ts": time.time(), "n_clips": len(rows),
        "face_recall": face_recall, "hand_false_blur_rate": hand_fbr, "pii_miss_rate": pii_miss,
        "n_face_frames": n_face, "n_hand_only_frames": n_hand, "n_pii_regions": n_pii,
    }
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"logged -> {RUNS_LOG} (the dataset card reads this)")


def _fmt(v):
    return f"{v:.2%}" if isinstance(v, float) else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bootstrap"); b.add_argument("videos", nargs="+")
    b.add_argument("--out", default="privacy_labels.json"); b.set_defaults(f=cmd_bootstrap)
    s = sub.add_parser("score"); s.add_argument("--manifest", required=True); s.set_defaults(f=cmd_score)
    args = ap.parse_args()
    args.f(args)


if __name__ == "__main__":
    main()
