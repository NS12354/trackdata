#!/usr/bin/env python3
"""Measure anonymization quality on a clip — coverage and (if labels given) leak rate.

Usage:
    # Detection/coverage report (no ground truth needed):
    python scripts/validate_anonymization.py <video.mp4>

    # Leak-rate report against ground-truth labels:
    python scripts/validate_anonymization.py <video.mp4> --labels labels.json

labels.json format — frames (0-indexed) that contain at least one real face:
    {"face_frames": [12, 13, 14, 88, 89, ...]}

A *leak* is a labeled face-frame that the pipeline does NOT blur. Leak rate =
leaks / labeled_face_frames. This turns "looks blurred" into a number you can
gate releases on. Build a labeled set from representative footage to certify the
detector before a pilot.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from pipeline.video_meta import probe, apply_rotation  # noqa: E402
from pipeline.face_detector import get_face_detector  # noqa: E402
from pipeline.anonymize import _build_filled_boxes, _dilate  # noqa: E402
from config import settings  # noqa: E402


def analyze(video: Path):
    meta = probe(video)
    pad = settings.face_box_padding
    max_gap = max(1, round(settings.temporal_max_gap_seconds * meta.fps))
    hold = max(0, round(settings.temporal_hold_seconds * meta.fps))

    detector = get_face_detector()
    per_frame = []
    cap = cv2.VideoCapture(str(video))
    candidates = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = apply_rotation(frame, meta.rotation)
            dets = detector.detect(frame)
            candidates += len(dets)
            per_frame.append([(_dilate(d.box, pad), d.strong) for d in dets])
    finally:
        cap.release()
        detector.close()

    filled, stats = _build_filled_boxes(
        per_frame, len(per_frame), max_gap, hold,
        settings.track_dilation_per_frame, settings.track_match_iou,
    )
    blurred = [bool(b) for b in filled]
    return per_frame, blurred, stats, candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--labels", help="JSON with face_frames list (ground truth)")
    args = ap.parse_args()

    video = Path(args.video)
    per_frame, blurred, stats, candidates = analyze(video)
    total = len(blurred)
    blurred_n = sum(blurred)

    print(f"video: {video.name}")
    print(f"detector: {settings.face_detector}")
    print(f"frames: {total}")
    print(f"candidate detections: {candidates}")
    print(f"confirmed tracks: {stats['confirmed_tracks']}  rejected (false positive): {stats['rejected_tracks']}")
    print(f"frames blurred (effective coverage): {blurred_n} ({blurred_n/total:.1%})" if total else "no frames")

    if args.labels:
        gt = set(json.loads(Path(args.labels).read_text()).get("face_frames", []))
        if not gt:
            print("labels file has no face_frames; nothing to score")
            return
        leaks = sorted(f for f in gt if f < total and not blurred[f])
        recall = 1 - len(leaks) / len(gt)
        print("--- ground-truth leak analysis ---")
        print(f"labeled face-frames: {len(gt)}")
        print(f"LEAKS (face present, not blurred): {len(leaks)}  -> leak rate {len(leaks)/len(gt):.2%}")
        print(f"face-frame blur recall: {recall:.2%}")
        if leaks:
            print(f"leak frames: {leaks[:50]}{' ...' if len(leaks) > 50 else ''}")


if __name__ == "__main__":
    main()
