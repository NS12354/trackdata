#!/usr/bin/env python3
"""Measure SPLICING accuracy — are the segment *cuts* landing in the right place?

``eval_labeling.py`` scores how good each segment's *label* is. This scores the
other half: whether the temporal *boundaries* are correct. That's the signal the
fused boundary detector (pipeline/boundary.py) newly makes tunable, and the one
thing the dataset card admits is unmeasured.

Metric: match predicted interior cut times to human ground-truth cuts greedily
within +/- ``tolerance`` seconds, then report precision / recall / F1 and the
mean absolute error of the matched cuts. (Boundary-detection F1@tolerance is the
standard temporal-segmentation metric.)

Workflow
--------
1) Bootstrap a frozen eval set (pre-filled with the detector's DRAFT cuts you
   then correct by scrubbing the video):

       python scripts/eval_boundaries.py bootstrap data/anonymized

   Edit data/eval/boundaries/manifest.json: fix each clip's ``true_boundaries``
   to the real cut times (seconds). Optionally label each resulting segment to
   also score label accuracy. Don't change it once you start comparing.

2) Score a config (re-run after any change to see if it improved):

       # fused detector cuts only (VLM-free, fast)
       python scripts/eval_boundaries.py score --mode fused

       # per-frame path's cuts (needs the VLM / Ollama running)
       python scripts/eval_boundaries.py score --mode perframe

Each ``score`` prints precision/recall/F1, mean abs error, and the DELTA vs the
previous run, and appends to data/eval/boundaries/runs.jsonl so the trend is
visible. Run from the repo root with the backend venv's python.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

EVAL_DIR = REPO_ROOT / "data" / "eval" / "boundaries"
MANIFEST = EVAL_DIR / "manifest.json"
RUNS_LOG = EVAL_DIR / "runs.jsonl"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# --------------------------------------------------------------------------- #
# Boundary sources.
# --------------------------------------------------------------------------- #

def _find_pose_products(video: Path):
    """Best-effort: locate processed/<stem>/ pose products beside the clip so the
    fused detector uses the same signals it would in production. Returns
    (hand_rows, head_frames), either may be None."""
    hand_rows = head_frames = None
    proc = REPO_ROOT / "data" / "processed" / video.stem
    try:
        hp = proc / "hand_pose.parquet"
        if hp.exists():
            from pipeline.hand_pose import load_hand_pose
            hand_rows = load_hand_pose(hp)
    except Exception:
        pass
    try:
        head = proc / "head_pose.json"
        if head.exists():
            head_frames = json.loads(head.read_text()).get("frames", [])
    except Exception:
        pass
    return hand_rows, head_frames


def _interior(boundaries: List[float], duration: float, eps: float = 0.05) -> List[float]:
    return sorted(b for b in boundaries if eps < b < duration - eps)


def predicted_boundaries(video: Path, mode: str) -> tuple[List[float], float]:
    """Interior cut times a given mode produces for a clip, plus its duration."""
    from config import settings
    from pipeline.video_meta import probe
    meta = probe(video)
    dur = meta.duration_seconds or 0.0

    if mode == "fused":
        # VLM-free: call the detector directly with whatever signals exist.
        from pipeline.boundary import detect_boundaries
        hand_rows, head_frames = _find_pose_products(video)
        res = detect_boundaries(
            video, meta, hand_rows=hand_rows, head_frames=head_frames,
            grid_fps=settings.boundary_sample_fps,
            min_segment_seconds=settings.boundary_min_segment_seconds,
            smooth_seconds=settings.boundary_smooth_seconds,
            window_seconds=settings.boundary_window_seconds,
            threshold_k=settings.boundary_threshold_k,
            max_segments=settings.boundary_max_segments,
        )
        return _interior(res.boundaries, dur), dur

    if mode == "perframe":
        # Full per-frame pipeline -> cut = where labels change. Needs the VLM.
        prev = settings.segmentation_boundary_mode
        settings.segmentation_boundary_mode = "perframe"
        try:
            from pipeline.segmentation import segment_video
            result = segment_video(video, video.stem)
        finally:
            settings.segmentation_boundary_mode = prev
        starts = [s.start_time for s in result.segments]
        return _interior(starts, dur), dur

    raise ValueError(f"unknown mode {mode!r}")


# --------------------------------------------------------------------------- #
# Matching + metrics.
# --------------------------------------------------------------------------- #

def match(pred: List[float], truth: List[float], tol: float) -> dict:
    """Greedy nearest-match within tol. Each truth/pred used at most once."""
    pred = sorted(pred)
    truth = sorted(truth)
    used_pred = [False] * len(pred)
    errors: List[float] = []
    tp = 0
    for t in truth:
        best_j, best_d = -1, tol + 1e-9
        for j, p in enumerate(pred):
            if used_pred[j]:
                continue
            d = abs(p - t)
            if d <= tol and d < best_d:
                best_j, best_d = j, d
        if best_j >= 0:
            used_pred[best_j] = True
            tp += 1
            errors.append(best_d)
    fp = len(pred) - tp
    fn = len(truth) - tp
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not truth else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "mae": round(sum(errors) / len(errors), 3) if errors else None,
    }


# --------------------------------------------------------------------------- #
# Commands.
# --------------------------------------------------------------------------- #

def _clips(target: Path) -> List[Path]:
    if target.is_dir():
        return sorted(p for p in target.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    return [target]


def cmd_bootstrap(args) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    clips = _clips(Path(args.target))
    if not clips:
        print(f"no videos found under {args.target}")
        return
    entries = []
    for c in clips:
        try:
            draft, dur = predicted_boundaries(c, "fused")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {c.name}: {exc}")
            draft, dur = [], 0.0
        entries.append({
            "video": str(c.relative_to(REPO_ROOT)) if c.is_relative_to(REPO_ROOT) else str(c),
            "duration_seconds": round(dur, 3),
            "true_boundaries": draft,   # DRAFT from the detector — CORRECT these.
            "segments": [],             # optional: [{"start","end","label"}] for label scoring
            "_note": "Replace true_boundaries with the real interior cut times (s).",
        })
        print(f"  + {c.name}: {len(draft)} draft cut(s), {dur:.1f}s")
    MANIFEST.write_text(json.dumps({"tolerance_seconds": args.tolerance, "clips": entries}, indent=2))
    print(f"\nwrote {MANIFEST}  ({len(entries)} clips)\nNow correct true_boundaries, then: "
          f"python scripts/eval_boundaries.py score --mode fused")


def cmd_score(args) -> None:
    if not MANIFEST.exists():
        print(f"no manifest at {MANIFEST} — run bootstrap first")
        return
    man = json.loads(MANIFEST.read_text())
    tol = args.tolerance or man.get("tolerance_seconds", 1.0)
    agg = {"tp": 0, "fp": 0, "fn": 0}
    all_err: List[float] = []
    print(f"scoring mode={args.mode} tolerance=+/-{tol}s\n")
    for clip in man["clips"]:
        truth = clip.get("true_boundaries") or []
        video = REPO_ROOT / clip["video"]
        if not video.exists():
            print(f"  ? {clip['video']}: missing, skipped")
            continue
        try:
            pred, _ = predicted_boundaries(video, args.mode)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {video.name}: {exc}")
            continue
        m = match(pred, truth, tol)
        agg["tp"] += m["tp"]; agg["fp"] += m["fp"]; agg["fn"] += m["fn"]
        if m["mae"] is not None:
            all_err.append(m["mae"])
        print(f"  {video.name:32s} P={m['precision']:.2f} R={m['recall']:.2f} "
              f"F1={m['f1']:.2f} mae={m['mae']} (pred={len(pred)} true={len(truth)})")

    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    mae = round(sum(all_err) / len(all_err), 3) if all_err else None
    summary = {"mode": args.mode, "tolerance": tol, "precision": round(P, 3),
               "recall": round(R, 3), "f1": round(F1, 3), "mae": mae,
               "tp": tp, "fp": fp, "fn": fn, "clips": len(man["clips"])}

    prev = _last_run(args.mode)
    delta = f"  (Δf1 {F1 - prev['f1']:+.3f})" if prev else ""
    print(f"\nOVERALL  precision={P:.3f}  recall={R:.3f}  F1={F1:.3f}{delta}  mae={mae}s")
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_LOG.open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"appended run to {RUNS_LOG}")


def _last_run(mode: str) -> Optional[dict]:
    if not RUNS_LOG.exists():
        return None
    last = None
    for line in RUNS_LOG.read_text().splitlines():
        try:
            r = json.loads(line)
            if r.get("mode") == mode:
                last = r
        except json.JSONDecodeError:
            pass
    return last


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate segment-boundary accuracy.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bootstrap", help="create a frozen eval set from clips (draft cuts pre-filled)")
    b.add_argument("target", help="a video file or a directory of clips")
    b.add_argument("--tolerance", type=float, default=1.0, help="match window (s)")
    b.set_defaults(func=cmd_bootstrap)
    s = sub.add_parser("score", help="score a mode against the eval set")
    s.add_argument("--mode", choices=["fused", "perframe"], default="fused")
    s.add_argument("--tolerance", type=float, default=None, help="override manifest tolerance (s)")
    s.set_defaults(func=cmd_score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
