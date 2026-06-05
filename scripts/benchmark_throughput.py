#!/usr/bin/env python3
"""Measure end-to-end pipeline throughput and project the cost of a large batch.

This is the number you want BEFORE renting GPUs to process 1000 hrs of footage.
It runs the full pipeline (anonymize -> hand pose -> head pose -> segmentation)
on one clip, times each stage, and reports:

  * wall-clock per stage and overall
  * throughput as a multiple of realtime (footage_seconds / compute_seconds)
  * a projection to N hours of footage: total GPU-hours, cost across common GPU
    rentals, how many GPUs to finish "in a day", and a pass/fail vs a budget.

Run it ON THE MACHINE YOU INTEND TO PROCESS WITH. The Mac (CPU) number is a
lower bound for planning; rent one GPU, run this there, and the projection
becomes the real number you can spend $500 against.

Usage:
    # Full pipeline on one clip (needs Ollama running for segmentation):
    python scripts/benchmark_throughput.py path/to/clip.mp4

    # Skip the VLM (no Ollama) -- projection will note segmentation is missing:
    python scripts/benchmark_throughput.py clip.mp4 --skip-segmentation

    # Sweep the dominant cost lever (VLM keyframe sampling rate):
    python scripts/benchmark_throughput.py clip.mp4 --segmentation-fps 0.2

    # Project a different batch size / budget:
    python scripts/benchmark_throughput.py clip.mp4 --target-hours 1000 --budget 500
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from config import settings  # noqa: E402
from pipeline.video_meta import probe  # noqa: E402
from pipeline.anonymize import anonymize_video  # noqa: E402
from pipeline.hand_pose import extract_hand_pose  # noqa: E402
from pipeline.ego_pose import estimate_head_trajectory  # noqa: E402

# Common cloud GPU rentals (USD/hr, spot/community pricing as of 2026). These are
# rough — override with the actual rate you're quoted. The cheap consumer cards
# (4090) are on RunPod/Vast community tiers; L4/A10G/A100 are managed clouds.
GPU_PRESETS = {
    "RTX 4090 (RunPod/Vast)": 0.40,
    "L4 (managed)": 0.80,
    "A10G (managed)": 1.00,
    "A100 80GB (managed)": 2.50,
}


@contextmanager
def timed(label: str, results: dict):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    results[label] = dt


def human_hms(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def run_pipeline(video: Path, *, skip_segmentation: bool, segmentation_fps: float | None):
    """Run each stage on `video`, returning {stage: wall_seconds} and footage seconds."""
    meta = probe(video)
    footage_s = meta.duration_seconds or (meta.frame_count / (meta.fps or 30.0))
    if footage_s <= 0:
        raise SystemExit(f"could not determine clip duration for {video}")

    vid = f"bench-{uuid.uuid4().hex[:8]}"
    work = ROOT / "data" / "_bench"
    work.mkdir(parents=True, exist_ok=True)
    anon_path = work / f"{vid}.mp4"
    parquet_path = work / f"{vid}.parquet"

    stages: dict[str, float] = {}

    print(f"\nClip: {video.name}")
    print(f"  {meta.width}x{meta.height} @ {meta.fps:.1f}fps, {human_hms(footage_s)} of footage")
    print(f"  running pipeline (this is real work, not a simulation)...\n")

    with timed("anonymize", stages):
        anonymize_video(video, anon_path)
    print(f"  [1/4] anonymize        {stages['anonymize']:7.1f}s")

    with timed("hand_pose", stages):
        extract_hand_pose(anon_path, parquet_path, vid)
    print(f"  [2/4] hand pose        {stages['hand_pose']:7.1f}s")

    with timed("head_pose", stages):
        estimate_head_trajectory(anon_path)
    print(f"  [3/4] head pose (VO)   {stages['head_pose']:7.1f}s")

    if skip_segmentation:
        print(f"  [4/4] segmentation       skipped (--skip-segmentation)")
    else:
        from pipeline.segmentation import segment_video  # local import: needs Ollama
        fps = segmentation_fps or settings.segmentation_sample_fps
        with timed("segmentation", stages):
            segment_video(anon_path, vid, sample_fps=fps)
        print(f"  [4/4] segmentation     {stages['segmentation']:7.1f}s  (VLM @ {fps:g} fps keyframes)")

    # Clean up the temp artifacts (footage stays private; this is just scratch).
    for p in (anon_path, parquet_path):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

    return stages, footage_s, meta


def report(stages: dict, footage_s: float, *, target_hours: float, budget: float,
           skip_segmentation: bool):
    total_wall = sum(stages.values())
    overall_x = footage_s / total_wall if total_wall else 0.0

    print("\n" + "=" * 66)
    print("THROUGHPUT (this machine)")
    print("=" * 66)
    print(f"  {'stage':<18}{'wall':>10}{'x realtime':>14}{'% of total':>12}")
    for name, dt in stages.items():
        x = footage_s / dt if dt else float("inf")
        pct = 100 * dt / total_wall if total_wall else 0
        print(f"  {name:<18}{dt:>9.1f}s{x:>12.2f}x{pct:>11.0f}%")
    print(f"  {'-'*54}")
    print(f"  {'TOTAL':<18}{total_wall:>9.1f}s{overall_x:>12.2f}x{100:>11.0f}%")
    print(f"\n  => {overall_x:.2f}x realtime: 1 hour of footage takes "
          f"{human_hms(3600 / overall_x)} of compute on this machine.")

    # Identify the bottleneck stage (the lever to tune).
    if stages:
        slow = max(stages, key=stages.get)
        print(f"  => bottleneck: '{slow}' ({100*stages[slow]/total_wall:.0f}% of time). "
              f"Tune this first.")
        if not skip_segmentation and slow == "segmentation":
            print("     Segmentation cost is ~linear in --segmentation-fps. Halving the "
                  "keyframe\n     rate ~halves this stage. Re-run with --segmentation-fps to compare.")

    # --- Projection to the batch ---
    compute_hours = target_hours / overall_x if overall_x else float("inf")
    print("\n" + "=" * 66)
    print(f"PROJECTION: {target_hours:g} hours of footage")
    print("=" * 66)
    if skip_segmentation:
        print("  !! segmentation was SKIPPED -- real numbers will be HIGHER. !!")
    print(f"  total compute needed: {compute_hours:,.0f} GPU-hours "
          f"(at {overall_x:.2f}x realtime)\n")
    print(f"  {'GPU rental':<26}{'$/hr':>7}{'batch cost':>13}{'GPUs for 1 day':>16}")
    for name, rate in GPU_PRESETS.items():
        cost = compute_hours * rate
        gpus_1d = math.ceil(compute_hours / 24)
        flag = "  <= within budget" if cost <= budget else "  over budget"
        print(f"  {name:<26}{rate:>6.2f}{('$'+format(cost,',.0f')):>13}"
              f"{gpus_1d:>14} {flag if rate <= 1.0 else ''}")

    # Headline: cheapest preset that fits the budget, and the 1-day fan-out.
    print("\n" + "-" * 66)
    cheapest = min(GPU_PRESETS.items(), key=lambda kv: kv[1])
    cost = compute_hours * cheapest[1]
    gpus_1d = math.ceil(compute_hours / 24)
    print(f"  Cheapest path: {cheapest[0]} -> ${cost:,.0f} total "
          f"({'FITS' if cost <= budget else 'EXCEEDS'} ${budget:g} budget).")
    print(f"  To finish in 24h: run ~{gpus_1d} of them in parallel "
          f"(cost is the same — it's just fan-out).")
    # Throughput needed to hit budget on each GPU.
    print(f"\n  Break-even throughput to fit ${budget:g} on this batch:")
    for name, rate in GPU_PRESETS.items():
        need_x = (target_hours * rate) / budget
        print(f"    {name:<26} need >= {need_x:.2f}x realtime  "
              f"({'OK' if overall_x >= need_x else 'short — tune the bottleneck'})")
    print("=" * 66)
    print("\nNote: run this ON A RENTED GPU to get the real number. CPU/Mac results")
    print("are a planning lower bound — the VLM (segmentation) speeds up most on GPU.\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path, help="a representative clip to benchmark")
    ap.add_argument("--skip-segmentation", action="store_true",
                    help="skip the VLM stage (use if Ollama isn't running)")
    ap.add_argument("--segmentation-fps", type=float, default=None,
                    help="override VLM keyframe sampling rate (the dominant cost lever)")
    ap.add_argument("--target-hours", type=float, default=1000.0,
                    help="batch size to project (default 1000)")
    ap.add_argument("--budget", type=float, default=500.0,
                    help="budget to check the projection against (default 500)")
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"not found: {args.video}")

    stages, footage_s, _ = run_pipeline(
        args.video,
        skip_segmentation=args.skip_segmentation,
        segmentation_fps=args.segmentation_fps,
    )
    report(stages, footage_s, target_hours=args.target_hours, budget=args.budget,
           skip_segmentation=args.skip_segmentation)


if __name__ == "__main__":
    main()
