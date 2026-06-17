"""Temporal task segmentation (Phase 3).

Samples the anonymized video at ~1fps, classifies each frame against the
waste-services task taxonomy with a vision model (local Ollama by default — free,
no data egress), then aggregates consecutive same-task frames into segments with
start/end times, a confidence, and a free-text description.

Output: data/processed/{video_id}/segments.json
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import cv2

from config import settings
from .video_meta import probe, apply_rotation
from .segmentation_providers import get_segmentation_provider, FrameLabel
from .boundary import detect_boundaries

log = logging.getLogger("revisent.segmentation")


@dataclass
class Segment:
    start_time: float
    end_time: float
    task_label: str
    confidence: float
    description: str

    @property
    def duration(self) -> float:
        return round(self.end_time - self.start_time, 3)


@dataclass
class SegmentationResult:
    video_id: str
    provider: str
    model: str
    sample_fps: float
    frames_classified: int
    cost_usd: float
    duration_seconds: float
    segments: List[Segment]
    # Present only in "fused" boundary mode: which signals drove the cuts, etc.
    boundary_meta: Optional[dict] = None

    def to_json(self) -> dict:
        doc = {
            "video_id": self.video_id,
            "provider": self.provider,
            "model": self.model,
            "sample_fps": self.sample_fps,
            "frames_classified": self.frames_classified,
            "cost_usd": round(self.cost_usd, 6),
            "duration_seconds": self.duration_seconds,
            "segments": [
                {**asdict(s), "duration_seconds": s.duration} for s in self.segments
            ],
        }
        if self.boundary_meta is not None:
            doc["boundary_meta"] = self.boundary_meta
        return doc


def _aggregate(per_frame: List[tuple], duration: float, sample_interval: float) -> List[Segment]:
    """Group consecutive same-task frames into segments.

    per_frame: list of (timestamp_s, FrameLabel).
    """
    if not per_frame:
        return []
    segments: List[Segment] = []
    cur_label = per_frame[0][1].task
    cur_start = per_frame[0][0]
    confs = [per_frame[0][1].confidence]
    descs = [per_frame[0][1].description]
    last_ts = per_frame[0][0]

    def close(end_ts: float):
        desc = next((d for d in descs if d), "")  # first non-empty description
        segments.append(Segment(
            start_time=round(cur_start, 3),
            end_time=round(end_ts, 3),
            task_label=cur_label,
            confidence=round(sum(confs) / len(confs), 3),
            description=desc,
        ))

    for ts, label in per_frame[1:]:
        if label.task != cur_label:
            close(ts)
            cur_label, cur_start = label.task, ts
            confs, descs = [label.confidence], [label.description]
        else:
            confs.append(label.confidence)
            descs.append(label.description)
        last_ts = ts
    # Final segment extends to the end of the clip (or last sample + one interval).
    close(min(duration, last_ts + sample_interval) if duration else last_ts + sample_interval)
    return segments


def _similar(a: str, b: str, thresh: float = 0.34) -> bool:
    """Token-overlap (Jaccard) similarity of two short activity labels."""
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return a == b
    inter = len(sa & sb)
    return inter / len(sa | sb) >= thresh or inter >= 2


def _aggregate_open(per_frame: List[tuple], duration: float, sample_interval: float) -> List[Segment]:
    """Open-mode aggregation: group consecutive frames with *similar* activity
    labels (open vocabulary varies frame-to-frame), keeping the most informative
    commentary per segment."""
    if not per_frame:
        return []
    segments: List[Segment] = []
    cur_start = per_frame[0][0]
    cur_label = per_frame[0][1].task
    members = [per_frame[0][1]]
    last_ts = per_frame[0][0]

    def close(end_ts: float):
        # Representative label = most common; commentary = longest (most detail).
        labels = [m.task for m in members]
        rep_label = Counter(labels).most_common(1)[0][0]
        commentary = max((m.description for m in members), key=len, default="")
        confs = [m.confidence for m in members]
        segments.append(Segment(
            start_time=round(cur_start, 3), end_time=round(end_ts, 3),
            task_label=rep_label, confidence=round(sum(confs) / len(confs), 3),
            description=commentary,
        ))

    for ts, label in per_frame[1:]:
        if _similar(label.task, cur_label):
            members.append(label)
        else:
            close(ts)
            cur_start, cur_label, members = ts, label.task, [label]
        last_ts = ts
    close(min(duration, last_ts + sample_interval) if duration else last_ts + sample_interval)
    return segments


def _merge_short(segments: List[Segment], min_seconds: float) -> List[Segment]:
    """Absorb sub-threshold segments into the longer adjacent neighbor so brief
    misclassifications don't fragment the timeline."""
    if not segments:
        return segments
    changed = True
    while changed and len(segments) > 1:
        changed = False
        for i, seg in enumerate(segments):
            if seg.duration >= min_seconds:
                continue
            prev = segments[i - 1] if i > 0 else None
            nxt = segments[i + 1] if i < len(segments) - 1 else None
            # Merge into whichever neighbor is longer (or the only one available).
            target = None
            if prev and nxt:
                target = prev if prev.duration >= nxt.duration else nxt
            else:
                target = prev or nxt
            if target is None:
                continue
            target.start_time = min(target.start_time, seg.start_time)
            target.end_time = max(target.end_time, seg.end_time)
            segments.pop(i)
            changed = True
            break
    return segments


# --------------------------------------------------------------------------- #
# Fused boundary mode: detect cuts from cheap signals, label each segment once.
# --------------------------------------------------------------------------- #

def _coalesce(segments: List[Segment]) -> List[Segment]:
    """Merge adjacent segments that share a label. A visual cut inside one task
    (e.g. the worker shifts position) produces two boundaries but the same label;
    coalescing collapses them back into one task segment."""
    if not segments:
        return segments
    open_mode = settings.segmentation_mode == "open"
    out: List[Segment] = [segments[0]]
    for seg in segments[1:]:
        last = out[-1]
        same = _similar(last.task_label, seg.task_label) if open_mode else last.task_label == seg.task_label
        if same:
            d1, d2 = max(last.duration, 1e-3), max(seg.duration, 1e-3)
            total = d1 + d2
            last.confidence = round((last.confidence * d1 + seg.confidence * d2) / total, 3)
            last.description = max((last.description, seg.description), key=len)
            if open_mode and d2 > d1:  # adopt the longer span's representative label
                last.task_label = seg.task_label
            last.end_time = seg.end_time
        else:
            out.append(seg)
    return out


def _load_pose_products(video_id: str):
    """Best-effort load of the Phase-2 hand-pose + head-pose products from storage.
    Any failure (no storage, missing file, parse error) degrades to None."""
    try:
        from storage import get_storage
        storage = get_storage()
    except Exception as exc:  # noqa: BLE001
        log.warning("storage unavailable for pose products: %s", exc)
        return None, None
    hand_rows = head_frames = None
    try:
        hp = storage.local_path(f"processed/{video_id}/hand_pose.parquet")
        if hp.exists():
            from .hand_pose import load_hand_pose
            hand_rows = load_hand_pose(hp)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load hand pose for %s: %s", video_id, exc)
    try:
        head = storage.local_path(f"processed/{video_id}/head_pose.json")
        if head.exists():
            head_frames = json.loads(head.read_text()).get("frames", [])
    except Exception as exc:  # noqa: BLE001
        log.warning("could not load head pose for %s: %s", video_id, exc)
    return hand_rows, head_frames


def _label_segments(
    video_path: Path, meta, provider, intervals: List[tuple], samples: int,
    on_label=None,
) -> tuple[List[FrameLabel], float]:
    """Label each interval from ``samples`` ordered frames, decoded in a single
    pass. When the provider supports it, all of a segment's frames go in ONE
    multi-image call — the model sees what CHANGES across the segment, which is
    what an action is (a lone frame of a massage gun reads as a hairdryer).
    Providers without classify_multi fall back to per-frame majority vote."""
    src_fps = meta.fps or 30.0
    k = max(1, samples)
    # Map each target frame index -> (interval, position-within-interval).
    targets: dict[int, tuple[int, int]] = {}
    for seg_i, (start, end) in enumerate(intervals):
        for j in range(k):
            frac = (j + 1) / (k + 1)
            fidx = int(round((start + frac * (end - start)) * src_fps))
            targets.setdefault(fidx, (seg_i, j))
    frames_per_seg: dict[int, list] = {i: [] for i in range(len(intervals))}
    md = settings.segmentation_frame_max_dim

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        idx = 0
        remaining = len(targets)
        while remaining > 0:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in targets:
                seg_i, j = targets[idx]
                frame = apply_rotation(frame, meta.rotation)
                h, w = frame.shape[:2]
                if md and max(h, w) > md:
                    s = md / max(h, w)
                    frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok2:
                    frames_per_seg[seg_i].append((j, buf.tobytes()))
                remaining -= 1
            idx += 1
    finally:
        cap.release()

    total_cost = 0.0
    labels: List[FrameLabel] = []
    for i in range(len(intervals)):
        ordered = [b for _, b in sorted(frames_per_seg[i], key=lambda t: t[0])]
        if not ordered:
            labels.append(FrameLabel(task="unclear", confidence=0.0, description=""))
            continue
        if hasattr(provider, "classify_multi"):
            label, cost = provider.classify_multi(ordered)
            total_cost += cost
            labels.append(label)
            if on_label:
                on_label(i, len(intervals), label)
            continue
        # Fallback: independent per-frame labels, majority vote.
        members = []
        for b in ordered:
            label, cost = provider.classify(b)
            total_cost += cost
            members.append(label)
        rep = Counter(m.task for m in members).most_common(1)[0][0]
        desc = max((m.description for m in members), key=len, default="")
        conf = round(sum(m.confidence for m in members) / len(members), 3)
        labels.append(FrameLabel(task=rep, confidence=conf, description=desc))
    return labels, total_cost


def _segment_fused(video_path: Path, video_id: str, provider, hand_rows, head_frames) -> SegmentationResult:
    """Boundary-first segmentation: cuts from fused signals, one VLM call/segment."""
    meta = probe(video_path)
    if hand_rows is None and head_frames is None:
        hand_rows, head_frames = _load_pose_products(video_id)

    bres = detect_boundaries(
        video_path, meta,
        hand_rows=hand_rows, head_frames=head_frames,
        grid_fps=settings.boundary_sample_fps,
        min_segment_seconds=settings.boundary_min_segment_seconds,
        smooth_seconds=settings.boundary_smooth_seconds,
        window_seconds=settings.boundary_window_seconds,
        threshold_k=settings.boundary_threshold_k,
        local_window_seconds=settings.boundary_local_window_seconds,
        max_segments=settings.boundary_max_segments,
        use_frame_diff=settings.boundary_use_frame_diff,
        use_hand_pose=settings.boundary_use_hand_pose,
        use_ego_motion=settings.boundary_use_ego_motion,
        weight_frame_diff=settings.boundary_weight_frame_diff,
        weight_hand_pose=settings.boundary_weight_hand_pose,
        weight_ego_motion=settings.boundary_weight_ego_motion,
    )
    intervals = bres.segments or [(0.0, meta.duration_seconds or 0.0)]

    from .progress import report as _progress
    _progress(video_id, "annotating activities",
              detail=f"{len(intervals)} segments found; labeling with VLM")
    labels, total_cost = _label_segments(
        video_path, meta, provider, intervals, settings.boundary_label_samples,
        on_label=lambda i, n, lab: _progress(
            video_id, "annotating activities", pct=100.0 * (i + 1) / n,
            detail=f"{i + 1}/{n}: “{(lab.description or lab.task)[:60]}”"),
    )
    segments = [
        Segment(start_time=round(start, 3), end_time=round(end, 3),
                task_label=lab.task, confidence=lab.confidence, description=lab.description)
        for (start, end), lab in zip(intervals, labels)
    ]
    segments = _coalesce(segments)
    segments = _merge_short(segments, settings.segmentation_min_segment_seconds)

    log.info(
        "segmented %s (fused): %d cut(s) -> %d segments via %s/%s, signals=%s (cost $%.4f)",
        video_id, len(intervals), len(segments), provider.name,
        getattr(provider, "model", "?"), ",".join(bres.signals_used) or "none", total_cost,
    )
    return SegmentationResult(
        video_id=video_id,
        provider=provider.name,
        model=getattr(provider, "model", ""),
        sample_fps=bres.grid_fps,
        frames_classified=sum(1 for _ in intervals) * max(1, settings.boundary_label_samples),
        cost_usd=total_cost,
        duration_seconds=meta.duration_seconds,
        segments=segments,
        boundary_meta={**bres.to_meta(), "mode": "fused",
                       "label_samples_per_segment": max(1, settings.boundary_label_samples)},
    )


def segment_video(
    video_path: Path, video_id: str, provider=None,
    sample_fps: Optional[float] = None,
    hand_rows=None, head_frames=None,
) -> SegmentationResult:
    """Classify the video into task segments using the configured VLM provider.

    Two boundary strategies (``settings.segmentation_boundary_mode``):
      * "perframe" — classify every sampled frame, cut where labels change.
      * "fused"    — detect cuts from cheap signals, label each segment once.
    ``hand_rows``/``head_frames`` are the Phase-2 products used by fused mode; if
    omitted they are loaded from storage by ``video_id`` (best-effort).
    """
    video_path = Path(video_path)
    provider = provider or get_segmentation_provider()
    sample_fps = sample_fps or settings.segmentation_sample_fps

    if settings.segmentation_boundary_mode == "fused":
        return _segment_fused(video_path, video_id, provider, hand_rows, head_frames)

    meta = probe(video_path)
    src_fps = meta.fps or 30.0
    stride = max(1, int(round(src_fps / sample_fps)))
    sample_interval = stride / src_fps

    per_frame: List[tuple] = []
    total_cost = 0.0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                frame = apply_rotation(frame, meta.rotation)
                # Downscale before sending to the VLM — fewer vision tokens = far
                # faster (critical for qwen2.5vl), with no loss for scene-level
                # description.
                md = settings.segmentation_frame_max_dim
                h, w = frame.shape[:2]
                if md and max(h, w) > md:
                    s = md / max(h, w)
                    frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok2:
                    label, cost = provider.classify(buf.tobytes())
                    total_cost += cost
                    per_frame.append((round(idx / src_fps, 3), label))
            idx += 1
    finally:
        cap.release()

    if settings.segmentation_mode == "open":
        # Keep only frames the VLM actually described; blanks aren't observations.
        observed = [(ts, lab) for ts, lab in per_frame if lab.description.strip()]
        segments = _aggregate_open(observed, meta.duration_seconds, sample_interval)
    else:
        segments = _aggregate(per_frame, meta.duration_seconds, sample_interval)
    segments = _merge_short(segments, settings.segmentation_min_segment_seconds)

    log.info(
        "segmented %s: %d frames -> %d segments via %s/%s (cost $%.4f)",
        video_id, len(per_frame), len(segments), provider.name,
        getattr(provider, "model", "?"), total_cost,
    )

    return SegmentationResult(
        video_id=video_id,
        provider=provider.name,
        model=getattr(provider, "model", ""),
        sample_fps=round(src_fps / stride, 3),
        frames_classified=len(per_frame),
        cost_usd=total_cost,
        duration_seconds=meta.duration_seconds,
        segments=segments,
    )


def load_segments(path: Path) -> dict:
    return json.loads(Path(path).read_text())
