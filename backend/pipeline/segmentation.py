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

    def to_json(self) -> dict:
        return {
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


def segment_video(
    video_path: Path, video_id: str, provider=None,
    sample_fps: Optional[float] = None,
) -> SegmentationResult:
    """Classify the video into task segments using the configured VLM provider."""
    video_path = Path(video_path)
    provider = provider or get_segmentation_provider()
    sample_fps = sample_fps or settings.segmentation_sample_fps

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
