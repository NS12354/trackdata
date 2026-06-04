"""Background job orchestration.

v1 uses FastAPI background tasks (a thread), not Celery. Each job opens its own
DB session, updates status defensively, and records failures on the row so the
UI can surface them rather than silently dropping the video.
"""
from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone

from config import settings
from db import session_scope
from models import Video, VideoStatus
from storage import get_storage
from .anonymize import anonymize_video
from .hand_pose import extract_hand_pose
from .segmentation import segment_video

log = logging.getLogger("revisent.jobs")


def _uploads_key(video_id: str, ext: str) -> str:
    return f"uploads/{video_id}{ext}"


def _anonymized_key(video_id: str) -> str:
    # Always mp4 (H.264) on output.
    return f"anonymized/{video_id}.mp4"


def _hand_pose_key(video_id: str) -> str:
    return f"processed/{video_id}/hand_pose.parquet"


def _segments_key(video_id: str) -> str:
    return f"processed/{video_id}/segments.json"


def _head_pose_key(video_id: str) -> str:
    return f"processed/{video_id}/head_pose.json"


def run_head_pose(video_id: str) -> None:
    """Estimate the head trajectory (visual odometry) from the anonymized video
    and store it. Best-effort foundation for the ego-body pose view."""
    storage = get_storage()
    anon_key = _anonymized_key(video_id)
    if not storage.exists(anon_key):
        raise FileNotFoundError(f"anonymized video missing for {video_id}")
    from .ego_pose import estimate_head_trajectory, trajectory_to_json
    in_path = storage.local_path(anon_key)
    out_path = storage.local_path(_head_pose_key(video_id))
    poses = estimate_head_trajectory(in_path)
    out_path.write_text(json.dumps(trajectory_to_json(video_id, poses)))
    log.info("head pose %s: %d poses stored", video_id, len(poses))


def run_anonymization(video_id: str, upload_key: str) -> None:
    """Anonymize an uploaded video and update its DB row.

    Designed to run in a background thread. Never raises to the caller; failures
    are written to the video row instead.
    """
    storage = get_storage()
    try:
        with session_scope() as s:
            video = s.get(Video, video_id)
            if video is None:
                log.error("anonymization: video %s not found", video_id)
                return
            video.status = VideoStatus.processing
            video.error_message = None

        in_path = storage.local_path(upload_key)
        out_key = _anonymized_key(video_id)
        out_path = storage.local_path(out_key)

        log.info("anonymizing %s -> %s", in_path, out_path)
        result = anonymize_video(in_path, out_path)
        log.info(
            "anonymized %s: coverage=%.1f%% faces=%d frames=%d codec=%s",
            video_id, result.coverage * 100, result.total_face_detections,
            result.frames_total, result.output_codec,
        )

        with session_scope() as s:
            video = s.get(Video, video_id)
            video.status = VideoStatus.anonymized
            video.anonymized_at = datetime.now(timezone.utc)
            video.anonymization_coverage = result.coverage
            video.anonymization_method = result.method
            video.anonymization_meta = json.dumps(result.as_meta())
            # Backfill duration/size if probe gave us better numbers.
            if result.duration_seconds and not video.duration_seconds:
                video.duration_seconds = result.duration_seconds

        # Privacy: delete the raw upload once anonymization succeeded, unless we
        # are explicitly retaining raw uploads for debugging.
        if not settings.retain_raw_uploads:
            storage.delete(upload_key)
            log.info("deleted raw upload %s (retain_raw_uploads=False)", upload_key)

    except Exception as exc:  # noqa: BLE001
        # Raise so the task layer can retry (transient) or mark-failed (terminal).
        log.error("anonymization failed for %s: %s", video_id, exc)
        log.debug("%s", traceback.format_exc())
        raise


def run_hand_pose(video_id: str) -> None:
    """Extract MediaPipe hand keypoints from the anonymized video (Phase 2).

    Raises on failure so the task layer can retry. Hand-pose failure is non-fatal
    to the video (it stays anonymized/usable); the task records the error.
    """
    storage = get_storage()
    anon_key = _anonymized_key(video_id)
    if not storage.exists(anon_key):
        raise FileNotFoundError(f"anonymized video missing for {video_id}")

    in_path = storage.local_path(anon_key)
    out_key = _hand_pose_key(video_id)
    out_path = storage.local_path(out_key)

    log.info("extracting hand pose for %s", video_id)
    result = extract_hand_pose(in_path, out_path, video_id)
    log.info(
        "hand pose %s: %d/%d sampled frames had hands (%.1f%%), sample_fps=%.1f",
        video_id, result.frames_with_any_hand, result.frames_sampled,
        result.coverage * 100, result.sample_fps,
    )

    with session_scope() as s:
        video = s.get(Video, video_id)
        if video is not None:
            video.hand_pose_extracted = True
            video.hand_pose_extracted_at = datetime.now(timezone.utc)
            video.error_message = None


def run_segmentation(video_id: str) -> None:
    """Segment the anonymized video into task segments (Phase 3).

    Uses the configured VLM provider (local Ollama by default — free, no data
    egress). Raises on failure so the task layer can retry. Records cost (0.0 for
    the local model) and flips the video to ``processed``.
    """
    storage = get_storage()
    anon_key = _anonymized_key(video_id)
    if not storage.exists(anon_key):
        raise FileNotFoundError(f"anonymized video missing for {video_id}")

    in_path = storage.local_path(anon_key)
    out_path = storage.local_path(_segments_key(video_id))

    log.info("segmenting %s", video_id)
    result = segment_video(in_path, video_id)
    out_path.write_text(json.dumps(result.to_json(), indent=2))

    # Derive a scene/setting label from the commentary (local LLM, best-effort).
    from .chat import derive_scene
    scene = derive_scene([s.description for s in result.segments])

    with session_scope() as s:
        video = s.get(Video, video_id)
        if video is not None:
            video.segmented = True
            video.segmented_at = datetime.now(timezone.utc)
            video.segmentation_cost_usd = result.cost_usd
            video.status = VideoStatus.processed
            if scene:
                video.scene = scene
            video.error_message = None
