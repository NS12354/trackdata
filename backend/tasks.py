"""Celery task definitions — the durable, retryable processing pipeline.

Orchestration:
    upload  --enqueue-->  anonymize_video_task
                              |-- success --> extract_hand_pose_task
                              |-- exhausted retries --> mark video failed

Each task is idempotent (re-running overwrites its output), acknowledges late
(re-delivered if a worker dies mid-job), and retries transient failures with
exponential backoff before giving up.
"""
from __future__ import annotations

import logging

from celery_app import celery
from config import settings
from db import session_scope
from models import Video, VideoStatus
from pipeline.jobs import run_anonymization, run_hand_pose, run_segmentation, run_head_pose
from pipeline.events import extract_events

log = logging.getLogger("revisent.tasks")


def _record_failure(video_id: str, stage: str, exc: Exception, fatal: bool) -> None:
    """Persist a terminal failure on the video row."""
    try:
        with session_scope() as s:
            video = s.get(Video, video_id)
            if video is None:
                return
            video.error_message = f"{stage} failed after retries: {exc}"
            if fatal:
                video.status = VideoStatus.failed
    except Exception:  # noqa: BLE001
        log.exception("could not record %s failure for %s", stage, video_id)


@celery.task(
    bind=True,
    name="revisent.anonymize_video",
    autoretry_for=(Exception,),
    retry_backoff=True,         # exponential backoff between retries
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=settings.task_max_retries,
)
def anonymize_video_task(self, video_id: str, upload_key: str) -> str:
    """Anonymize an uploaded video, then chain hand-pose extraction.

    Anonymization is the privacy-critical, fatal-on-failure stage: if it can't
    complete, the video must NOT proceed and is marked failed.
    """
    try:
        run_anonymization(video_id, upload_key)
    except Exception as exc:
        if self.request.retries >= self.max_retries:
            log.error("anonymization permanently failed for %s: %s", video_id, exc)
            _record_failure(video_id, "anonymization", exc, fatal=True)
        raise  # triggers autoretry until retries exhausted

    # Chain the next stage as a separate, independently-retryable task.
    extract_hand_pose_task.delay(video_id)
    return video_id


@celery.task(
    bind=True,
    name="revisent.extract_hand_pose",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=settings.task_max_retries,
)
def extract_hand_pose_task(self, video_id: str) -> str:
    """Extract hand keypoints, then chain segmentation. Non-fatal on failure."""
    try:
        run_hand_pose(video_id)
    except Exception as exc:
        if self.request.retries >= self.max_retries:
            log.error("hand pose permanently failed for %s: %s", video_id, exc)
            _record_failure(video_id, "hand_pose", exc, fatal=False)
        raise
    # Head-pose visual odometry (best-effort; doesn't block the pipeline).
    try:
        run_head_pose(video_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("head pose failed for %s: %s", video_id, exc)
    # Chain segmentation as a separate, independently-retryable task.
    segment_video_task.delay(video_id)
    return video_id


@celery.task(
    bind=True,
    name="revisent.segment_video",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=settings.task_max_retries,
)
def segment_video_task(self, video_id: str) -> str:
    """Task-segment the video (local VLM by default). Non-fatal: failure leaves
    the video anonymized + hand-posed but not segmented. On success the video
    advances to 'processed'."""
    try:
        run_segmentation(video_id)
    except Exception as exc:
        if self.request.retries >= self.max_retries:
            log.error("segmentation permanently failed for %s: %s", video_id, exc)
            _record_failure(video_id, "segmentation", exc, fatal=False)
        raise
    # Chain event extraction (cheap, local) as the final enrichment stage.
    extract_events_task.delay(video_id)
    return video_id


@celery.task(
    bind=True,
    name="revisent.extract_events",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
    max_retries=settings.task_max_retries,
)
def extract_events_task(self, video_id: str) -> str:
    """Derive operational events from segments + pose. Non-fatal; pure local."""
    try:
        extract_events(video_id)
    except Exception as exc:
        if self.request.retries >= self.max_retries:
            log.error("event extraction permanently failed for %s: %s", video_id, exc)
            _record_failure(video_id, "events", exc, fatal=False)
        raise
    return video_id
