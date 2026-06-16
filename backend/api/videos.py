"""Video ingestion & retrieval endpoints (Phase 1)."""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter, Depends, File, Form, HTTPException, Request, UploadFile,
)
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import settings
from db import db_dependency
from models import Video, VideoStatus
from pipeline.jobs import (
    _anonymized_key, _hand_pose_key, _segments_key, _head_pose_key, _body_pose_key,
    delete_video_artifacts,
)
from pipeline.hand_pose import load_hand_pose, read_hand_pose_metadata
from pipeline.segmentation import load_segments
from pipeline.events import summarize_video
from pipeline.export import build_export
from storage import get_storage
from tasks import (
    anonymize_video_task, extract_hand_pose_task, segment_video_task,
    extract_events_task,
)
from .security import require_api_key, validate_video_upload

log = logging.getLogger("revisent.api")
# Auth is enforced on every route here (no-op when no key is configured in dev).
router = APIRouter(
    prefix="/api/videos", tags=["videos"], dependencies=[Depends(require_api_key)]
)

CHUNK = 4 * 1024 * 1024  # 4 MB


@router.post("", status_code=201)
async def upload_video(
    file: UploadFile = File(...),
    operator_id: str = Form("default-operator"),
    worker_id_anonymized: Optional[str] = Form(None),
    property_tag: Optional[str] = Form(None),
    operator_height_cm: Optional[int] = Form(None),
    camera_model: str = Form(""),
    db: Session = Depends(db_dependency),
):
    """Accept an mp4/mov upload, persist it, and trigger async anonymization."""
    original = file.filename or "upload"
    ext = Path(original).suffix.lower()

    video_id = str(uuid.uuid4())
    storage = get_storage()
    upload_key = f"uploads/{video_id}{ext}"
    dest = storage.local_path(upload_key)

    # Stream to disk in chunks, enforcing the size cap as we go. The first chunk
    # is sniffed (magic bytes) to confirm it's really an mp4/mov before we accept
    # untrusted media for ffmpeg/OpenCV to parse.
    size = 0
    first = True
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                if first:
                    validate_video_upload(ext, chunk[:512])  # raises 400 if invalid
                    first = False
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    out.close()
                    os.remove(dest)
                    raise HTTPException(
                        status_code=413,
                        detail=f"file exceeds max upload size of {settings.max_upload_bytes} bytes",
                    )
                out.write(chunk)
    except HTTPException:
        if dest.exists():
            os.remove(dest)
        raise
    except Exception as exc:  # noqa: BLE001
        if dest.exists():
            os.remove(dest)
        log.error("upload write failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"failed to store upload: {exc}")
    finally:
        await file.close()

    if size == 0:
        if dest.exists():
            os.remove(dest)
        raise HTTPException(status_code=400, detail="empty file")

    video = Video(
        id=video_id,
        original_filename=original,
        operator_id=operator_id,
        worker_id_anonymized=worker_id_anonymized,
        property_tag=property_tag,
        operator_height_cm=operator_height_cm,
        camera_model=(camera_model.strip() or settings.default_camera_model),
        status=VideoStatus.uploaded,
        file_size=size,
    )
    db.add(video)
    db.commit()
    db.refresh(video)

    # Enqueue durable processing (Celery). The worker anonymizes, then chains
    # hand-pose extraction. The API returns immediately.
    anonymize_video_task.delay(video_id, upload_key)

    log.info("uploaded video %s (%s, %d bytes)", video_id, original, size)
    return video.to_dict()


@router.get("")
def list_videos(db: Session = Depends(db_dependency)):
    rows = db.execute(select(Video).order_by(Video.uploaded_at.desc())).scalars().all()
    return [v.to_dict() for v in rows]


@router.get("/{video_id}")
def get_video(video_id: str, db: Session = Depends(db_dependency)):
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="video not found")
    return video.to_dict()


@router.delete("/{video_id}", status_code=200)
def delete_video(video_id: str, db: Session = Depends(db_dependency)):
    """Cancel and/or delete a video at any stage.

    Deleting the DB row is the cancellation signal: any in-flight pipeline job
    notices on its next checkpoint and stops cooperatively (the worker is solo-
    pool, so we can't terminate a single task). We then remove all on-disk
    artifacts. Idempotent — deleting an already-gone video returns ok.
    """
    video = db.get(Video, video_id)
    if video is None:
        # Already gone (or never existed) — treat as success so the UI can
        # delete optimistically without racing the worker.
        return {"id": video_id, "deleted": True, "already_gone": True}

    was_processing = video.status in (VideoStatus.uploaded, VideoStatus.processing)

    # 1) Drop the row first — this is what the running job polls to self-cancel.
    db.delete(video)
    db.commit()

    # 2) Remove all files (best-effort; safe even if a job is mid-write).
    try:
        delete_video_artifacts(video_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("artifact cleanup incomplete for %s: %s", video_id, exc)

    log.info("deleted video %s (was_processing=%s)", video_id, was_processing)
    return {"id": video_id, "deleted": True, "was_processing": was_processing}


def _ranged_file_response(path: Path, request: Request, media_type: str) -> Response:
    """Serve a file with HTTP range support so browsers can scrub/seek.

    Returns 206 Partial Content for a valid Range header, else a full 200.
    """
    file_size = path.stat().st_size
    range_header = request.headers.get("range")
    if not range_header or not range_header.startswith("bytes="):
        return FileResponse(path, media_type=media_type)

    # Parse "bytes=start-end" (end optional).
    try:
        spec = range_header.split("=", 1)[1].split(",")[0]
        start_s, _, end_s = spec.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except ValueError:
        return FileResponse(path, media_type=media_type)

    end = min(end, file_size - 1)
    if start > end or start >= file_size:
        return Response(
            status_code=416,
            headers={"Content-Range": f"bytes */{file_size}"},
        )

    length = end - start + 1

    def iter_chunk(chunk=1024 * 1024):
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    }
    return StreamingResponse(iter_chunk(), status_code=206, headers=headers, media_type=media_type)


@router.get("/{video_id}/anonymized")
def stream_anonymized(
    video_id: str, request: Request, db: Session = Depends(db_dependency)
):
    """Serve the anonymized video file (supports HTTP range for seeking)."""
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="video not found")
    storage = get_storage()
    key = _anonymized_key(video_id)
    if not storage.exists(key):
        raise HTTPException(status_code=404, detail="anonymized video not ready")
    path = storage.local_path(key)
    return _ranged_file_response(path, request, "video/mp4")


@router.post("/{video_id}/hand-pose", status_code=202)
def trigger_hand_pose(video_id: str, db: Session = Depends(db_dependency)):
    """(Re)run hand-pose extraction on the anonymized video."""
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="video not found")
    storage = get_storage()
    if not storage.exists(_anonymized_key(video_id)):
        raise HTTPException(status_code=409, detail="video is not anonymized yet")
    extract_hand_pose_task.delay(video_id)
    return {"status": "scheduled", "video_id": video_id}


@router.get("/{video_id}/hand-pose")
def get_hand_pose(video_id: str, db: Session = Depends(db_dependency)):
    """Return per-frame hand keypoints + sampling metadata (for the overlay)."""
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="video not found")
    storage = get_storage()
    key = _hand_pose_key(video_id)
    if not storage.exists(key):
        raise HTTPException(status_code=404, detail="hand pose not extracted yet")
    path = storage.local_path(key)
    return {
        "video_id": video_id,
        "metadata": read_hand_pose_metadata(path),
        "frames": load_hand_pose(path),
    }


@router.get("/{video_id}/head-pose")
def get_head_pose(video_id: str, db: Session = Depends(db_dependency)):
    """Per-frame head pose (position + orientation) from visual odometry."""
    if db.get(Video, video_id) is None:
        raise HTTPException(status_code=404, detail="video not found")
    storage = get_storage()
    key = _head_pose_key(video_id)
    if not storage.exists(key):
        raise HTTPException(status_code=404, detail="head pose not computed yet")
    import json as _json
    return _json.loads(storage.local_path(key).read_text())


@router.post("/{video_id}/head-pose", status_code=202)
def trigger_head_pose(video_id: str, db: Session = Depends(db_dependency)):
    """(Re)run head-pose visual odometry."""
    if db.get(Video, video_id) is None:
        raise HTTPException(status_code=404, detail="video not found")
    if not get_storage().exists(_anonymized_key(video_id)):
        raise HTTPException(status_code=409, detail="video is not anonymized yet")
    from pipeline.jobs import run_head_pose
    run_head_pose(video_id)
    return {"status": "done", "video_id": video_id}


@router.get("/{video_id}/body-pose")
def get_body_pose(video_id: str, db: Session = Depends(db_dependency)):
    """Per-frame skeletal pose (joints + per-joint confidence + provenance)."""
    if db.get(Video, video_id) is None:
        raise HTTPException(status_code=404, detail="video not found")
    storage = get_storage()
    key = _body_pose_key(video_id)
    if not storage.exists(key):
        raise HTTPException(status_code=404, detail="body pose not computed yet")
    import json as _json
    return _json.loads(storage.local_path(key).read_text())


@router.post("/{video_id}/body-pose", status_code=202)
def trigger_body_pose(video_id: str, db: Session = Depends(db_dependency)):
    """(Re)assemble the skeletal body-pose data product."""
    if db.get(Video, video_id) is None:
        raise HTTPException(status_code=404, detail="video not found")
    if not get_storage().exists(_anonymized_key(video_id)):
        raise HTTPException(status_code=409, detail="video is not anonymized yet")
    from pipeline.jobs import run_body_pose
    run_body_pose(video_id)
    return {"status": "done", "video_id": video_id}


@router.post("/{video_id}/segments", status_code=202)
def trigger_segmentation(video_id: str, db: Session = Depends(db_dependency)):
    """(Re)run task segmentation on the anonymized video."""
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="video not found")
    if not get_storage().exists(_anonymized_key(video_id)):
        raise HTTPException(status_code=409, detail="video is not anonymized yet")
    segment_video_task.delay(video_id)
    return {"status": "scheduled", "video_id": video_id}


@router.get("/{video_id}/segments")
def get_segments(video_id: str, db: Session = Depends(db_dependency)):
    """Return the task segments (start/end/label/confidence/description) + cost."""
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="video not found")
    storage = get_storage()
    key = _segments_key(video_id)
    if not storage.exists(key):
        raise HTTPException(status_code=404, detail="segments not extracted yet")
    return load_segments(storage.local_path(key))


@router.post("/{video_id}/events", status_code=202)
def trigger_events(video_id: str, db: Session = Depends(db_dependency)):
    """(Re)derive operational events from the video's segments."""
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="video not found")
    if not get_storage().exists(_segments_key(video_id)):
        raise HTTPException(status_code=409, detail="video is not segmented yet")
    extract_events_task.delay(video_id)
    return {"status": "scheduled", "video_id": video_id}


@router.get("/{video_id}/events")
def get_events(video_id: str, db: Session = Depends(db_dependency)):
    """Operational events + per-video summary (time-per-task, idle/downtime, ...)."""
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="video not found")
    return summarize_video(video_id)


@router.get("/{video_id}/export")
def export_bundle(video_id: str, db: Session = Depends(db_dependency)):
    """Build and download a structured export bundle (.zip) for the video."""
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="video not found")
    try:
        zip_path = build_export(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    fname = f"revisent_{video_id}.zip"
    return FileResponse(zip_path, media_type="application/zip", filename=fname)
