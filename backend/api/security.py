"""API authentication and upload validation."""
from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Header, HTTPException, Query, status

from config import settings

# Allowed video container types (sniffed by magic bytes, not just extension).
# ffmpeg/OpenCV ingest all of these; pipeline output is always MP4 regardless.
ALLOWED_MIME = {"video/mp4", "video/quicktime", "video/x-m4v",
                "video/x-msvideo", "video/x-matroska", "video/webm"}
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


async def require_api_key(
    x_api_key: Optional[str] = Header(default=None),
    api_key: Optional[str] = Query(default=None),
) -> None:
    """FastAPI dependency enforcing the API key.

    Accepts the key via the ``X-API-Key`` header (normal API calls) or an
    ``?api_key=`` query parameter (so a browser ``<video>`` element, which can't
    set headers, can still authenticate to the stream endpoint).

    Auth is enforced whenever an API key is configured, and always in production
    (a production deploy with no key set is rejected at startup; see main.py).
    Uses a constant-time comparison to avoid timing oracles.
    """
    if not settings.auth_required:
        return
    provided = x_api_key or api_key
    if not provided or not hmac.compare_digest(provided, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid API key",
            headers={"WWW-Authenticate": "API-Key"},
        )


def sniff_video_type(head: bytes) -> str | None:
    """Return a sniffed MIME type for the first bytes of an upload, or None."""
    try:
        import filetype
        kind = filetype.guess(head)
        return kind.mime if kind else None
    except Exception:
        return None


def validate_video_upload(filename_ext: str, head: bytes) -> None:
    """Reject uploads that aren't actually an allowed video container.

    Extension alone is trivially spoofed; we also sniff the leading bytes so that
    untrusted media handed to ffmpeg/OpenCV is at least a real mp4/mov container.
    """
    if filename_ext.lower() not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported extension {filename_ext!r}; allowed: {sorted(ALLOWED_EXT)}",
        )
    if not settings.validate_upload_content:
        return
    mime = sniff_video_type(head)
    if mime is None or mime not in ALLOWED_MIME:
        raise HTTPException(
            status_code=400,
            detail=f"file content is not a supported video container (sniffed: {mime})",
        )
