"""Global processing settings — face-blur toggle and camera profile listing.

Lens undistortion is profile-driven (per camera_model); there are no tunable
knobs to expose — calibrate a camera to enable correction. See
pipeline/calibration_profiles/.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from config import settings
from pipeline.proc_flags import get_flags, set_flags
from pipeline.undistort import list_profiles
from .security import require_api_key

router = APIRouter(prefix="/api/settings", tags=["settings"],
                   dependencies=[Depends(require_api_key)])


class ProcessingFlags(BaseModel):
    face_blur: Optional[bool] = None


@router.get("/processing")
def get_processing():
    return get_flags()


@router.put("/processing")
def put_processing(flags: ProcessingFlags):
    return set_flags(**{k: v for k, v in flags.model_dump().items() if v is not None})


@router.get("/cameras")
def get_cameras():
    """Available camera calibration profiles (for the upload picker / dashboard)."""
    return {"profiles": list_profiles(), "default": settings.default_camera_model}
