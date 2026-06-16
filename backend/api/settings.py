"""Global processing settings — currently the lens de-warp (de-fisheye) params.

These are runtime overrides (persisted to data/undistort.json) that take effect on
the next processed clip without a server restart, so the UI slider is live.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from pipeline.undistort import get_runtime_params, set_runtime_params, is_calibrated
from .security import require_api_key

router = APIRouter(prefix="/api/settings", tags=["settings"],
                   dependencies=[Depends(require_api_key)])


class UndistortParams(BaseModel):
    enabled: Optional[bool] = None
    strength: Optional[float] = None     # 0..1 (estimate mode)
    zoom: Optional[float] = None         # 0..1
    fov_deg: Optional[float] = None      # assumed lens FOV (estimate mode)


def _payload() -> dict:
    p = get_runtime_params()
    p["calibrated"] = is_calibrated()
    p["mode"] = "calibrated" if p["calibrated"] else "estimate"
    return p


@router.get("/undistort")
def get_undistort():
    return _payload()


@router.put("/undistort")
def put_undistort(params: UndistortParams):
    kw = {k: v for k, v in params.model_dump().items() if v is not None}
    # Clamp to sane ranges.
    if "strength" in kw:
        kw["strength"] = max(0.0, min(1.0, float(kw["strength"])))
    if "zoom" in kw:
        kw["zoom"] = max(0.0, min(1.0, float(kw["zoom"])))
    if "fov_deg" in kw:
        kw["fov_deg"] = max(20.0, min(170.0, float(kw["fov_deg"])))
    set_runtime_params(**kw)
    return _payload()
