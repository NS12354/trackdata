"""Operator-wide metrics endpoints (dashboard home page)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends

from pipeline.events import operator_overview
from .security import require_api_key

router = APIRouter(
    prefix="/api/metrics", tags=["metrics"], dependencies=[Depends(require_api_key)]
)


@router.get("/overview")
def overview(operator_id: Optional[str] = None):
    """Roll-up across all processed videos: total hours, service events, downtime,
    contamination flags, and a per-property breakdown."""
    return operator_overview(operator_id)
