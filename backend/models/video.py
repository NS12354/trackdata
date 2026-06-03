"""The ``videos`` table — one row per uploaded clip and its processing state."""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Enum, Float, Integer, String, Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column

from db import Base


class VideoStatus(str, enum.Enum):
    uploaded = "uploaded"
    processing = "processing"
    anonymized = "anonymized"
    processed = "processed"
    failed = "failed"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Video(Base):
    __tablename__ = "videos"

    # UUID string (also used as the storage filename stem).
    id: Mapped[str] = mapped_column(String(36), primary_key=True)

    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    operator_id: Mapped[str] = mapped_column(String(128), nullable=False, default="default-operator")
    # Anonymized/opaque worker identifier (never the worker's real name).
    worker_id_anonymized: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # Property tag the operator associates on upload (Phase 4 per-property metrics).
    property_tag: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    status: Mapped[VideoStatus] = mapped_column(
        Enum(VideoStatus, native_enum=False, length=16),
        default=VideoStatus.uploaded,
        nullable=False,
    )

    file_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # --- Anonymization metadata (Phase 1) ---
    anonymized_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # Fraction of frames in which at least one face was detected & blurred, plus
    # mean detections/frame. Stored as JSON-encoded text for flexibility.
    anonymization_coverage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    anonymization_method: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    anonymization_meta: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # --- Hand pose metadata (Phase 2) ---
    hand_pose_extracted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    hand_pose_extracted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # --- Segmentation metadata (Phase 3) ---
    segmented: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    segmented_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    segmentation_cost_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Populated on failure for surfacing in the UI.
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "original_filename": self.original_filename,
            "uploaded_at": self.uploaded_at.isoformat() if self.uploaded_at else None,
            "operator_id": self.operator_id,
            "worker_id_anonymized": self.worker_id_anonymized,
            "property_tag": self.property_tag,
            "status": self.status.value if isinstance(self.status, VideoStatus) else self.status,
            "file_size": self.file_size,
            "duration_seconds": self.duration_seconds,
            "anonymized_at": self.anonymized_at.isoformat() if self.anonymized_at else None,
            "anonymization_coverage": self.anonymization_coverage,
            "anonymization_method": self.anonymization_method,
            "hand_pose_extracted": self.hand_pose_extracted,
            "segmented": self.segmented,
            "segmentation_cost_usd": self.segmentation_cost_usd,
            "error_message": self.error_message,
        }
