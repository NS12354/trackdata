"""The ``events`` table — derived operational events (populated in Phase 4).

Defined now so the schema is stable; remains empty until event extraction lands.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from db import Base


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(String(36), ForeignKey("videos.id"), nullable=False, index=True)

    # e.g. "task", "idle", "service", "contamination"
    type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(256), nullable=False)

    start_time: Mapped[float] = mapped_column(Float, nullable=False)  # seconds into video
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)

    property_tag: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "video_id": self.video_id,
            "type": self.type,
            "label": self.label,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "property_tag": self.property_tag,
            "description": self.description,
        }
