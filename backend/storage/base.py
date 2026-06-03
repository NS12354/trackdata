"""Storage interface.

Paths are expressed as POSIX-style relative keys (e.g. "uploads/<uuid>.mp4").
A backend maps a key to a concrete location. Callers should never assume the
backend is local — use ``local_path`` only when a library genuinely needs a
filesystem path (OpenCV, ffmpeg), and treat it as a temporary materialization.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import BinaryIO


class Storage(abc.ABC):
    @abc.abstractmethod
    def save_stream(self, key: str, stream: BinaryIO, *, chunk_size: int = 1024 * 1024) -> int:
        """Write a binary stream to ``key``. Returns the number of bytes written."""

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        ...

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abc.abstractmethod
    def size(self, key: str) -> int:
        """Size in bytes of the object at ``key``."""

    @abc.abstractmethod
    def local_path(self, key: str) -> Path:
        """Return a filesystem path usable by local tools (OpenCV/ffmpeg).

        For the local backend this is the real, persistent path. An S3 backend
        would download to a temp file and return that.
        """

    @abc.abstractmethod
    def url(self, key: str) -> str:
        """A reference string for the object (relative key for local backend)."""
