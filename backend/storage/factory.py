"""Storage factory — returns the configured backend."""
from __future__ import annotations

from functools import lru_cache

from config import settings
from .base import Storage
from .local import LocalStorage


@lru_cache(maxsize=1)
def get_storage() -> Storage:
    if settings.storage_backend == "local":
        return LocalStorage(settings.data_dir)
    raise NotImplementedError(f"storage_backend {settings.storage_backend!r} not supported in v1")
