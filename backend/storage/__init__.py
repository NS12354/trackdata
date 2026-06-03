"""Storage abstraction.

v1 ships a local-filesystem backend. The interface is intentionally small so we
can swap in an S3-backed implementation later without touching callers.
"""
from .base import Storage
from .local import LocalStorage
from .factory import get_storage

__all__ = ["Storage", "LocalStorage", "get_storage"]
