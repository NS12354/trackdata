"""Local-filesystem storage backend."""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO


class LocalStorage:
    """Maps relative keys onto a root directory on the local filesystem."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _full(self, key: str) -> Path:
        # Guard against path traversal escaping the root.
        full = (self.root / key).resolve()
        root = self.root.resolve()
        if root not in full.parents and full != root:
            raise ValueError(f"key {key!r} escapes storage root")
        return full

    def save_stream(self, key: str, stream: BinaryIO, *, chunk_size: int = 1024 * 1024) -> int:
        full = self._full(key)
        full.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(full, "wb") as out:
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
        return written

    def exists(self, key: str) -> bool:
        return self._full(key).exists()

    def delete(self, key: str) -> None:
        full = self._full(key)
        if full.exists():
            full.unlink()

    def size(self, key: str) -> int:
        return self._full(key).stat().st_size

    def local_path(self, key: str) -> Path:
        full = self._full(key)
        full.parent.mkdir(parents=True, exist_ok=True)
        return full

    def url(self, key: str) -> str:
        return key
