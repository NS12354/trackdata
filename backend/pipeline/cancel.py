"""Cooperative job cancellation.

The Celery worker runs ``--pool=solo`` on macOS, so ``revoke(terminate=True)``
would kill the entire worker process rather than a single task. Instead we
cancel cooperatively: when a video row is deleted, the running pipeline notices
on its next checkpoint and raises ``JobCancelled``. Tasks treat this as a clean
stop — no retry, no chaining to the next stage.
"""
from __future__ import annotations


class JobCancelled(Exception):
    """Raised inside a pipeline job when its video has been deleted."""
