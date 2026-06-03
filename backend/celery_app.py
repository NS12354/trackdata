"""Celery application — the durable job queue (replaces FastAPI BackgroundTasks).

Why this exists: in-process background tasks are lost on restart, can't retry,
and have no concurrency control. Celery + Redis gives durable, retryable,
horizontally-scalable processing. Workers run the heavy CV pipeline; the API
just enqueues.

Run a worker:
    celery -A celery_app.celery worker --loglevel=INFO --concurrency=2
Monitor:
    celery -A celery_app.celery flower
"""
from __future__ import annotations

from celery import Celery

from config import settings

celery = Celery(
    "revisent",
    broker=settings.broker_url,
    backend=settings.result_backend,
    include=["tasks"],
)

celery.conf.update(
    # Reliability: a task is acknowledged only after it completes, so if a worker
    # dies mid-job the broker re-delivers it to another worker.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # CV jobs are long; one at a time per worker process keeps memory bounded and
    # lets us scale by adding worker replicas rather than threads.
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=20,  # recycle workers to bound any native leaks
    task_track_started=True,
    task_time_limit=60 * 60,         # hard ceiling: 1h per task
    task_soft_time_limit=55 * 60,
    result_expires=60 * 60 * 24,
    # Eager mode for tests / no-broker dev: run inline and propagate errors.
    task_always_eager=settings.celery_task_always_eager,
    task_eager_propagates=settings.celery_task_always_eager,
    timezone="UTC",
    broker_connection_retry_on_startup=True,
)
