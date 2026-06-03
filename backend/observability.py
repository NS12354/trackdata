"""Logging, metrics, and health-check helpers.

- Structured JSON logs in production (LOG_JSON=true) for ingestion by a log
  aggregator; human-readable in dev.
- Prometheus counters/histograms for HTTP + task activity.
- Dependency health checks (DB, Redis) for readiness probes.
"""
from __future__ import annotations

import logging
import sys

import structlog

from config import settings


def configure_logging() -> None:
    """Configure stdlib + structlog. JSON renderer in production."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=shared + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (uvicorn, sqlalchemy, our modules) through the same
    # handler/level so everything is consistent.
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=level,
        stream=sys.stdout,
        force=True,
    )


# --- Health checks ---
def check_database() -> bool:
    from sqlalchemy import text
    from db import engine
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def check_redis() -> bool:
    try:
        import redis
        client = redis.from_url(settings.broker_url, socket_connect_timeout=2)
        return bool(client.ping())
    except Exception:
        return False
