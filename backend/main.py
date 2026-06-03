"""Revisent backend — FastAPI application entry point.

Dev:        uvicorn main:app --reload --port 8000
Production: gunicorn main:app -k uvicorn.workers.UvicornWorker (see docker-entrypoint.sh)
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from api import videos_router, metrics_router
from config import settings
from db import init_db
from observability import configure_logging, check_database, check_redis

configure_logging()
log = logging.getLogger("revisent.main")

app = FastAPI(title="Revisent MVP", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    # Header-based auth (X-API-Key), not cookies — so credentials are off, which
    # also lets CORS_ORIGINS="*" work for browser uploads during dev/demo.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.metrics_enabled:
    # Instrument HTTP latency/counts and expose /metrics.
    try:
        from prometheus_client import make_asgi_app
        app.mount("/metrics", make_asgi_app())
    except Exception:  # noqa: BLE001
        log.warning("prometheus metrics unavailable")


@app.on_event("startup")
def _startup() -> None:
    # Fail fast on an insecure production config.
    if settings.is_production and not settings.api_key:
        raise RuntimeError(
            "ENVIRONMENT=production requires API_KEY to be set (refusing to start "
            "an unauthenticated production API)."
        )
    for d in (settings.uploads_dir, settings.anonymized_dir,
              settings.processed_dir, settings.exports_dir):
        d.mkdir(parents=True, exist_ok=True)
    init_db()
    log.info("startup complete (env=%s, detector=%s)", settings.environment, settings.face_detector)


@app.get("/health")
def health() -> dict:
    """Liveness — process is up."""
    return {"status": "ok", "version": app.version}


@app.get("/ready")
def ready(response: Response) -> dict:
    """Readiness — dependencies reachable (for k8s/compose health gating)."""
    db_ok = check_database()
    # Redis is only required where Celery isn't eager.
    redis_ok = True if settings.celery_task_always_eager else check_redis()
    ok = db_ok and redis_ok
    if not ok:
        response.status_code = 503
    return {"status": "ok" if ok else "degraded", "database": db_ok, "redis": redis_ok}


app.include_router(videos_router)
app.include_router(metrics_router)
