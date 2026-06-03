#!/usr/bin/env bash
# Container entrypoint. First arg selects the role: api | worker | flower | beat.
set -euo pipefail

run_migrations() {
  echo "[entrypoint] running database migrations..."
  alembic upgrade head
}

case "${1:-api}" in
  api)
    run_migrations
    echo "[entrypoint] starting API (gunicorn + uvicorn workers)..."
    exec gunicorn main:app \
      --workers "${WEB_CONCURRENCY:-2}" \
      --worker-class uvicorn.workers.UvicornWorker \
      --bind "0.0.0.0:${PORT:-8000}" \
      --timeout 120 \
      --access-logfile - --error-logfile -
    ;;
  worker)
    # Workers don't run migrations (the api/migrate service owns schema).
    echo "[entrypoint] starting Celery worker..."
    exec celery -A celery_app.celery worker \
      --loglevel="${CELERY_LOGLEVEL:-INFO}" \
      --concurrency="${CELERY_CONCURRENCY:-2}"
    ;;
  flower)
    exec celery -A celery_app.celery flower --port="${FLOWER_PORT:-5555}"
    ;;
  migrate)
    run_migrations
    ;;
  *)
    exec "$@"
    ;;
esac
