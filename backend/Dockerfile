# Revisent backend image — shared by the API and the Celery workers.
# Build context is the repo root (see docker-compose.yml), so both backend/ and
# scripts/ are available.
FROM python:3.11-slim AS base

# System deps: ffmpeg (encode/probe) + libGL/glib for OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps first for layer caching.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/backend/requirements.txt

# App source + utility scripts (mirror the repo layout).
COPY backend /app/backend
COPY scripts /app/scripts

# Ensure model weights are present (no-op if already copied with the source).
RUN python /app/scripts/fetch_models.py || true

# Run as a non-root user; it owns the app and the data mount point.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data
USER appuser

ENV DATA_DIR=/data
WORKDIR /app/backend

EXPOSE 8000
ENTRYPOINT ["/app/backend/docker-entrypoint.sh"]
CMD ["api"]
