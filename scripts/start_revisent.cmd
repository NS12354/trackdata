@echo off
REM ============================================================
REM  Revisent one-click launcher: Redis + Celery worker + API + UI
REM  Run from anywhere; everything starts in its own window.
REM ============================================================
set ROOT=%~dp0..
cd /d %ROOT%

REM 1) Redis (portable, project-local). Skipped if missing.
if exist "%ROOT%\data\redis\redis-server.exe" (
  start "revisent-redis" /min "%ROOT%\data\redis\redis-server.exe" --port 6379 --maxmemory 256mb
) else (
  echo [!] data\redis\redis-server.exe not found - queue will not work.
)

REM 2) Celery worker (GPU pipeline; solo pool = required on Windows and
REM    enforces one GPU job at a time).
start "revisent-worker" cmd /k "cd /d %ROOT%\backend && .venv\Scripts\celery.exe -A celery_app.celery worker --loglevel=INFO --pool=solo"

REM 3) Backend API.
start "revisent-api" cmd /k "cd /d %ROOT%\backend && .venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000"

REM 4) Frontend dashboard.
start "revisent-ui" cmd /k "cd /d %ROOT%\frontend && npm run dev"

echo.
echo Revisent starting: redis :6379, worker, api :8000, ui :3000
echo Close the spawned windows (or run scripts\kill_uvicorn.ps1) to stop.
