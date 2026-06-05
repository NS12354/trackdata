#!/usr/bin/env bash
# One-time setup: make Redis + the Celery worker start automatically on login,
# so video uploads/processing keep working after a reboot with zero manual steps.
#
# Run once:   bash scripts/install_autostart.sh
# Undo:       bash scripts/install_autostart.sh --uninstall
set -euo pipefail

ROOT="/Users/nayansarma/Desktop/DataProcess/revisent-mvp"
PLIST_SRC="$ROOT/scripts/com.revisent.worker.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.revisent.worker.plist"

if [[ "${1:-}" == "--uninstall" ]]; then
  echo "Removing auto-start..."
  launchctl unload "$PLIST_DST" 2>/dev/null || true
  rm -f "$PLIST_DST"
  brew services stop redis 2>/dev/null || true
  echo "Done. Redis + worker will no longer auto-start."
  exit 0
fi

echo "1/2  Redis as a login service (auto-starts on every login)..."
redis-cli shutdown nosave 2>/dev/null || true   # free :6379 if a manual redis is up
brew services start redis

echo "2/2  Celery worker as a login service..."
pkill -f "celery_app.celery worker" 2>/dev/null || true   # stop any manual worker
mkdir -p "$HOME/Library/LaunchAgents"
cp "$PLIST_SRC" "$PLIST_DST"
launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"

sleep 3
echo
echo "Status:"
redis-cli ping >/dev/null 2>&1 && echo "  redis  : OK" || echo "  redis  : not responding (re-run in a few seconds)"
pgrep -f "celery_app.celery worker" >/dev/null && echo "  worker : running" || echo "  worker : starting (KeepAlive will retry until Redis is up)"
echo
echo "Done — Redis + the worker now start automatically on every login."
echo "Note: the API (:8000) and frontend (:3000) are separate; tell Claude if you"
echo "want those auto-started too."
