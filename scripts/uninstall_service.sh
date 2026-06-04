#!/usr/bin/env bash
# Stop and remove the background backend+tunnel service installed by
# install_service.sh.
set -euo pipefail
LABEL="com.revisent.serve"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"

launchctl unload "$PLIST" 2>/dev/null || true
rm -f "$PLIST"
# Also stop any lingering processes it started.
pkill -f "ngrok http" 2>/dev/null || true
lsof -ti tcp:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
echo "✅ Service stopped and removed. The hosted site is now offline."
