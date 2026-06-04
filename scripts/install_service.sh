#!/usr/bin/env bash
#
# Run the backend + tunnel as a BACKGROUND SERVICE (launchd) — no terminal window.
# It starts at login and restarts if it crashes. Your Mac must be on and awake.
#
#   ./scripts/install_service.sh      # install + start
#   ./scripts/uninstall_service.sh    # stop + remove
#
# ⚠️  This keeps your backend (anonymized footage + metadata) reachable on your
#     ngrok URL whenever your Mac is on, protected only by the API key. That's a
#     bigger exposure than an occasional terminal session — uninstall when not
#     needed.
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.."; pwd)"
LABEL="com.revisent.serve"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"

[ -f "$REPO/scripts/serve_public.sh" ] || { echo "serve_public.sh missing"; exit 1; }

mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>exec "${REPO}/scripts/serve_public.sh"</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/revisent_service.log</string>
  <key>StandardErrorPath</key><string>/tmp/revisent_service.err</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
sleep 6

echo "✅ Service installed and started ($LABEL)."
echo "   It now runs in the background (no terminal) and restarts on login/crash."
echo "   Logs: /tmp/revisent_service.log   (errors: /tmp/revisent_service.err)"
echo
echo "Public URL (set once in Vercel, then never again):"
grep -oE "https://[a-z0-9.-]+\.ngrok-free\.(dev|app)" /tmp/revisent_tunnel.log 2>/dev/null | head -1 || \
  echo "  (starting… check /tmp/revisent_service.log in a few seconds)"
echo
echo "To stop/remove:  ./scripts/uninstall_service.sh"
echo "⚠️  Your Mac must stay on and awake for the site to work."
