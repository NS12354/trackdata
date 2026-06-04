#!/usr/bin/env bash
#
# Expose your LOCAL backend to the internet so a Vercel-hosted dashboard can use
# it — keeping all processing free + local. Runs the API (auth-protected) and a
# tunnel, then prints the public URL + API key + the exact Vercel env vars.
#
#   ./scripts/serve_public.sh
#
# Tunnel selection:
#   * If NGROK_DOMAIN is set (recommended) -> ngrok with a STABLE url. Set the
#     Vercel env once and never again. Configure your token first:
#         ngrok config add-authtoken <token>
#     then run:  NGROK_DOMAIN=your-name.ngrok-free.app ./scripts/serve_public.sh
#   * Otherwise -> a Cloudflare quick tunnel (no signup, but the url changes
#     every run, so you'd re-set the Vercel env each time).
#
# ⚠️  This publishes your backend (anonymized footage + metadata) to a public URL,
#     protected only by the API key below. Stop it (Ctrl-C) when you're done.
#
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
NGROK_DOMAIN="${NGROK_DOMAIN:-}"
# Fall back to a locally-saved domain (gitignored) so you can just run the script.
DOMAIN_FILE="backend/.serve_ngrok_domain"
if [ -z "$NGROK_DOMAIN" ] && [ -f "$DOMAIN_FILE" ]; then
  NGROK_DOMAIN="$(tr -d '[:space:]' < "$DOMAIN_FILE")"
fi

# Persist the API key so it's STABLE across runs (set Vercel env once). Override
# by exporting API_KEY. Stored locally, gitignored — never committed.
KEY_FILE="backend/.serve_api_key"
if [ -z "${API_KEY:-}" ]; then
  if [ -f "$KEY_FILE" ]; then
    API_KEY="$(cat "$KEY_FILE")"
  else
    API_KEY="$(openssl rand -hex 24)"
    echo "$API_KEY" > "$KEY_FILE"
  fi
fi

[ -d backend/.venv ] || { echo "Backend venv missing. See README setup."; exit 1; }

cleanup() { kill "${API_PID:-}" "${TUN_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

echo "→ starting backend (auth on, processing local)…"
(
  cd backend
  source .venv/bin/activate
  API_KEY="$API_KEY" CELERY_TASK_ALWAYS_EAGER=true CORS_ORIGINS="*" \
    uvicorn main:app --port "$PORT" --log-level warning
) &
API_PID=$!
sleep 4

URL=""
if [ -n "$NGROK_DOMAIN" ]; then
  command -v ngrok >/dev/null || { echo "Install ngrok: brew install ngrok"; exit 1; }
  echo "→ opening ngrok tunnel on https://${NGROK_DOMAIN} …"
  ngrok http "$PORT" --url "https://${NGROK_DOMAIN}" --log /tmp/revisent_tunnel.log --log-format json >/dev/null 2>&1 &
  TUN_PID=$!
  # Verify the tunnel actually established (ngrok's local API), don't assume it.
  URL=""
  for _ in $(seq 1 15); do
    if curl -s http://localhost:4040/api/tunnels 2>/dev/null | grep -q "${NGROK_DOMAIN}"; then
      URL="https://${NGROK_DOMAIN}"; break
    fi
    sleep 1
  done
  if [ -z "$URL" ]; then
    echo "✗ ngrok failed to connect. Recent log:"
    tail -15 /tmp/revisent_tunnel.log 2>/dev/null
    echo "Common causes: wrong domain, authtoken not set (ngrok config add-authtoken …),"
    echo "or the domain belongs to a different ngrok account."
    exit 1
  fi
else
  command -v cloudflared >/dev/null || { echo "Install cloudflared: brew install cloudflared (or set NGROK_DOMAIN)"; exit 1; }
  echo "→ opening Cloudflare quick tunnel (ephemeral url)…"
  cloudflared tunnel --url "http://localhost:${PORT}" >/tmp/revisent_tunnel.log 2>&1 &
  TUN_PID=$!
  for _ in $(seq 1 20); do
    URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" /tmp/revisent_tunnel.log | head -1 || true)
    [ -n "$URL" ] && break
    sleep 1
  done
fi
[ -n "$URL" ] || { echo "Tunnel did not come up; see /tmp/revisent_tunnel.log"; exit 1; }

cat <<EOF

────────────────────────────────────────────────────────────────────
  ✅  Backend is live at:  $URL
  🔑  API key:             $API_KEY

  Set these in Vercel → Project → Settings → Environment Variables,
  then redeploy (Root Directory = frontend, Framework = Next.js):

      NEXT_PUBLIC_API_BASE = $URL
      NEXT_PUBLIC_API_KEY  = $API_KEY

$( [ -n "$NGROK_DOMAIN" ] && echo "  (Stable url — set Vercel env once; just rerun this script next time.)" \
                          || echo "  (Ephemeral url — changes each run. Use NGROK_DOMAIN for a stable one.)" )

  Leave this terminal running. Ctrl-C to stop and take the backend offline.
────────────────────────────────────────────────────────────────────
EOF

wait
