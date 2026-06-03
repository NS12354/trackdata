#!/usr/bin/env bash
#
# Expose your LOCAL backend to the internet so a Vercel-hosted dashboard can use
# it — keeping all processing free + local. Runs the API (auth-protected) and a
# Cloudflare quick tunnel, then prints the public URL + API key + the exact
# Vercel env vars to set.
#
#   ./scripts/serve_public.sh
#
# ⚠️  This publishes your backend (anonymized footage + metadata) to a public URL,
#     protected only by the API key below. Anyone with the URL + key can read it.
#     Stop it (Ctrl-C) when you're done. The quick-tunnel URL changes each run;
#     for a STABLE url see DEPLOYMENT.md (ngrok static domain or named tunnel).
#
set -euo pipefail
cd "$(dirname "$0")/.."

API_KEY="${API_KEY:-$(openssl rand -hex 24)}"
PORT="${PORT:-8000}"

command -v cloudflared >/dev/null || { echo "Install cloudflared first: brew install cloudflared"; exit 1; }
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

echo "→ opening Cloudflare tunnel…"
cloudflared tunnel --url "http://localhost:${PORT}" >/tmp/revisent_tunnel.log 2>&1 &
TUN_PID=$!

URL=""
for _ in $(seq 1 20); do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" /tmp/revisent_tunnel.log | head -1 || true)
  [ -n "$URL" ] && break
  sleep 1
done
[ -n "$URL" ] || { echo "Tunnel did not come up; see /tmp/revisent_tunnel.log"; exit 1; }

cat <<EOF

────────────────────────────────────────────────────────────────────
  ✅  Backend is live at:  $URL
  🔑  API key:             $API_KEY

  Set these in Vercel → Project → Settings → Environment Variables,
  then redeploy (and make sure Root Directory = frontend):

      NEXT_PUBLIC_API_BASE = $URL
      NEXT_PUBLIC_API_KEY  = $API_KEY

  Leave this terminal running. Ctrl-C to stop and take the backend offline.
────────────────────────────────────────────────────────────────────
EOF

wait
