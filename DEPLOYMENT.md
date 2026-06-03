# Deployment

The app is two halves:

- **Frontend** (`frontend/`) — a Next.js app. Hosts fine on **Vercel**.
- **Backend** (`backend/`) — FastAPI + Celery + Postgres + Redis + **Ollama** +
  ffmpeg/OpenCV doing real video processing. **This cannot run on Vercel** (no
  long-running processes, no ffmpeg, no Ollama, no persistent storage). It needs
  a real server.

So the frontend always needs a **publicly reachable backend URL**. Pick a path:

---

## Why the Vercel deploy 404'd

Vercel looked for a Next.js app at the repo root, but it lives in `frontend/`.

**Fix:** Vercel → Project → **Settings → Build & Development → Root Directory →
`frontend`** → redeploy. (Do this for any of the paths below.)

---

## Path A — Frontend on Vercel + local backend via tunnel (free, private) ✅ recommended

Keeps all processing on your machine: $0, and no footage stored in the cloud.
Your laptop (and Ollama) must be running while the dashboard is in use.

1. **One command on your machine** (installs nothing beyond `cloudflared`):
   ```bash
   brew install cloudflared            # once
   ./scripts/serve_public.sh
   ```
   It starts the backend (auth-protected) + a tunnel and prints:
   ```
   NEXT_PUBLIC_API_BASE = https://<random>.trycloudflare.com
   NEXT_PUBLIC_API_KEY  = <generated key>
   ```
2. In **Vercel → Settings → Environment Variables**, add those two, set **Root
   Directory = `frontend`**, and **redeploy**.
3. Open your Vercel URL — the dashboard now talks to your local backend.

> ⚠️ The tunnel makes your backend reachable on the public internet, protected
> only by the API key. Stop `serve_public.sh` (Ctrl-C) when done.
>
> ⚠️ The quick-tunnel URL **changes every run**, so you'd re-set the Vercel env
> each time. For a **stable** free URL, use an **ngrok static domain**:
> ```bash
> ngrok config add-authtoken <token>          # free account
> ngrok http 8000 --domain=<your>.ngrok-free.app
> ```
> then set `NEXT_PUBLIC_API_BASE` to that fixed domain once. (Run the backend
> with `API_KEY=... CELERY_TASK_ALWAYS_EAGER=true uvicorn main:app --port 8000`.)

---

## Path B — Always-on cloud backend (no laptop required, costs money)

Run the Docker stack on a host that supports containers (a VM, **Render**,
**Railway**, **Fly.io**):

```bash
cp .env.docker.example .env     # set POSTGRES_PASSWORD, API_KEY, CORS_ORIGINS=<vercel domain>
docker compose up --build -d
```

Then set Vercel `NEXT_PUBLIC_API_BASE` to the host's public URL + the API key,
Root Directory = `frontend`, redeploy.

Caveats, honestly:
- **Ollama** (the free local segmentation model) needs a box with real CPU/RAM;
  free tiers won't run it well. Either run Ollama on a sized instance and point
  `OLLAMA_BASE_URL` at it, **or** switch `SEGMENTATION_PROVIDER=claude` (sets
  `ANTHROPIC_API_KEY`) — which reintroduces a small per-video cost.
- This stores footage in the cloud, changing the privacy posture vs. local.
- Set `ENVIRONMENT=production` and a strong `API_KEY` (the API refuses to start
  in production without one). Put TLS / a reverse proxy in front.

---

## Vercel env summary

| Variable | Value |
| --- | --- |
| Root Directory (project setting) | `frontend` |
| `NEXT_PUBLIC_API_BASE` | public backend URL (tunnel or host) |
| `NEXT_PUBLIC_API_KEY` | the backend's API key (if auth on) |

`NEXT_PUBLIC_*` vars are baked at build time — change them and **redeploy**.
