# Revisent MVP — Video Processing & Operational Dashboard

Operational video-processing pipeline and dashboard for a waste-services pilot.
Chest-worn camera footage is uploaded daily, anonymized, and processed into
hand-pose, task-segment, and event extracts that drive an operator dashboard and
a natural-language chatbot — and double as clean structured extracts for research
conversations.

> Status: **v1 feature-complete** — Phases 1–5 + 7 done (Ingestion & Anonymization;
> Hand Pose; Task Segmentation; Event Extraction & Metrics; Dashboard; Export
> Bundle). Phase 6 chatbot deferred by choice. **All processing runs locally at
> $0 — no per-video API cost.**

## Architecture

- **Backend** — Python / FastAPI API + **Celery** workers (`backend/`)
- **Frontend** — Next.js 14 + TypeScript dashboard (`frontend/`, Phase 5)
- **DB** — **PostgreSQL** in production (SQLite for local dev), via SQLAlchemy +
  **Alembic** migrations
- **Job queue** — **Celery + Redis** (durable, retryable, scalable). The API
  enqueues; workers run the heavy CV pipeline.
- **Storage** — local filesystem behind a small `Storage` interface (S3 backend
  is a future swap)
- **Processing** — CPU-friendly local models (YuNet + MediaPipe, OpenCV) + the
  Anthropic API
- **Deploy** — Docker Compose stack (api, worker, postgres, redis, flower)

```
revisent-mvp/
├── backend/
│   ├── main.py            FastAPI entry (health/ready/metrics, CORS, auth)
│   ├── celery_app.py      Celery application
│   ├── tasks.py           durable pipeline tasks (retries, chaining)
│   ├── config.py          settings (.env)
│   ├── api/               route handlers + security (auth, upload validation)
│   ├── pipeline/          anonymize, face_detector, hand_pose, video_meta, jobs
│   ├── models/            SQLAlchemy models (videos, events)
│   ├── ml_models/         model weights (YuNet ONNX)
│   ├── storage/           storage abstraction (local backend)
│   ├── db/                engine / session / init
│   ├── alembic/           database migrations
│   ├── observability.py   logging, health checks, metrics
│   ├── Dockerfile
│   └── tests/             unit + integration tests
├── frontend/              Next.js dashboard (Phase 5)
├── data/                  uploads / anonymized / processed / exports
├── scripts/               fetch_models, validate_anonymization, overlays, ...
├── docker-compose.yml     full stack
└── .github/workflows/     CI (pytest on Postgres)
```

## Setup (local dev)

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ../scripts/fetch_models.py      # YuNet weights
cp ../.env.example ../.env             # optional; defaults work for local dev

# Option A — no broker (tasks run inline):
CELERY_TASK_ALWAYS_EAGER=true uvicorn main:app --reload --port 8000

# Option B — with Redis broker + a worker (closer to production):
uvicorn main:app --reload --port 8000
celery -A celery_app.celery worker --loglevel=INFO --concurrency=2
```

Requires `ffmpeg` on PATH (H.264 re-encode + metadata probing).

## Deploy (Docker — production-style)

```bash
cp .env.docker.example .env            # set POSTGRES_PASSWORD, API_KEY, ...
docker compose up --build              # api, worker, postgres, redis
docker compose --profile monitoring up flower   # optional Celery UI :5555
```

The API container runs Alembic migrations on start, then serves via
gunicorn+uvicorn workers; the worker container runs Celery. `ENVIRONMENT=production`
requires `API_KEY` (the app refuses to start an unauthenticated production API).

### Production hardening (Phases 1–2)

- **Durable jobs** — Celery + Redis: tasks acknowledge late (re-delivered if a
  worker dies), retry transient failures with exponential backoff, and mark the
  video failed only after retries are exhausted. Anonymization failure is fatal
  (the video does not proceed); hand-pose failure is non-fatal.
- **Migrations** — Alembic (`alembic upgrade head`), no destructive `create_all`
  in production.
- **Security** — API-key auth on all `/api` routes (constant-time check; header
  or `?api_key=` for the `<video>` element); uploads validated by magic bytes,
  not just extension; CORS allowlist; non-root container; size cap enforced while
  streaming.
- **Observability** — structured JSON logs, `/health` (liveness), `/ready`
  (DB + Redis readiness), Prometheus `/metrics`.
- **Tests/CI** — unit + integration tests (API → eager Celery → DB), GitHub
  Actions running pytest against a real Postgres service.

## API (Phase 1)

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | liveness |
| `POST` | `/api/videos` | upload an mp4/mov (multipart `file`; optional `operator_id`, `worker_id_anonymized`, `property_tag`) — triggers async anonymization |
| `GET` | `/api/videos` | list all videos |
| `GET` | `/api/videos/{id}` | video detail + processing status |
| `GET` | `/api/videos/{id}/anonymized` | stream the anonymized (face-blurred) mp4 |
| `POST` | `/api/videos/{id}/hand-pose` | (re)run hand-pose extraction |
| `GET` | `/api/videos/{id}/hand-pose` | per-frame hand keypoints + sampling metadata |
| `POST` | `/api/videos/{id}/segments` | (re)run task segmentation |
| `GET` | `/api/videos/{id}/segments` | task segments (label/time/confidence/description) + cost |
| `POST` | `/api/videos/{id}/events` | (re)derive operational events |
| `GET` | `/api/videos/{id}/events` | events + per-video summary (time-per-task, idle/downtime, service, contamination) |
| `GET` | `/api/metrics/overview` | operator-wide rollup (hours, service events, downtime, contamination, per-property) |
| `GET` | `/api/videos/{id}/export` | download a structured export bundle (.zip) |

Pipeline chaining: a successful anonymization automatically triggers hand-pose
extraction. Hand-pose failure is non-fatal — the video stays usable, the flag
stays `false`, and the error is recorded on the row.

Upload flow: file is written to `data/uploads/<uuid>.<ext>`, a `videos` row is
created (`status=uploaded`), and a FastAPI background task anonymizes it
(`processing` → `anonymized`, or `failed`). The anonymized H.264 mp4 lands in
`data/anonymized/<uuid>.mp4`.

### Quick verification

```bash
# generate a synthetic clip (no real faces — coverage will read ~0%, honestly)
source backend/.venv/bin/activate
python scripts/make_test_video.py /tmp/test.mp4 3 30

# upload it
curl -F "file=@/tmp/test.mp4" -F "property_tag=Greystar-Maple" http://localhost:8000/api/videos

# poll status until "anonymized"
curl http://localhost:8000/api/videos
```

## Anonymization

Faces are detected with a **union of two detectors for redundancy** — **YuNet**
(`cv2.FaceDetectorYN`, primary) and **MediaPipe Face Detection** (secondary) — and
blurred with an OpenCV Gaussian blur. A face missed by one detector may be caught
by the other (`FACE_DETECTOR=union|yunet|mediapipe`). Output is re-encoded to
browser-friendly H.264 (`yuv420p`, `+faststart`) via an ffmpeg pipe. **Audio is
dropped in v1** — voices are PII we do not yet redact.

Each run records a **coverage metric** (fraction of frames with ≥1 detected
face), mean faces/frame, and method, stored on the video row — so low detection
is surfaced, not hidden. Per the privacy principle, raw uploads are deleted once
anonymization succeeds; set `RETAIN_RAW_UPLOADS=true` (default during dev) to keep
them.

**Anti-flicker (temporal tracking).** Per-frame detection alone leaves a face
un-blurred whenever the detector misses (motion blur, profile, partial face).
Because processing is offline/batch we run two passes: detect on every frame and
link detections into face *tracks*, then bridge short gaps (interpolate) and hold
the blur a few frames before/after a face appears — so a momentarily-undetected
face stays covered. Two coverage numbers are reported: `raw_detection_coverage`
(frames with a confirmed detection) and `coverage` (frames blurred after gap-fill).

**False-positive rejection (track confirmation).** Detectors occasionally fire on
hands/objects. Measured on real clips, those false positives top out around 0.71
(MediaPipe) / 0.67 (YuNet) while genuine faces reach ~0.90; so a face *track* is
only blurred if at least one of its detections clears its detector's confirm
threshold (`FACE_CONFIRM_CONFIDENCE` 0.74 for MediaPipe, `YUNET_CONFIRM_SCORE`
0.80 for YuNet). Tracks that never produce a strong detection are dropped as false
positives — this is what prevents a worker's hands or equipment from being
blurred. Candidate detection still runs at a low threshold so weak frames of a
real (already-confirmed) face are recovered by gap-filling. The result reports
`rejected_tracks` alongside confirmed ones. Use
`scripts/validate_anonymization.py` to measure coverage and (against labels) the
leak rate, so the privacy claim is a number, not a guess.

**Orientation (auto-corrected).** Phone videos store a rotation flag that players
honor but OpenCV ignores; the pipeline reads it via ffprobe. That metadata is
sometimes missing or wrong, so a clip can arrive sideways or upside-down — and to
fix that, the anonymizer also does **content-based orientation detection**: it
samples frames, tries all four rotations, and picks the one where face detection
is strongest (real footage has upright faces), overriding the metadata when it
disagrees. With no faces in frame (e.g. a chest-cam clip) it falls back to the
metadata rotation. The decoded frames are rotated upright before processing, so
the anonymized output and hand-pose landmarks are always correctly oriented
(`AUTO_ORIENT`, default on).

**Performance.** Anonymization is the heavy step (face detection + full-res blur +
H.264 re-encode on CPU). Three optimizations bring it to roughly **real-time** on
HD footage — a 26s 1080p clip processes in ~27s (~30 fps) on a laptop, even with
the two-detector union:
- detection on a downscaled copy (`FACE_DETECTION_MAX_DIM`, default 960px; verified recall-safe),
- detection only every Nth frame (`FACE_DETECTION_STRIDE`, default 2) with the
  temporal gap-fill bridging skipped frames (privacy-verified: still 0% false
  blur, 100% real-face coverage),
- skipped frames advance via the cheaper `grab()` instead of a full decode.

Workers scale horizontally (add `worker` replicas / `CELERY_CONCURRENCY`) for
throughput; GPU decode/inference is the next lever if needed.

## Hand pose (Phase 2)

`pipeline/hand_pose.py` runs **MediaPipe Hands** on the *anonymized* video and
writes per-frame 21-point keypoints to `data/processed/{video_id}/hand_pose.parquet`.

- Frames are sampled at **10 fps** by default (`HAND_POSE_SAMPLE_FPS`) — CPU-cheap
  and smooth enough for the dashboard overlay. The effective sampling rate, source
  fps, stride, and model are embedded in the Parquet schema metadata.
- Landmarks are stored **normalized** (x,y in [0,1], z relative depth), so they
  overlay on any canvas size. Each hand also carries a confidence (handedness
  classification score). Absent hands are stored as nulls.
- Handedness: MediaPipe labels assume a mirrored selfie image; for a forward-facing
  chest cam the labels are swapped back (`HAND_POSE_SWAP_HANDEDNESS=true`).
- Coverage (fraction of sampled frames with ≥1 hand) is reported, so low hand
  visibility is surfaced rather than hidden.

## Dashboard (Phase 5)

Next.js 14 (App Router, TypeScript, Tailwind) in `frontend/`. Clean dark theme,
functional over fancy.

```bash
cd frontend
npm install
cp .env.local.example .env.local      # set NEXT_PUBLIC_API_BASE / _API_KEY if needed
npm run dev                           # http://localhost:3000
```

- **`/`** — operator overview: total hours, service events, downtime, contamination
  flags, recent videos, and a per-property breakdown.
- **`/videos`** — all uploads with status, duration, blur coverage, and a
  pipeline-progress indicator.
- **`/videos/[id]`** — detail: anonymized **video player** on the left with a
  toggleable **hand-pose canvas overlay** (21-point skeleton synced to the
  playhead via `requestAnimationFrame`); a clickable **task-segment timeline** on
  the right (click a segment to seek; the active segment highlights as it plays);
  and **event metrics** (time-per-task, service/downtime/contamination, active-hand
  time) below.

The API client (`lib/api.ts`) sends the API key via header, and the `<video>`
element via `?api_key=` (it can't set headers). The backend serves the anonymized
file with HTTP range support so scrubbing works.

## Task segmentation (Phase 3)

`pipeline/segmentation.py` samples the anonymized video at ~1fps and classifies
each frame against the waste-services task taxonomy (*approaching property, moving
container, opening gate/enclosure, manipulating lock/latch, handling
overflow/contamination, loading/unloading, transit/walking, idle/waiting*), then
aggregates consecutive same-task frames into segments with start/end times,
confidence, and a free-text description. Output:
`data/processed/{video_id}/segments.json`.

**Free, local, private by default.** The vision model runs through a pluggable
provider (`SEGMENTATION_PROVIDER`):
- `ollama` (default) — a **local** vision model via Ollama (`OLLAMA_VLM_MODEL`,
  default `qwen2.5vl:7b`, which follows the taxonomy well and emits structured
  JSON). **$0 cost and no frame ever leaves the machine.** Lighter option:
  `moondream` (~1.9B, faster) with `OLLAMA_USE_JSON=false` (caption + keyword
  mapping, since tiny models don't do structured output).
- `claude` — the Anthropic API (`ANTHROPIC_API_KEY`). Higher-quality labels, paid;
  opt-in only. Per-video cost is logged (`segmentation_cost_usd`).

**Honest quality note.** Small local VLMs caption well but classify weakly, so the
local path produces good descriptions and serviceable-but-noisy task labels. For
sharper labels, point `OLLAMA_VLM_MODEL` at a stronger local model (e.g.
`llava:7b`, `qwen2.5vl`) at some speed cost, or switch the provider to `claude`.
Every segment carries a confidence so low-certainty labels are visible.

## Event extraction & metrics (Phase 4)

`pipeline/events.py` turns the task segments (+ hand-pose) into operational
**events** in the `events` table — the data the dashboard and chatbot query. Pure
local computation, no cost. Each segment becomes an event typed as:

- `service` — container/lock/gate/load manipulation (the billable work)
- `contamination` — overflow/contamination handling (a flag)
- `idle` / `downtime` — idle ≥ `IDLE_DOWNTIME_SECONDS` (default 30s) is downtime
- `transit` — walking / approaching
- `task` — anything else

Events carry the video's `property_tag`, so per-property metrics fall out of a
query. `GET /api/videos/{id}/events` returns the events plus a per-video summary
(time-per-task, idle/downtime totals, service & contamination counts, and a
pose-derived `active_hand_seconds`). `GET /api/metrics/overview` rolls everything
up across videos (total hours, service events, downtime, contamination flags, and
a per-property breakdown). Re-running is idempotent (events are rebuilt, not
duplicated).

## Export bundle (Phase 7)

`pipeline/export.py` packages a processed video's extracts into a single `.zip`
under `data/exports/` — the lab-facing structured extract. An **Export bundle**
button on the video detail page downloads it (`GET /api/videos/{id}/export`).

```
{video_id}.zip
  ├── anonymized.mp4      face-blurred H.264 video (audio removed)
  ├── hand_pose.parquet   per-frame 21-point hand keypoints (normalized x,y,z)
  ├── segments.json       temporal task segments
  ├── events.json         derived operational events + summary
  └── manifest.json       contents, formats, capture metadata, processing
                          provenance (models/sampling/cost), and a consent reference
```

The manifest records capture metadata, anonymization method/coverage, hand-pose
and segmentation provenance, event counts, and a (operator-managed) consent
reference field. **Format is JSON/Parquet for v1.** LeRobot / RLDS / HDF5 native
output is planned future work (e.g. via a tool like Forge) and is intentionally
not built here.

## Privacy

- Anonymization is non-negotiable; the dashboard only ever serves the anonymized
  file, never the raw upload.
- `worker_id_anonymized` is an opaque tag — never a worker's real name.
- Raw uploads are deletable post-anonymization (`RETAIN_RAW_UPLOADS=false`).

## Future work (explicitly NOT in v1)

- **EgoBlur** (Meta) anonymization — preferred, but heavy PyTorch/detectron2 deps
  and large weights; deferred in favor of the MediaPipe fallback.
- Audio redaction.
- Ego-body pose imputation (EgoPoser / Ego-Exo4D).
- 6-DoF object pose tracking (FoundationPose / DOPE).
- Self-hosted large VLMs (Qwen2-VL) — using the Anthropic API instead.
- LeRobot / RLDS / HDF5 native export (planned Forge integration).
- Real-time/streaming processing, multi-tenancy/auth, production deploy, mobile.
```
