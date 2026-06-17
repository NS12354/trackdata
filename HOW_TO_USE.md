# How to Use This Folder — Revisent Operator Manual

The practical guide to running the egocentric training-data factory: record video
of yourself doing tasks → get curated, language-annotated, robot-trainable
episodes out. (Strategy/architecture live in `DATA_PRODUCT.md` and `README.md`;
this file is *how to drive it*.)

---

## 1. Start everything (one double-click)

```
scripts\start_revisent.cmd
```

Launches four windows: **Redis** (job queue, :6379) → **Celery worker** (runs the
GPU pipeline) → **API** (:8000) → **Dashboard** (http://localhost:3000).

- Heartbeat: the **top-right of every dashboard page** shows what the pipeline is
  doing right now (clip, stage, %, live detail) or `● pipeline idle`.
- Stop: close the spawned windows. To kill a stuck API:
  `powershell -File scripts\kill_uvicorn.ps1` (Celery-safe).

## 2. Get footage in

**Dashboard (normal path):** `localhost:3000/upload` — mp4/mov/avi/mkv/webm, up
to 5 GB. Set **wearer height in cm** (it scales every exported coordinate; ±5 cm
is fine, blank costs ~3% scale error). Upload returns instantly; the clip queues
and the page narrates each stage live, then auto-reveals results.

**CLI (power path — adds the quality gate & per-clip flags):**
```
backend\.venv\Scripts\python scripts\process_clip.py <video...> ^
    --mount head --height-cm 170 --no-blur --register
```
Useful flags: `--rotate 0|90|180|270` (override orientation), `--no-vlm` (skip
labeling), `--no-gate`, `--no-export`. `--register` puts results on the dashboard.

**Recording rules (the only human protocol left):**
- Film **real tasks** (cooking, assembly, folding) — hands-on-objects footage
  yields 3–4× more usable training signal than show-and-tell.
- Any orientation is fine (the gate auto-detects via VLM), any common phone.
- **Blur is OFF** (`BLUR_ENABLED=false` in `.env`) for solo footage. **Turn it
  back on before processing anyone else's face.**

## 3. Review and correct (this is where value concentrates)

Open a clip: anonymized player with live WiLoR hand skeleton, grasp chips in cm,
metric 3D hand panel, annotated activity timeline, metrics.

- **✂ edit cuts**: drag the white handles on the timeline bar — the video scrubs
  to the cut frame like a video editor. Stage several, then **Save N cuts ✓** or
  Cancel. Nothing persists without explicit save.
- **edit** on any segment row: fix the label/description/times inline. Saving
  marks it **human-verified ✓** (confidence 1.0).
- Every edit is logged to `data/annotations/edits.jsonl` (before→after, when) —
  your audit trail, and it feeds the eval ground truth automatically.
- Pipeline reruns **never overwrite human-verified segments** (delete the clip's
  `segments.json` to force re-annotation).

## 4. Export for robot-learning labs

Builds automatically per CLI run, or rebuild for chosen clips:
```
data\exports\lerobot_demo        ← LeRobot v2.1 (one episode per activity,
                                   imperative task strings, 8-dim actions
                                   incl. gripper, hands-free episodes filtered)
```
Convert to the current v3.0 format labs load directly:
```
data\tmp_lerobot_env\Scripts\python -m lerobot.scripts.convert_dataset_v21_to_v30 ^
    --repo-id=revisent/demo --root=<dataset-copy> --push-to-hub=false
```
(Verified loading end-to-end in `lerobot` 0.5.1.) Per-clip bundles: the
**Export bundle** button on any clip (video + hand parquet + segments + events
+ manifest).

## 5. Accuracy & benchmarks

- **Grasp calibration**: `scripts\calibrate_grasp.py` — mines thresholds from
  natural footage; no ritual needed. Re-run as the corpus grows.
- **Camera calibration (do once, 10 min)**: print `data\calibration\` board,
  film ~30 s, run `scripts\calibrate_camera.py` → wrist depth ±5%→~1%.
- **Palm length (do once)**: measure wrist-crease→middle-knuckle, set
  `WEARER_PALM_LENGTH_CM` in `.env`.
- **Boundary benchmark**: correct `data\eval\boundaries\manifest.json` once,
  then `scripts\eval_boundaries.py score --mode fused` → F1 + mean cut error,
  trend-logged in `data\eval\runs.jsonl`.
- **Pose benchmark**: `scripts\eval_pose.py` vs AssemblyHands (needs dataset
  download) — produces the PA-MPJPE number labs ask for first.

## 6. Settings that matter (`.env`)

| Setting | Current | Notes |
|---|---|---|
| `HAND_POSE_BACKEND` | `wilor` | GPU mesh model; occlusion-robust, measured wrist depth. `mediapipe` = CPU fallback |
| `HAND_POSE_SMOOTHER` | `kalman` | zero-lag prediction smoothing |
| `OLLAMA_VLM_MODEL` | `qwen3-vl:8b` | labeler; `SEGMENTATION_PROVIDER=claude` + API key = sale-grade labels |
| `BLUR_ENABLED` | `false` | **re-enable for any footage with bystanders** |
| `CAMERA_MOUNT` | `head` | forehead rig geometry |
| `CELERY_TASK_ALWAYS_EAGER` | `false` | queue mode; `true` = inline (no Redis needed, UI freezes) |
| `LEROBOT_MIN_HAND_FRACTION` | 0.05 | episode curation threshold |

## 7. Where things live

```
data\uploads\        raw uploads (retained)
data\anonymized\     processed video (always mp4, orientation-corrected)
data\processed\<id>\ hand_pose.parquet, body_pose.json, segments.json,
                     head_pose.json, capture.json, progress.json
data\annotations\    edits.jsonl (human-correction audit log)
data\exports\        LeRobot datasets + per-clip bundles
data\eval\           frozen benchmarks + run history
data\redis\          portable Redis (delete folder = uninstall)
data\tmp_wilor_env\  WiLoR GPU env   data\tmp_lerobot_env\  lerobot toolchain
```

## 8. Troubleshooting (lessons already paid for)

- **Upside-down/sideways video** → the gate warns and usually auto-fixes; force
  with `--rotate`. (Root cause was fixed: rotation-metadata sign convention.)
- **Dashboard slow during processing** → only in eager mode; with the queue
  running it shouldn't happen. Check the worker window is alive.
- **Worker dead?** Jobs wait safely in Redis; restart the worker and they resume.
- **GitHub downloads fail** (`CRYPT_E_NO_REVOCATION_CHECK`) → add
  `--ssl-no-revoke` to curl (this machine's TLS quirk).
- **Hand jitter at crossovers** → fixed in-pipeline (velocity tracking + physics
  gate); if seen again, screenshot with timestamp.
- **Never run uvicorn `--workers N` on Windows** — it orphans workers that serve
  stale code. One worker + the queue is the supported setup.
- Tests: `backend\.venv\Scripts\python backend\tests\<file>.py` (13 suites, all
  should pass).

## 9. The loop, in one line

**Record real tasks → upload (watch the heartbeat) → 2-minute correction pass →
verified episodes accumulate → export → benchmarks improve themselves.**
Footage volume and task diversity are the product now — everything else is
automated.
