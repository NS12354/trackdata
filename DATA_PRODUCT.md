# Revisent — Egocentric Manipulation Data Product

The skeletal/manipulation **data product** for robot-learning buyers (Physical
Intelligence, NVIDIA, Meta FAIR, 1X) and the vertical-automation companies
deploying into waste / demanufacturing. This doc is the map: what's built, how to
run it, and the handful of steps that need **you**.

## The thesis (one paragraph)
A wearable egocentric camera (chest or forehead mount; recorded per clip)
**measures** the hands (in view) and the rig's motion, and **infers** the unseen
parts (legs; head on a chest rig). We don't pretend
otherwise — every joint ships with a `confidence` and a `provenance` flag. The
sellable wedge is **in-the-wild egocentric manipulation demonstrations** (hands +
arms + first-person video) at a scale and diversity teleoperation can't reach —
the exact data bottleneck for VLA / humanoid foundation models.

## What's built (all tested)

### Pipeline (backend/pipeline/)
| Module | What it produces |
|---|---|
| `body_pose.py` | Per-frame skeleton (17 joints) with confidence + provenance. Wrist placed by **camera-intrinsic back-projection** (2D measured, depth reach-bounded), arms by 2-bone IK, torso oriented by VO, head/legs as labeled priors. |
| `pose_export.py` | `pose_joints.parquet` (long format) + `skeleton.json` (joint order, kinematic tree, coord system, provenance legend). |
| `lerobot_export.py` | LeRobot **v2.1** dataset (segment-per-episode, imperative language tasks, gripper channels) + structural validator. For current lerobot (>=0.5, format v3.0) run the official converter: `python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=any/name --root=<dataset> --push-to-hub=false` — **verified loading end-to-end in lerobot 0.5.1** (env: `data/tmp_lerobot_env`). |
| `smplx_export.py` | SMPL-X joint mapping + per-frame fitting (license-gated weights documented, not shipped). |
| `capture_meta.py` | Per-clip capture record (camera mount, intrinsics, operator, consent) for the manifest. |

Wired into the pipeline (`jobs.py` / `tasks.py`): body pose runs after head+hand
pose. Served at `GET /api/videos/{id}/body-pose`; included in the export bundle.

### Tooling (scripts/)
| Script | Purpose |
|---|---|
| `benchmark_throughput.py` | Per-stage speed + 1000-hr GPU cost projection. |
| `eval_pose.py` | **MPJPE / PA-MPJPE** vs public ground truth, provenance-stratified. Synthetic self-test **passes**. |
| `eval_labeling.py` | Caption-accuracy benchmark (dormant labeling path). |
| `calibrate_camera.py` | Camera intrinsics from a checkerboard (non-optional for 3D data). |
| `validate_privacy.py` | Face-blur recall, hand false-blur rate, PII-miss rate. |
| `build_dataset_card.py` | `DATASET_CARD.md` — the artifact buyers read first. |
| `build_sample_bundle.py` | The complete "send under NDA" bundle (all formats + card + quickstart + manifest). |

## How to run it
```bash
cd revisent-mvp
PY=backend/.venv/bin/python

# regenerate pose for a processed clip + all exports
PYTHONPATH=backend $PY -m pipeline.jobs            # (run_body_pose is called in-pipeline)
$PY scripts/eval_pose.py synthetic                 # verify the metric core (PASS)
$PY scripts/build_sample_bundle.py --all --zip     # build the shippable bundle
$PY scripts/build_dataset_card.py                  # (re)generate the card
```

## The priority order to a sale (do these in this order)
1. **Benchmark on AssemblyHands** — the credential. `eval_pose.py assemblyhands --root ...`
2. **Fix anything the benchmark exposes** (the wrist/IK is improved but unmeasured vs GT).
3. **LeRobot bundle** — already built; `build_sample_bundle.py`.
4. **Privacy validation set** — the diligence killer; `validate_privacy.py`.
5. **Dataset card + sample bundle** — built; refresh after (1) and (4).
6. **SMPL-X** — secondary; enable when a buyer asks.

---

## ⚠️ Steps that require YOU (cannot be automated here)

1. **Download a public ground-truth dataset** (registration/license-gated) and run
   the benchmark — this is the single most important step:
   - AssemblyHands → https://assemblyhands.github.io (hand MPJPE)
   - Ego-Exo4D → https://ego-exo4d-data.org (arm/torso MPJPE)
   - `backend/.venv/bin/python scripts/eval_pose.py assemblyhands --root /path`
   - Then re-run `build_dataset_card.py` so the card quotes a real number.

2. **Calibrate the camera once** per camera model (print a checkerboard, ~20 shots):
   - `scripts/calibrate_camera.py --images calib/*.jpg --cols 9 --rows 6 --square-mm 25`
   - Writes `backend/camera_intrinsics.json`; the capture metadata + back-projection use it.

3. **Build a labeled privacy set** (~100 clips across lighting/motion/occlusion/gloves):
   - `scripts/validate_privacy.py bootstrap <clips> --out labels.json`, fill the arrays,
     then `score --manifest labels.json`. Card auto-reads the result.

4. **SMPL-X license** (only if a buyer wants parametric):
   - Register at https://smpl-x.is.tue.mpg.de, `pip install smplx torch`,
     `export SMPLX_MODEL_DIR=/path/to/models`. Fitting then runs.

5. **Real LeRobot load test** (optional, recommended before shipping):
   - `pip install lerobot` then load `data/exports/sample_bundle/lerobot` to confirm
     a clean parse on a fresh machine.

## Honest status
- The pose **structure, formats, benchmark harness, and packaging are production-grade.**
- The pose **accuracy is unmeasured against ground truth** — step (1) tells you whether
  the current kinematic model is good enough or whether to invest in a learned ego-body
  model (EgoEgo). Don't sell until you have that number. Don't soft-pedal it.

## Model stack assessment & upgrade paths (June 2026)

What runs today, what it's good for, and the next model up when quality demands it:

| Stage | Today | Honest grade | Upgrade path |
|---|---|---|---|
| Hands (2D + metric shape) | MediaPipe Hands (legacy solutions API), image + **world landmarks**, gap-fill + One-Euro | Good recall on clean footage; world landmarks give metric hand SHAPE (not camera depth) | MediaPipe Tasks `HandLandmarker` (newer, faster); **WiLoR / HaMeR** (transformer 3D hand mesh) when GPU budget exists — true metric 3D, robust to blur |
| Wrist depth | Depth-from-hand-scale (real palm size vs apparent px size, pinhole) with reach-bound fallback | A real measurement, accuracy bounded by intrinsics + hand-size estimate | Calibrated intrinsics (`calibrate_camera.py`) tightens it; metric mono-depth (Depth Anything v2) or RGB-D rig replaces it |
| Ego-motion | Monocular ORB visual odometry | Orientation reliable short-term (84-93% tracked on test clips); translation up-to-scale, drifts | Add IMU (phone capture apps export it); DPVO/DROID-SLAM class for production |
| Torso (head rig) | Yaw-only heading proxy from head VO, low-passed ~1s | Honest proxy, labeled `oriented` at reduced confidence | Torso IMU, or learned ego-body models (EgoEgo-class) |
| Faces (privacy) | YuNet + MediaPipe union, temporal tracks | Near-100% recall; known false-positives on face-like objects (plush toys) | Threshold tuning per environment; segmentation-based redaction |
| Activity labels | qwen2.5vl:7b via Ollama (local, $0), fused boundary detection | Specific, usable open-vocab descriptions; occasional camera-aware phrasing | Claude provider flag exists for quality passes; prompt tuning in `segmentation_providers.py` |

The intake **quality gate** (`pipeline/quality_gate.py`) fronts all of this:
orientation is decided by local-VLM votes on sampled frames (faces never work on
ego footage, and hand heuristics proved grip-style-confounded on real clips —
kept as advisory evidence only), plus exposure and blur screens — bad capture is
caught before it becomes plausible-looking bad data.
