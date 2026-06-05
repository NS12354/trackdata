# Revisent — Egocentric Manipulation Data Product

The skeletal/manipulation **data product** for robot-learning buyers (Physical
Intelligence, NVIDIA, Meta FAIR, 1X) and the vertical-automation companies
deploying into waste / demanufacturing. This doc is the map: what's built, how to
run it, and the handful of steps that need **you**.

## The thesis (one paragraph)
A chest camera **measures** the hands (in view) and torso motion (the camera is on
the body), and **infers** the unseen parts (head, legs). We don't pretend
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
| `lerobot_export.py` | LeRobot **v2** dataset (the lead format; π0 / HF ecosystem) + structural validator. |
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
