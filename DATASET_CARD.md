# Revisent Ego-Manipulation Dataset

**Version:** 0.1.0  ·  **Schema:** `body_pose/1.0`

Egocentric human **manipulation demonstrations** for robot imitation learning /
VLA training. Captured from chest-mounted cameras worn by workers in real
industrial settings (waste, demanufacturing). The value: in-the-wild manipulation
at a scale and diversity teleoperation can't reach, with **honest per-joint
provenance** so you can filter to exactly the joints you trust.

## Contents at a glance
- **Episodes:** 2
- **Total frames (pose):** 697
- **Total footage:** 0.0 hours
- **Skeleton:** 17-joint (`revisent-ego-17`, SMPL-compatible names)
- **Hands:** 21-point per hand (MediaPipe order)

## Modalities
| Stream | Description |
|---|---|
| `observation.images.ego` | Anonymized egocentric video (faces blurred, audio removed) |
| `observation.state` | Per-frame body joints (meters, body frame) |
| `observation.confidence` | Per-joint confidence (0–1) |
| `action` | Both-wrist end-effector targets (xyz) |
| hands | 21-pt per-hand keypoints |

## Format options
- **LeRobot v2** dataset (lead format; π0 / HF ecosystem)
- **Raw joints** parquet + `skeleton.json` (universal, no license)
- **SMPL-X** parameters (integration shipped; license-gated weights — see README)

## Benchmark (accuracy)
**Hands — the measured, sellable signal (PA-MPJPE):**
- **Egocentric (on-point): 22.6 mm** on `AssemblyHands` head-cam (400 hands) — head/chest-cam relevant.
- Third-person cross-check: 15.02 mm on `FreiHAND` (377 hands, 94% detection).
- Detector = MediaPipe Hands (a fast general detector, not task-SOTA: FreiHAND SOTA ≈7 mm; ego is much harder). Numbers verified by per-joint sanity.


*PA-MPJPE is the fair metric for monocular, height-scaled pose — quote it first.*

## Coordinate system & skeleton
- Units: **meters**; origin **pelvis**; **+X** right, **+Y** up, **+Z** forward (right-handed).
- Joint order, kinematic tree, and provenance legend ship in `skeleton.json`.

## Provenance (what's measured vs inferred)
A chest camera measures some of the body and **infers the rest** — we label every joint:
- `measured` — directly observed (hands).
- `oriented` — position templated, **orientation measured** by torso visual odometry.
- `ik` — solved by inverse kinematics from measured joints (arms).
- `inferred` — anthropometric prior; **never observed** by a chest cam (head, legs).

## Privacy
> Privacy metrics not yet measured. Run `scripts/validate_privacy.py` on a labeled set and regenerate.

## Known limitations (read this)
- Monocular: metric scale comes from operator height, not absolute depth; wrist
  depth along the image ray is reach-bounded.
- A chest camera never sees the wearer's **head or legs** — those joints are priors.
- Visual odometry drifts over long clips; torso orientation is most reliable short-term.

## How to load
```python
# Raw joints (no dependencies beyond pandas/pyarrow):
import pandas as pd
df = pd.read_parquet("pose_joints.parquet")
print(df.query("provenance == 'measured'").head())

# LeRobot:
# pip install lerobot
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# ds = LeRobotDataset(repo_id="local", root="path/to/dataset")
```

## License & consent
Operator-managed consent; training/redistribution rights are explicit per clip in
each export manifest's `capture_metadata.consent`. Do not share externally without
the attached consent reference.
