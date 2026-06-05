#!/usr/bin/env python3
"""Dataset card generator — the artifact a buyer reads BEFORE evaluating the data.

Scans processed clips, aggregates totals, pulls the latest benchmark + privacy
numbers from the eval logs, and emits DATASET_CARD.md (Hugging-Face style). The
benchmark number is the credibility anchor; the provenance + limitations section
is the honesty that earns trust with sophisticated buyers.

Usage:
    python scripts/build_dataset_card.py --name "Revisent Ego-Manipulation v0.1"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
DATA = ROOT / "data"


def _last_run(log_path: Path, exclude_datasets=("synthetic-selftest",)):
    """Last logged run, skipping self-test entries (never quote a fake benchmark)."""
    if not log_path.exists():
        return None
    last = None
    for line in log_path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("dataset") in exclude_datasets:
            continue
        last = rec
    return last


def _scan_clips():
    """Return (episodes, total_frames, total_seconds) from processed body_pose.json."""
    from pipeline.video_meta import probe
    eps, frames, secs = 0, 0, 0.0
    for bp in sorted((DATA / "processed").glob("*/body_pose.json")):
        try:
            doc = json.loads(bp.read_text())
        except Exception:
            continue
        eps += 1
        frames += doc.get("frame_count", 0)
        anon = DATA / "anonymized" / f"{bp.parent.name}.mp4"
        if anon.exists():
            try:
                secs += probe(anon).duration_seconds or 0.0
            except Exception:
                pass
    return eps, frames, secs


def build_card(name: str, version: str) -> str:
    from pipeline.body_pose import JOINT_NAMES, SCHEMA_VERSION
    eps, frames, secs = _scan_clips()
    hours = secs / 3600.0
    def _latest_per_dataset(log_path):
        out = {}
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[r.get("dataset")] = r
        return out

    pose = _last_run(DATA / "eval" / "pose_runs.jsonl")
    hands = _latest_per_dataset(DATA / "eval" / "hand_runs.jsonl")
    ego = hands.get("AssemblyHands-ego")
    frei = hands.get("FreiHAND")
    priv = _last_run(DATA / "eval" / "privacy_runs.jsonl")

    def bench_block():
        out = []
        if ego or frei:
            out.append("**Hands — the measured, sellable signal (PA-MPJPE):**")
            if ego:
                out.append(f"- **Egocentric (on-point): {ego['pa_mpjpe_mm']} mm** on "
                           f"`AssemblyHands` head-cam ({ego['n']} hands) — head/chest-cam relevant.")
            if frei:
                dr = int(round(frei.get("detection_rate", 0) * 100))
                out.append(f"- Third-person cross-check: {frei['pa_mpjpe_mm']} mm on "
                           f"`FreiHAND` ({frei['n']} hands, {dr}% detection).")
            out.append("- Detector = MediaPipe Hands (a fast general detector, not task-SOTA: "
                       "FreiHAND SOTA ≈7 mm; ego is much harder). Numbers verified by per-joint sanity.")
            out.append("")
        if pose and pose.get("dataset") != "synthetic-selftest":
            out += [
                "**Full skeleton (vs egocentric GT):**",
                f"- PA-MPJPE {pose['pa_mpjpe_mm']} mm on `{pose['dataset']}` ({pose['n_samples']} samples)",
            ]
            bp = pose.get("by_provenance_pa_mm") or {}
            if bp:
                out.append("- By provenance (PA-MPJPE mm):")
                for k in ("measured", "ik", "oriented", "inferred"):
                    if k in bp:
                        out.append(f"    - `{k}`: {bp[k]} mm")
        if not out:
            return ("> **Benchmark: not yet run.** Run `scripts/eval_hand_freihand.py` (hands) "
                    "and `scripts/eval_pose.py` (body) against public ground truth, then "
                    "regenerate this card. **A frontier lab asks for this on the first call.**\n")
        return "\n".join(out) + "\n"

    def priv_block():
        if not priv:
            return ("> Privacy metrics not yet measured. Run "
                    "`scripts/validate_privacy.py` on a labeled set and regenerate.\n")
        return (f"- Face-blur recall: **{priv.get('face_recall', '?')}** "
                f"(faces correctly blurred)\n"
                f"- Hand false-blur rate: **{priv.get('hand_false_blur_rate', '?')}**\n"
                f"- PII-miss rate: **{priv.get('pii_miss_rate', '?')}**\n"
                f"- Labeled clips: {priv.get('n_clips', '?')}\n")

    return f"""# {name}

**Version:** {version}  ·  **Schema:** `{SCHEMA_VERSION}`

Egocentric human **manipulation demonstrations** for robot imitation learning /
VLA training. Captured from chest-mounted cameras worn by workers in real
industrial settings (waste, demanufacturing). The value: in-the-wild manipulation
at a scale and diversity teleoperation can't reach, with **honest per-joint
provenance** so you can filter to exactly the joints you trust.

## Contents at a glance
- **Episodes:** {eps}
- **Total frames (pose):** {frames:,}
- **Total footage:** {hours:.1f} hours
- **Skeleton:** {len(JOINT_NAMES)}-joint (`revisent-ego-17`, SMPL-compatible names)
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
{bench_block()}
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
{priv_block()}
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
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="Revisent Ego-Manipulation Dataset")
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--out", default=str(ROOT / "DATASET_CARD.md"))
    args = ap.parse_args()
    card = build_card(args.name, args.version)
    Path(args.out).write_text(card)
    print(f"wrote {args.out} ({len(card)} chars)")
    print("\n--- preview (head) ---")
    print("\n".join(card.splitlines()[:30]))


if __name__ == "__main__":
    main()
