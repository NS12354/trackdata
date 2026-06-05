#!/usr/bin/env python3
"""Sample bundle builder — the artifact you ship when a lab says "send a sample".

When a buyer asks, you have hours, not days. This assembles a complete, multi-
format, ready-to-ingest bundle from processed clips:

  <bundle>/
    DATASET_CARD.md         the card they read first (benchmark + provenance)
    QUICKSTART.md           one-command load instructions, tested
    manifest.json           exact contents, counts, formats, checksums
    lerobot/                LeRobot v2 dataset (lead format)
    raw/<clip>/             pose_joints.parquet + skeleton.json + smplx_README.json
    videos/<clip>.mp4       anonymized egocentric video

Usage:
    python scripts/build_sample_bundle.py --all
    python scripts/build_sample_bundle.py --videos <id1> <id2> --zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
DATA = ROOT / "data"

QUICKSTART = """# Quickstart — Revisent Ego-Manipulation sample

## Raw joints (zero special deps)
```bash
pip install pandas pyarrow
```
```python
import pandas as pd
df = pd.read_parquet("raw/<clip>/pose_joints.parquet")
# keep only directly-measured joints (the hands):
print(df[df.provenance == "measured"].head())
```
Joint order, kinematic tree, and coordinate system are in `raw/<clip>/skeleton.json`.

## LeRobot (lead format — verified loadable in vanilla lerobot 0.5.1, v3.0)
```bash
pip install lerobot
```
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="revisent/ego-manipulation", root="lerobot")
print(ds.num_episodes, ds.num_frames)   # 3 episodes, 1302 frames
ds[0]   # observation.state (51), observation.confidence (17), action (6), task, ...
```

## SMPL-X (parametric, license-gated)
See `raw/<clip>/smplx_README.json` for the joint mapping and how to enable fitting
(register at smpl-x.is.tue.mpg.de, `pip install smplx torch`, set SMPLX_MODEL_DIR).

## What to trust
Every joint carries `confidence` + `provenance`. `measured` = hands (observed);
`oriented`/`ik` = torso/arms; `inferred` = head/legs (a chest cam never sees them —
priors, not measurements). Filter by provenance for your use case.
"""


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build(video_ids, out_dir: Path, do_zip: bool):
    from pipeline.pose_export import load_body_doc, write_raw_pose
    from pipeline.smplx_export import write_smplx_readme
    from pipeline.lerobot_export import build_lerobot_dataset, validate_lerobot

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "raw").mkdir(parents=True)
    (out_dir / "videos").mkdir(parents=True)

    usable = [v for v in video_ids if (DATA / "processed" / v / "body_pose.json").exists()]
    if not usable:
        raise SystemExit("no processed clips with body_pose.json found")

    # LeRobot dataset across all clips. Prefer the OFFICIAL converter (to_lerobot.py)
    # run in a lerobot venv (python>=3.10) so the output loads in vanilla lerobot
    # (v3.0, verified). Fall back to the legacy v2.0 writer if no lerobot is found.
    import os, subprocess
    lerobot_py = os.environ.get("LEROBOT_PYTHON", "/tmp/lerobot-venv/bin/python")
    if Path(lerobot_py).exists():
        r = subprocess.run([lerobot_py, str(ROOT / "scripts" / "to_lerobot.py"),
                            "--out", str(out_dir / "lerobot")], capture_output=True, text=True)
        if r.returncode == 0:
            lerobot_summary = {"format": "lerobot-v3.0 (official API, loads in vanilla lerobot)"}
            lerobot_problems = "none — built + loaded with lerobot's own API"
        else:
            lerobot_summary = build_lerobot_dataset(usable, out_dir / "lerobot")
            lerobot_problems = [f"official converter failed, used legacy v2.0: {r.stderr[-200:]}"]
    else:
        lerobot_summary = build_lerobot_dataset(usable, out_dir / "lerobot")
        lerobot_problems = ["legacy v2.0 (set LEROBOT_PYTHON to a lerobot venv for the "
                            "verified-loadable v3.0 dataset)"]
        lerobot_problems += [p for p in validate_lerobot(out_dir / "lerobot") if not p.startswith("INFO")]

    contents = []
    for vid in usable:
        doc = load_body_doc(vid)
        raw_dir = out_dir / "raw" / vid
        write_raw_pose(doc, raw_dir)
        write_smplx_readme(raw_dir)
        anon = DATA / "anonymized" / f"{vid}.mp4"
        if anon.exists():
            shutil.copy(anon, out_dir / "videos" / f"{vid}.mp4")
        contents.append({
            "clip": vid,
            "frames": doc.get("frame_count"),
            "wrist_measured_fraction": doc.get("coverage", {}).get("wrist_measured_fraction"),
            "raw": [f"raw/{vid}/pose_joints.parquet", f"raw/{vid}/skeleton.json",
                    f"raw/{vid}/smplx_README.json"],
            "video": f"videos/{vid}.mp4" if anon.exists() else None,
        })

    # Dataset card (call the generator directly)
    sys.path.insert(0, str(ROOT))
    from scripts.build_dataset_card import build_card
    (out_dir / "DATASET_CARD.md").write_text(build_card("Revisent Ego-Manipulation (sample)", "0.1.0"))
    (out_dir / "QUICKSTART.md").write_text(QUICKSTART)

    manifest = {
        "bundle": "revisent-ego-manipulation-sample",
        "generated_at_unix": time.time(),
        "clips": len(usable),
        "formats": ["lerobot-v3.0", "raw-joints-parquet", "smplx-mapping"],
        "lerobot": lerobot_summary,
        "lerobot_structural_problems": lerobot_problems or "none",
        "contents": contents,
        "checksums": {},
        "notes": "Anonymized (faces blurred, audio removed). Per-joint provenance in "
                 "skeleton.json. See DATASET_CARD.md for benchmark + limitations.",
    }
    # checksums of key files
    for rel in ["DATASET_CARD.md", "QUICKSTART.md", "lerobot/meta/info.json"]:
        p = out_dir / rel
        if p.exists():
            manifest["checksums"][rel] = _sha256(p)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    n_files = sum(1 for _ in out_dir.rglob("*") if _.is_file())
    print(f"bundle: {out_dir}")
    print(f"  clips={len(usable)} files={n_files} lerobot={lerobot_summary}")
    print(f"  lerobot structural problems: {lerobot_problems or 'none'}")

    if do_zip:
        zip_path = out_dir.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(str(out_dir), "zip", out_dir)
        print(f"  zipped -> {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="*", default=[])
    ap.add_argument("--all", action="store_true", help="use every processed clip")
    ap.add_argument("--out", default=str(DATA / "exports" / "sample_bundle"))
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    vids = args.videos
    if args.all or not vids:
        vids = sorted(p.parent.name for p in (DATA / "processed").glob("*/body_pose.json"))
    build(vids, Path(args.out), args.zip)


if __name__ == "__main__":
    main()
