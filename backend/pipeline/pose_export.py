"""Raw per-joint pose export — the honest, format-agnostic fallback.

Two artifacts a research engineer can load with zero custom code:

  pose_joints.parquet  — long format, one row per (frame, joint):
      frame_idx, timestamp_ms, joint_id, joint_name, x, y, z,
      confidence, provenance, root_qw, root_qx, root_qy, root_qz, root_tracked
  skeleton.json        — joint ordering, kinematic tree, coordinate system,
                         units, provenance legend, camera mount. Everything
                         needed to interpret the parquet spatially.

Coordinates: meters, body frame, origin at pelvis, +X right, +Y up, +Z forward.
Hands (21-pt) ship in the companion hand_pose.parquet (image-normalized).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from .body_pose import JOINT_NAMES, SKELETON_EDGES, SCHEMA_VERSION

log = logging.getLogger("revisent.pose_export")

_JOINT_ID = {name: i for i, name in enumerate(JOINT_NAMES)}


def skeleton_spec(body_doc: dict) -> dict:
    """The self-describing skeleton spec (skeleton.json)."""
    return {
        "schema": SCHEMA_VERSION,
        "skeleton": "revisent-ego-17 (SMPL-compatible joint names)",
        "units": "meters",
        "coordinate_system": {
            "origin": "pelvis",
            "axes": "+X = subject right, +Y = up, +Z = forward",
            "handedness": "right-handed",
        },
        "joint_names": JOINT_NAMES,
        "joint_ids": _JOINT_ID,
        "kinematic_tree": [
            {"parent": p, "child": c, "parent_id": _JOINT_ID[p], "child_id": _JOINT_ID[c]}
            for p, c in SKELETON_EDGES
        ],
        "provenance_legend": body_doc.get("provenance_legend", {}),
        "camera_mount": body_doc.get("camera_mount", "chest"),
        "hands": {
            "note": "21-point per-hand landmarks ship in hand_pose.parquet "
                    "(MediaPipe order, image-normalized x,y,z).",
            "landmark_count": 21,
        },
        "limitations": [
            "Monocular: positions are metric-scaled by operator height, not "
            "absolute depth; wrist depth along the image ray is reach-bounded.",
            "A chest camera never observes the wearer's head or legs — those "
            "joints are anthropometric priors (provenance=inferred), not measured.",
        ],
    }


def write_raw_pose(body_doc: dict, out_dir: Path) -> dict:
    """Write pose_joints.parquet + skeleton.json into out_dir. Returns file info."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx, ts, jid, jname = [], [], [], []
    xs, ys, zs, conf, prov = [], [], [], [], []
    rqw, rqx, rqy, rqz, rtr = [], [], [], [], []

    for fi, f in enumerate(body_doc["frames"]):
        q = f["root_quaternion"]
        tracked = bool(f["root_tracked"])
        for name in JOINT_NAMES:
            j = f["joints"][name]
            frame_idx.append(fi)
            ts.append(float(f["timestamp_ms"]))
            jid.append(_JOINT_ID[name])
            jname.append(name)
            xs.append(float(j["pos"][0])); ys.append(float(j["pos"][1])); zs.append(float(j["pos"][2]))
            conf.append(float(j["confidence"])); prov.append(j["source"])
            rqw.append(float(q[0])); rqx.append(float(q[1])); rqy.append(float(q[2])); rqz.append(float(q[3]))
            rtr.append(tracked)

    table = pa.table({
        "frame_idx": pa.array(frame_idx, pa.int32()),
        "timestamp_ms": pa.array(ts, pa.float64()),
        "joint_id": pa.array(jid, pa.int16()),
        "joint_name": pa.array(jname, pa.string()),
        "x": pa.array(xs, pa.float32()), "y": pa.array(ys, pa.float32()), "z": pa.array(zs, pa.float32()),
        "confidence": pa.array(conf, pa.float32()),
        "provenance": pa.array(prov, pa.string()),
        "root_qw": pa.array(rqw, pa.float32()), "root_qx": pa.array(rqx, pa.float32()),
        "root_qy": pa.array(rqy, pa.float32()), "root_qz": pa.array(rqz, pa.float32()),
        "root_tracked": pa.array(rtr, pa.bool_()),
    })
    meta = {
        b"schema": SCHEMA_VERSION.encode(),
        b"units": b"meters",
        b"coord": b"pelvis-origin,+X-right,+Y-up,+Z-forward",
        b"video_id": str(body_doc.get("video_id", "")).encode(),
    }
    table = table.replace_schema_metadata(meta)
    parquet_path = out_dir / "pose_joints.parquet"
    pq.write_table(table, parquet_path)

    skel = skeleton_spec(body_doc)
    (out_dir / "skeleton.json").write_text(json.dumps(skel, indent=2))

    log.info("raw pose export: %d rows -> %s", table.num_rows, parquet_path)
    return {
        "pose_joints.parquet": parquet_path.stat().st_size,
        "skeleton.json": (out_dir / "skeleton.json").stat().st_size,
        "rows": table.num_rows,
    }


def load_body_doc(video_id: str, data_root: Optional[Path] = None) -> dict:
    """Load a stored body_pose.json (helper for exporters/tests)."""
    root = Path(data_root) if data_root else Path(__file__).resolve().parents[1].parent / "data"
    p = root / "processed" / video_id / "body_pose.json"
    if not p.exists():
        raise FileNotFoundError(f"body_pose.json missing for {video_id} ({p})")
    return json.loads(p.read_text())


if __name__ == "__main__":
    import sys
    vid = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("../data/exports") / vid / "raw"
    doc = load_body_doc(vid)
    info = write_raw_pose(doc, out)
    print(f"wrote raw pose for {vid}: {info}")
    # read-back sanity
    t = pq.read_table(out / "pose_joints.parquet")
    print(f"read-back: {t.num_rows} rows, cols={t.column_names}")
    print(f"provenance counts: ", end="")
    import collections
    c = collections.Counter(t.column('provenance').to_pylist())
    print(dict(c))
