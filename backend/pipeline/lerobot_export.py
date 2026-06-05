"""LeRobot v2 dataset export — the lead format for VLA / robot-learning buyers.

Produces a Hugging Face LeRobot-v2.0-compatible dataset directory:

  <root>/
    meta/info.json          feature schema, fps, totals, path templates
    meta/episodes.jsonl     one line per episode {episode_index, tasks, length}
    meta/tasks.jsonl        {task_index, task}
    meta/stats.json         per-feature mean/std/min/max/count
    data/chunk-000/episode_000000.parquet   per-frame feature rows
    videos/chunk-000/observation.images.ego/episode_000000.mp4

Per-frame features:
  observation.state       float32[51]  body joint xyz (17 joints, documented order)
  observation.confidence  float32[17]  per-joint confidence (0..1)
  observation.images.ego  video        the anonymized egocentric clip
  action                  float32[6]   [l_wrist xyz, r_wrist xyz] end-effector targets

Buyers care most about the measured hands/wrists; `observation.confidence` lets
them mask the inferred joints. Hands' 21-pt detail is in the companion raw export.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .body_pose import JOINT_NAMES
from .pose_export import load_body_doc
from .video_meta import probe

log = logging.getLogger("revisent.lerobot_export")

CODEBASE_VERSION = "v2.0"
CHUNK_SIZE = 1000
STATE_NAMES = [f"{j}.{a}" for j in JOINT_NAMES for a in ("x", "y", "z")]   # 51
ACTION_NAMES = ["l_wrist.x", "l_wrist.y", "l_wrist.z", "r_wrist.x", "r_wrist.y", "r_wrist.z"]


def _episode_rows(body_doc: dict):
    """Yield per-frame feature dicts for one clip."""
    for fi, f in enumerate(body_doc["frames"]):
        joints = f["joints"]
        state = []
        conf = []
        for name in JOINT_NAMES:
            state.extend(joints[name]["pos"])
            conf.append(joints[name]["confidence"])
        action = joints["l_wrist"]["pos"] + joints["r_wrist"]["pos"]
        yield {
            "observation.state": [float(v) for v in state],
            "observation.confidence": [float(v) for v in conf],
            "action": [float(v) for v in action],
            "timestamp": float(f["timestamp_ms"]) / 1000.0,
            "frame_index": fi,
        }


def _running_stats():
    return {"min": None, "max": None, "sum": None, "sqsum": None, "count": 0}


def _update_stats(st, arr: np.ndarray):
    st["count"] += arr.shape[0]
    s = arr.sum(axis=0); sq = (arr * arr).sum(axis=0)
    mn = arr.min(axis=0); mx = arr.max(axis=0)
    st["sum"] = s if st["sum"] is None else st["sum"] + s
    st["sqsum"] = sq if st["sqsum"] is None else st["sqsum"] + sq
    st["min"] = mn if st["min"] is None else np.minimum(st["min"], mn)
    st["max"] = mx if st["max"] is None else np.maximum(st["max"], mx)


def _finalize_stats(st):
    n = max(1, st["count"])
    mean = st["sum"] / n
    var = np.maximum(0.0, st["sqsum"] / n - mean * mean)
    return {
        "mean": mean.tolist(), "std": np.sqrt(var).tolist(),
        "min": st["min"].tolist(), "max": st["max"].tolist(), "count": [st["count"]],
    }


def build_lerobot_dataset(
    video_ids: List[str],
    out_root: Path,
    task_label: str = "egocentric manipulation demonstration",
    data_root: Optional[Path] = None,
) -> dict:
    """Build a LeRobot v2.0 dataset from processed videos. Returns a summary."""
    out_root = Path(out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "meta").mkdir(parents=True)
    (out_root / "data" / "chunk-000").mkdir(parents=True)
    vid_dir = out_root / "videos" / "chunk-000" / "observation.images.ego"
    vid_dir.mkdir(parents=True)

    root = Path(data_root) if data_root else Path(__file__).resolve().parents[1].parent / "data"
    fps_guess = None
    total_frames = 0
    episodes_meta = []
    stats = {k: _running_stats() for k in ("observation.state", "observation.confidence", "action")}
    ts_stats, fidx_stats = _running_stats(), _running_stats()
    height = width = None

    for ep_idx, vid in enumerate(video_ids):
        body_doc = load_body_doc(vid, data_root=root)
        rows = list(_episode_rows(body_doc))
        if not rows:
            log.warning("skip %s: no frames", vid); continue
        n = len(rows)

        # video + dims/fps
        anon = root / "anonymized" / f"{vid}.mp4"
        if anon.exists():
            m = probe(anon)
            height, width = m.height, m.width
            if fps_guess is None and n > 1:
                span = rows[-1]["timestamp"] - rows[0]["timestamp"]
                fps_guess = round((n - 1) / span, 3) if span > 0 else (m.fps or 10.0)
            shutil.copy(anon, vid_dir / f"episode_{ep_idx:06d}.mp4")

        idx_base = total_frames
        table = pa.table({
            "observation.state": pa.array([r["observation.state"] for r in rows], pa.list_(pa.float32())),
            "observation.confidence": pa.array([r["observation.confidence"] for r in rows], pa.list_(pa.float32())),
            "action": pa.array([r["action"] for r in rows], pa.list_(pa.float32())),
            "timestamp": pa.array([r["timestamp"] for r in rows], pa.float32()),
            "frame_index": pa.array([r["frame_index"] for r in rows], pa.int64()),
            "episode_index": pa.array([ep_idx] * n, pa.int64()),
            "index": pa.array(list(range(idx_base, idx_base + n)), pa.int64()),
            "task_index": pa.array([0] * n, pa.int64()),
        })
        pq.write_table(table, out_root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")

        _update_stats(stats["observation.state"], np.array([r["observation.state"] for r in rows], np.float64))
        _update_stats(stats["observation.confidence"], np.array([r["observation.confidence"] for r in rows], np.float64))
        _update_stats(stats["action"], np.array([r["action"] for r in rows], np.float64))
        _update_stats(ts_stats, np.array([[r["timestamp"]] for r in rows], np.float64))
        _update_stats(fidx_stats, np.array([[r["frame_index"]] for r in rows], np.float64))

        episodes_meta.append({"episode_index": ep_idx, "tasks": [task_label], "length": n})
        total_frames += n

    fps = fps_guess or 10.0
    features = {
        "observation.state": {"dtype": "float32", "shape": [len(STATE_NAMES)], "names": STATE_NAMES},
        "observation.confidence": {"dtype": "float32", "shape": [len(JOINT_NAMES)], "names": list(JOINT_NAMES)},
        "action": {"dtype": "float32", "shape": [len(ACTION_NAMES)], "names": ACTION_NAMES},
        "observation.images.ego": {
            "dtype": "video", "shape": [height or 1920, width or 1080, 3],
            "names": ["height", "width", "channel"],
            "info": {"video.fps": fps, "video.codec": "h264", "video.is_depth_map": False},
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "human_egocentric_chest_cam",
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episodes_meta),
        "total_chunks": 1,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes_meta)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (out_root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    with (out_root / "meta" / "episodes.jsonl").open("w") as f:
        for e in episodes_meta:
            f.write(json.dumps(e) + "\n")
    with (out_root / "meta" / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_label}) + "\n")

    stats_out = {
        "observation.state": _finalize_stats(stats["observation.state"]),
        "observation.confidence": _finalize_stats(stats["observation.confidence"]),
        "action": _finalize_stats(stats["action"]),
        "timestamp": _finalize_stats(ts_stats),
        "frame_index": _finalize_stats(fidx_stats),
    }
    (out_root / "meta" / "stats.json").write_text(json.dumps(stats_out, indent=2))

    log.info("LeRobot dataset: %d episodes, %d frames -> %s", len(episodes_meta), total_frames, out_root)
    return {"episodes": len(episodes_meta), "frames": total_frames, "fps": fps, "root": str(out_root)}


def validate_lerobot(out_root: Path) -> list[str]:
    """Structural validation; returns a list of problems (empty == OK)."""
    out_root = Path(out_root)
    problems = []
    info_p = out_root / "meta" / "info.json"
    if not info_p.exists():
        return ["missing meta/info.json"]
    info = json.loads(info_p.read_text())
    for rel in ("meta/episodes.jsonl", "meta/tasks.jsonl", "meta/stats.json"):
        if not (out_root / rel).exists():
            problems.append(f"missing {rel}")
    # episode parquet columns must cover all non-video features
    non_video = [k for k, v in info["features"].items() if v["dtype"] != "video"]
    ep0 = out_root / "data" / "chunk-000" / "episode_000000.parquet"
    if ep0.exists():
        cols = set(pq.read_schema(ep0).names)
        missing = [c for c in non_video if c not in cols]
        if missing:
            problems.append(f"episode parquet missing feature cols: {missing}")
        # frame count consistency
        n_rows = pq.read_metadata(ep0).num_rows
        ep_meta = [json.loads(l) for l in (out_root / "meta" / "episodes.jsonl").read_text().splitlines()]
        if ep_meta and ep_meta[0]["length"] != n_rows:
            problems.append(f"episode 0 length {ep_meta[0]['length']} != parquet rows {n_rows}")
    else:
        problems.append("missing data/chunk-000/episode_000000.parquet")
    # optional: real library load
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
        LeRobotDataset(repo_id="local", root=str(out_root))
        log.info("lerobot library loaded the dataset OK")
    except ImportError:
        problems.append("INFO: lerobot not installed — structural check only "
                        "(pip install lerobot to do a real load test)")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"lerobot load error: {exc}")
    return problems


if __name__ == "__main__":
    import sys
    vids = sys.argv[1:] or ["91b674ab-ab61-426d-b3d8-631bc84e10fe"]
    out = Path("../data/exports/_lerobot_demo")
    summary = build_lerobot_dataset(vids, out)
    print("built:", summary)
    print("validation:")
    for p in validate_lerobot(out) or ["  OK — no structural problems"]:
        print("  -", p)
