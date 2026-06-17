"""LeRobot v2.1 dataset export — the lead format for VLA / robot-learning buyers.

Produces a Hugging Face LeRobot-v2.1-compatible dataset directory:

  <root>/
    meta/info.json            feature schema, fps, totals, path templates
    meta/episodes.jsonl       one line per episode {episode_index, tasks, length}
    meta/tasks.jsonl          {task_index, task}
    meta/episodes_stats.jsonl per-episode per-feature mean/std/min/max/count
                              (v2.1 replaced the global stats.json)
    data/chunk-000/episode_000000.parquet   per-frame feature rows
    videos/chunk-000/observation.images.ego/episode_000000.mp4

Current lerobot (>= 0.5) consumes format v3.0; it ships an official converter
for exactly this layout:
  python -m lerobot.scripts.convert_dataset_v21_to_v30 \
      --repo-id=<any/name> --root=<this dataset dir> --push-to-hub=false

Per-frame features:
  observation.state       float32[51]  body joint xyz (17 joints, documented order)
  observation.confidence  float32[17]  per-joint confidence (0..1)
  observation.images.ego  video        the anonymized egocentric clip
  action                  float32[8]   [l_wrist xyz, l_gripper, r_wrist xyz, r_gripper]
                                       end-effector targets + gripper commands.
                                       Gripper = grasp aperture_norm (1 = open,
                                       0 = closed), derived from the measured
                                       21-pt hand; holds its last observed value
                                       across detection dropouts (starts open).

Episodes (settings.lerobot_episode_mode):
  "segment" (default) — one episode per annotated activity segment; the VLM's
  natural-language description IS the episode's task string, so the dataset is
  directly consumable by language-conditioned VLA recipes (pi0/GR00T/OpenVLA).
  Episode videos are cut frame-accurately from the anonymized clip.
  "clip" — one episode per whole video with a single generic task label.

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

try:
    from config import settings
except Exception:  # pragma: no cover - allows import without app config
    settings = None

log = logging.getLogger("revisent.lerobot_export")

CODEBASE_VERSION = "v2.1"
CHUNK_SIZE = 1000
STATE_NAMES = [f"{j}.{a}" for j in JOINT_NAMES for a in ("x", "y", "z")]   # 51
_L_WRIST_IDX = JOINT_NAMES.index("l_wrist")
_R_WRIST_IDX = JOINT_NAMES.index("r_wrist")
# A wrist confidence at/above this means MEASURED (0.9) rather than the
# inferred fallback (0.225) — see body_pose confidence assignment.
_MEASURED_CONF = 0.5
ACTION_NAMES = ["l_wrist.x", "l_wrist.y", "l_wrist.z", "l_gripper",
                "r_wrist.x", "r_wrist.y", "r_wrist.z", "r_gripper"]


def _episode_rows(body_doc: dict, start_ms: Optional[float] = None,
                  end_ms: Optional[float] = None):
    """Yield per-frame feature dicts for one episode.

    With start_ms/end_ms the episode is the body frames inside [start, end);
    timestamps are re-based to the episode start. Gripper channels hold their
    last observed value across hand-detection dropouts (a policy needs a
    continuous command); open until first seen — frames before the window still
    update the held value so an episode starts with the true current hand state.
    """
    g_l = g_r = 1.0
    fi_out = 0
    base_ms = start_ms or 0.0
    for f in body_doc["frames"]:
        ts = float(f["timestamp_ms"])
        lg, rg = f.get("left_grasp"), f.get("right_grasp")
        if lg:
            g_l = float(lg["aperture_norm"])
        if rg:
            g_r = float(rg["aperture_norm"])
        if start_ms is not None and ts < start_ms:
            continue
        if end_ms is not None and ts >= end_ms:
            break
        joints = f["joints"]
        state = []
        conf = []
        for name in JOINT_NAMES:
            state.extend(joints[name]["pos"])
            conf.append(joints[name]["confidence"])
        action = joints["l_wrist"]["pos"] + [g_l] + joints["r_wrist"]["pos"] + [g_r]
        yield {
            "observation.state": [float(v) for v in state],
            "observation.confidence": [float(v) for v in conf],
            "action": [float(v) for v in action],
            "timestamp": (ts - base_ms) / 1000.0,
            "frame_index": fi_out,
        }
        fi_out += 1


def _load_segments(vid: str, root: Path) -> list:
    """Best-effort load of the clip's annotated activity segments."""
    p = root / "processed" / vid / "segments.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text()).get("segments") or []
    except Exception:  # noqa: BLE001
        return []


def _cut_video(src: Path, dst: Path, start: float, end: float) -> None:
    """Frame-accurate episode clip (re-encode; stream-copy cuts at keyframes)."""
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src),
         "-t", f"{max(0.05, end - start):.3f}", "-c:v", "libx264", "-crf", "20",
         "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-an", str(dst)],
        check=True,
    )


def _array_stats(arr: np.ndarray) -> dict:
    """v2.1-style per-feature stats for one episode's values (n, dim)."""
    n = arr.shape[0]
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return {
        "min": arr.min(axis=0).tolist(), "max": arr.max(axis=0).tolist(),
        "mean": mean.tolist(), "std": std.tolist(), "count": [int(n)],
    }


def _video_stats(video_path: Path, samples: int = 8) -> Optional[dict]:
    """Per-channel image stats (shape (3,1,1), values 0..1) from sampled frames —
    the v2.1 convention for video features in episodes_stats.jsonl."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    pix = []
    try:
        for idx in {int(round(i)) for i in np.linspace(0, max(0, total - 1), samples)}:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            pix.append(rgb.reshape(-1, 3))
    finally:
        cap.release()
    if not pix:
        return None
    a = np.concatenate(pix, axis=0)  # (N, 3)
    shape = lambda v: [[[float(x)]] for x in v]  # noqa: E731 - (3,1,1) nesting
    return {
        "min": shape(a.min(axis=0)), "max": shape(a.max(axis=0)),
        "mean": shape(a.mean(axis=0)), "std": shape(a.std(axis=0)),
        "count": [int(len(pix))],
    }


def build_lerobot_dataset(
    video_ids: List[str],
    out_root: Path,
    task_label: str = "egocentric manipulation demonstration",
    data_root: Optional[Path] = None,
    episode_mode: Optional[str] = None,
) -> dict:
    """Build a LeRobot v2.0 dataset from processed videos. Returns a summary.

    episode_mode "segment" (default via settings): one episode per annotated
    activity segment, the VLM description as the episode's task string — the
    shape language-conditioned VLA recipes consume. "clip": one episode per
    whole video with the generic task_label. Videos without segments fall back
    to clip mode.
    """
    mode = episode_mode or (getattr(settings, "lerobot_episode_mode", "segment")
                            if settings is not None else "segment")
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
    mount = "chest"
    episodes_meta = []
    episodes_stats = []
    tasks: dict[str, int] = {}
    height = width = None
    ep_idx = 0
    skipped = 0
    video_key = "observation.images.ego"

    for vid in video_ids:
        body_doc = load_body_doc(vid, data_root=root)
        mount = body_doc.get("camera_mount", mount)
        anon = root / "anonymized" / f"{vid}.mp4"

        # Episode plan: (start_s, end_s, task) — segments when available.
        segs = _load_segments(vid, root) if mode == "segment" else []
        if segs:
            plan = [
                (float(sg["start_time"]), float(sg["end_time"]),
                 (sg.get("description") or sg.get("task_label") or task_label).strip() or task_label)
                for sg in segs
            ]
        else:
            plan = [(None, None, task_label)]

        min_hand = (getattr(settings, "lerobot_min_hand_fraction", 0.05)
                    if settings is not None else 0.05)
        for (t0, t1, task) in plan:
            rows = list(_episode_rows(
                body_doc,
                start_ms=None if t0 is None else t0 * 1000.0,
                end_ms=None if t1 is None else t1 * 1000.0,
            ))
            if len(rows) < 2:
                skipped += 1
                log.warning("skip episode %s [%s-%s]: %d pose rows",
                            vid, t0, t1, len(rows))
                continue
            n = len(rows)
            # Manipulation datasets shouldn't ship hands-free episodes.
            hand_frac = sum(
                1 for r in rows
                if max(r["observation.confidence"][_L_WRIST_IDX],
                       r["observation.confidence"][_R_WRIST_IDX]) >= _MEASURED_CONF
            ) / n
            if min_hand and hand_frac < min_hand:
                skipped += 1
                log.warning("skip episode %s [%s-%s] %r: measured-hand fraction "
                            "%.0f%% < %.0f%%", vid, t0, t1, task[:40],
                            hand_frac * 100, min_hand * 100)
                continue

            dst = None
            if anon.exists():
                m = probe(anon)
                height, width = m.height, m.width
                dst = vid_dir / f"episode_{ep_idx:06d}.mp4"
                if t0 is None:
                    shutil.copy(anon, dst)
                else:
                    _cut_video(anon, dst, t0, t1)
            if fps_guess is None and n > 1:
                span = rows[-1]["timestamp"] - rows[0]["timestamp"]
                if span > 0:
                    fps_guess = round((n - 1) / span, 3)

            task_idx = tasks.setdefault(task, len(tasks))
            idx_base = total_frames
            table = pa.table({
                "observation.state": pa.array([r["observation.state"] for r in rows], pa.list_(pa.float32())),
                "observation.confidence": pa.array([r["observation.confidence"] for r in rows], pa.list_(pa.float32())),
                "action": pa.array([r["action"] for r in rows], pa.list_(pa.float32())),
                "timestamp": pa.array([r["timestamp"] for r in rows], pa.float32()),
                "frame_index": pa.array([r["frame_index"] for r in rows], pa.int64()),
                "episode_index": pa.array([ep_idx] * n, pa.int64()),
                "index": pa.array(list(range(idx_base, idx_base + n)), pa.int64()),
                "task_index": pa.array([task_idx] * n, pa.int64()),
            })
            pq.write_table(table, out_root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")

            # Per-episode stats for every parquet feature (the v2.1 contract),
            # plus per-channel image stats sampled from the episode video.
            ep_stats = {
                "observation.state": _array_stats(np.array([r["observation.state"] for r in rows], np.float64)),
                "observation.confidence": _array_stats(np.array([r["observation.confidence"] for r in rows], np.float64)),
                "action": _array_stats(np.array([r["action"] for r in rows], np.float64)),
                "timestamp": _array_stats(np.array([[r["timestamp"]] for r in rows], np.float64)),
                "frame_index": _array_stats(np.array([[r["frame_index"]] for r in rows], np.float64)),
                "episode_index": _array_stats(np.full((n, 1), ep_idx, np.float64)),
                "index": _array_stats(np.arange(idx_base, idx_base + n, dtype=np.float64).reshape(-1, 1)),
                "task_index": _array_stats(np.full((n, 1), task_idx, np.float64)),
            }
            if dst is not None:
                vstats = _video_stats(dst)
                if vstats:
                    ep_stats[video_key] = vstats
            episodes_stats.append({"episode_index": ep_idx, "stats": ep_stats})

            episodes_meta.append({"episode_index": ep_idx, "tasks": [task], "length": n})
            total_frames += n
            ep_idx += 1

    if not tasks:
        tasks = {task_label: 0}
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
        "robot_type": f"human_egocentric_{mount}_cam",
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
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
        for task, idx in sorted(tasks.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    # v2.1: per-episode stats replace the global stats.json.
    with (out_root / "meta" / "episodes_stats.jsonl").open("w") as f:
        for e in episodes_stats:
            f.write(json.dumps(e) + "\n")

    log.info("LeRobot dataset (%s mode): %d episodes, %d tasks, %d frames (%d skipped) -> %s",
             mode, len(episodes_meta), len(tasks), total_frames, skipped, out_root)
    return {"episodes": len(episodes_meta), "tasks": len(tasks), "frames": total_frames,
            "fps": fps, "episode_mode": mode, "skipped_episodes": skipped,
            "root": str(out_root)}


def validate_lerobot(out_root: Path) -> list[str]:
    """Structural validation; returns a list of problems (empty == OK)."""
    out_root = Path(out_root)
    problems = []
    info_p = out_root / "meta" / "info.json"
    if not info_p.exists():
        return ["missing meta/info.json"]
    info = json.loads(info_p.read_text())
    for rel in ("meta/episodes.jsonl", "meta/tasks.jsonl", "meta/episodes_stats.jsonl"):
        if not (out_root / rel).exists():
            problems.append(f"missing {rel}")
    es_path = out_root / "meta" / "episodes_stats.jsonl"
    if es_path.exists():
        n_stats = len(es_path.read_text().splitlines())
        if n_stats != info.get("total_episodes"):
            problems.append(f"episodes_stats.jsonl has {n_stats} lines != "
                            f"total_episodes {info.get('total_episodes')}")
    # episode/task counts must match the metadata files
    ep_lines = (out_root / "meta" / "episodes.jsonl")
    if ep_lines.exists():
        n_eps = len(ep_lines.read_text().splitlines())
        if n_eps != info.get("total_episodes"):
            problems.append(f"episodes.jsonl has {n_eps} lines != total_episodes {info.get('total_episodes')}")
    task_lines = (out_root / "meta" / "tasks.jsonl")
    if task_lines.exists():
        n_tasks = len(task_lines.read_text().splitlines())
        if n_tasks != info.get("total_tasks"):
            problems.append(f"tasks.jsonl has {n_tasks} lines != total_tasks {info.get('total_tasks')}")
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
