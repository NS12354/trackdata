"""Process local video files end-to-end WITHOUT the API/DB — the shakedown tool.

Runs the full pipeline on each input clip and prints an honest per-stage report:

  anonymize -> hand pose -> head VO -> body pose (+grasp) -> segmentation
  -> LeRobot v2 export across all clips

Outputs land in the standard data/ tree (anonymized/, processed/<vid>/), so the
results are also loadable by the exporters and (after a DB upload) the dashboard.

Usage (from repo root):
  backend/.venv/Scripts/python scripts/process_clip.py data/demo/IMG_0077.MOV \
      data/demo/IMG_0078.MOV --mount head --height-cm 175
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from config import settings  # noqa: E402


def _vid_slug(path: Path) -> str:
    return re.sub(r"[^a-z0-9]+", "", path.stem.lower()) or "clip"


def _stage(name: str):
    print(f"  [{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
    return time.time()


def _done(t0: float, msg: str):
    print(f"      {msg}  ({time.time() - t0:.1f}s)", flush=True)


def process(src: Path, mount: str, height_cm: float, run_vlm: bool, data_root: Path,
            no_blur: bool = False, rotate: int | None = None,
            hand_conf: float | None = None, gate: bool = True) -> dict:
    import subprocess

    from pipeline.anonymize import anonymize_video
    from pipeline.hand_pose import extract_hand_pose, load_hand_pose
    from pipeline.ego_pose import estimate_head_trajectory, trajectory_to_json
    from pipeline.body_pose import estimate_body_pose, body_pose_to_json
    from pipeline.video_meta import probe

    vid = _vid_slug(src)
    anon = data_root / "anonymized" / f"{vid}.mp4"
    proc = data_root / "processed" / vid
    proc.mkdir(parents=True, exist_ok=True)
    report: dict = {"video_id": vid, "source": str(src), "camera_mount": mount}
    print(f"\n=== {src.name} -> {vid} ===", flush=True)

    # Camera intrinsics: calibration > device preset > configured default.
    # Set globally so VO and body-pose back-projection share the same FOV.
    from pipeline.intrinsics import device_model, fov_for_video
    fov, fov_source = fov_for_video(src)
    settings.ego_camera_fov_deg = fov
    report["camera"] = {"device": device_model(src), "fov_deg": fov,
                        "fov_source": fov_source}
    (proc / "capture.json").write_text(json.dumps(report["camera"], indent=2))
    print(f"  camera: {report['camera']['device'] or 'unknown'} -> "
          f"FOV {fov:.0f} deg ({fov_source})", flush=True)

    # Phase 0: intake quality gate — orientation (by HAND evidence; ego footage
    # has no upright faces), brightness, blur. Catches upside-down mounts before
    # they silently corrupt every downstream coordinate.
    if gate:
        from pipeline.quality_gate import probe_clip
        t = _stage("quality gate (orientation / exposure / hands)")
        q = probe_clip(src)
        report["quality_gate"] = q.as_dict()
        for w in q.warnings:
            print(f"      WARNING: {w}")
        if rotate is None and q.rotation_decided and q.suggested_rotation:
            if no_blur:
                rotate = q.suggested_rotation
                print(f"      auto-correcting orientation: --rotate {rotate}")
            else:
                print(f"      NOTE: clip needs rotate {q.suggested_rotation} deg; the blur "
                      "path can't apply extra rotation — rerun with --no-blur or fix the source")
        _done(t, f"VLM votes {q.vlm_votes} -> rotate {q.suggested_rotation} "
                 f"(decided={q.rotation_decided}); hand evidence (advisory) "
                 f"{q.hand_scores_by_rotation}")
    rotate = rotate or 0

    if no_blur:
        # Skip face blurring (footage with no bystanders / consented demo only).
        # Still normalizes like the anonymizer: bake orientation into pixels,
        # H.264, drop audio. Rotation is EXPLICIT (-noautorotate + transpose from
        # our probe + the extra --rotate) so output orientation never depends on
        # ffmpeg's display-matrix interpretation.
        total = (probe(src).rotation + rotate) % 360
        t = _stage(f"transcode (NO face blur, total rotate {total})")
        vf = {0: None, 90: "transpose=1", 180: "hflip,vflip", 270: "transpose=2"}[total]
        cmd = ["ffmpeg", "-y", "-v", "error", "-noautorotate", "-i", str(src)]
        if vf:
            cmd += ["-vf", vf]
        cmd += ["-metadata:s:v", "rotate=0",
                "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(anon)]
        anon.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)
        report["anonymize"] = {"skipped": True, "note": "blur disabled",
                               "rotate_extra": rotate, "rotate_total": total}
        _done(t, f"orientation-corrected copy written (blur OFF, total rotate {total} deg)")
    else:
        if rotate:
            print("      note: --rotate is only applied in --no-blur mode; ignoring")
        t = _stage("anonymize (face blur)")
        ar = anonymize_video(src, anon)
        report["anonymize"] = {
            "duration_s": ar.duration_seconds, "frames": ar.frames_total,
            "coverage": ar.coverage, "tracks": ar.tracks, "rejected_tracks": ar.rejected_tracks,
        }
        _done(t, f"{ar.frames_total} frames, blur coverage {ar.coverage:.1%}, "
                 f"{ar.tracks} face track(s), {ar.rejected_tracks} rejected")

    if hand_conf is not None:
        # Stricter than the recall-tuned 0.1 default: clean well-lit footage
        # trades phantom hands away for a few missed blurry ones.
        settings.hand_pose_min_detection_confidence = hand_conf
        settings.hand_pose_min_tracking_confidence = hand_conf
        report["hand_confidence_override"] = hand_conf

    backend = getattr(settings, "hand_pose_backend", "mediapipe")
    t = _stage(f"hand pose ({backend} 21-pt)")
    if backend == "wilor":
        from pipeline.hand_pose import extract_hand_pose_wilor
        hr = extract_hand_pose_wilor(anon, proc / "hand_pose.parquet", vid)
    else:
        hr = extract_hand_pose(anon, proc / "hand_pose.parquet", vid)
    report["hand_pose"] = {
        "coverage": hr.coverage, "left_frames": hr.left_hand_frames,
        "right_frames": hr.right_hand_frames, "sampled": hr.frames_sampled,
    }
    _done(t, f"hands in {hr.coverage:.1%} of {hr.frames_sampled} sampled frames "
             f"(L {hr.left_hand_frames}, R {hr.right_hand_frames})")

    t = _stage("head pose (monocular VO)")
    poses = estimate_head_trajectory(anon)
    (proc / "head_pose.json").write_text(json.dumps(trajectory_to_json(vid, poses)))
    tracked = sum(1 for p in poses if p.tracked)
    report["head_vo"] = {"poses": len(poses), "tracked_fraction": tracked / max(1, len(poses))}
    _done(t, f"{tracked}/{len(poses)} poses tracked ({tracked / max(1, len(poses)):.0%})")

    t = _stage(f"body pose + grasp (mount={mount})")
    m = probe(anon)
    hand_rows = load_hand_pose(proc / "hand_pose.parquet")
    head_frames = json.loads((proc / "head_pose.json").read_text())["frames"]
    frames = estimate_body_pose(head_frames, hand_rows, height_cm=height_cm,
                                image_wh=(m.width, m.height))
    doc = body_pose_to_json(vid, frames, height_cm)
    (proc / "body_pose.json").write_text(json.dumps(doc))
    closed = sum(1 for f in frames
                 for g in (f.left_grasp, f.right_grasp) if g and g["closed"])
    report["body_pose"] = {**doc["coverage"], "frames": doc["frame_count"],
                           "closed_grasp_samples": closed}
    _done(t, f"{doc['frame_count']} frames, wrist measured {doc['coverage']['wrist_measured_fraction']:.1%}, "
             f"grasp observed {doc['coverage']['grasp_observed_fraction']:.1%}, "
             f"{closed} closed-grasp samples")

    if run_vlm:
        # Human corrections are authoritative: never silently overwrite a
        # timeline a person has verified. Re-annotation requires deleting the
        # existing segments.json deliberately.
        seg_path = proc / "segments.json"
        if seg_path.exists():
            try:
                existing = json.loads(seg_path.read_text())
                if any(s.get("human_verified") for s in existing.get("segments", [])):
                    print("      SKIP: segments contain human-verified labels; "
                          "delete segments.json to force re-annotation")
                    report["segments"] = existing.get("segments", [])
                    run_vlm = False
            except Exception:
                pass
    if run_vlm:
        t = _stage(f"segmentation ({settings.segmentation_boundary_mode} + "
                   f"{settings.segmentation_provider}/{settings.ollama_vlm_model})")
        try:
            from pipeline.segmentation import segment_video
            sr = segment_video(anon, vid, hand_rows=hand_rows, head_frames=head_frames)
            (proc / "segments.json").write_text(json.dumps(sr.to_json(), indent=2))
            report["segments"] = [
                {"start": s.start_time, "end": s.end_time, "label": s.task_label,
                 "confidence": s.confidence, "description": s.description}
                for s in sr.segments
            ]
            _done(t, f"{len(sr.segments)} segment(s), {sr.frames_classified} frame(s) classified")
            for s in sr.segments:
                desc = (s.description or "")[:90]
                print(f"      {s.start_time:7.1f}-{s.end_time:7.1f}s  {s.task_label}  | {desc}")
        except Exception as exc:  # noqa: BLE001 - VLM down: keep cuts, skip labels
            print(f"      VLM labeling failed ({exc}); falling back to boundary-only")
            from pipeline.boundary import detect_boundaries
            bres = detect_boundaries(anon, m, hand_rows=hand_rows, head_frames=head_frames)
            report["boundaries"] = bres.to_meta()
            (proc / "segments.json").write_text(json.dumps(
                {"video_id": vid, "segments": [
                    {"start_time": a, "end_time": b, "task_label": "unlabeled",
                     "confidence": 0.0, "description": ""} for a, b in bres.segments],
                 "boundary_meta": bres.to_meta()}, indent=2))
            _done(t, f"{len(bres.segments)} segment(s) from signals {bres.signals_used}")
    return report


def register_clip(report: dict, src: Path, data_root: Path, height_cm: float,
                  location: str, wearer_id: str) -> None:
    """Upsert the clip's DB row (and derive events) so it appears on the
    dashboard — no API server or manual SQL needed."""
    from datetime import datetime, timezone

    from db import init_db, session_scope
    from models import Video, VideoStatus
    from pipeline.video_meta import probe

    init_db()
    vid = report["video_id"]
    anon = data_root / "anonymized" / f"{vid}.mp4"
    m = probe(anon)
    now = datetime.now(timezone.utc)
    has_segments = bool(report.get("segments"))
    blur_skipped = bool(report.get("anonymize", {}).get("skipped"))
    with session_scope() as s:
        v = s.get(Video, vid)
        if v is None:
            v = Video(id=vid, original_filename=src.name)
            s.add(v)
        v.status = VideoStatus.processed if has_segments else VideoStatus.anonymized
        v.operator_id = "local-collector"
        v.worker_id_anonymized = wearer_id
        v.property_tag = location
        v.operator_height_cm = int(height_cm)
        v.file_size = src.stat().st_size
        v.duration_seconds = m.duration_seconds
        v.anonymized_at = now
        v.anonymization_coverage = report.get("anonymize", {}).get("coverage")
        v.anonymization_method = "none (blur disabled - no bystanders)" if blur_skipped else "union"
        v.hand_pose_extracted = True
        v.hand_pose_extracted_at = now
        v.segmented = has_segments
        v.segmented_at = now if has_segments else None
        v.segmentation_cost_usd = 0.0
        v.error_message = None
    if has_segments:
        from pipeline.events import extract_events
        summary = extract_events(vid)
        print(f"  registered {vid}: {summary['event_count']} events -> dashboard")
    else:
        print(f"  registered {vid} (no segments yet) -> dashboard")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--mount", choices=["chest", "head"], default=None,
                    help="camera mount (default: settings.camera_mount)")
    ap.add_argument("--height-cm", type=float, default=175.0)
    ap.add_argument("--no-vlm", action="store_true", help="skip VLM segment labeling")
    ap.add_argument("--no-export", action="store_true", help="skip LeRobot export")
    ap.add_argument("--no-blur", action="store_true",
                    help="skip face blurring (no-bystander footage only)")
    ap.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=None,
                    help="extra rotation on top of metadata (no-blur mode). Omit to "
                         "let the quality gate auto-detect from hand evidence")
    ap.add_argument("--hand-conf", type=float, default=None,
                    help="override hand detection/tracking confidence (e.g. 0.5)")
    ap.add_argument("--no-gate", action="store_true",
                    help="skip the intake quality gate (orientation/exposure checks)")
    ap.add_argument("--register", action="store_true",
                    help="upsert the clip into the dashboard DB and derive events")
    ap.add_argument("--location", default="demo", help="location/scene tag for --register")
    ap.add_argument("--wearer-id", default="demo-wearer", help="anonymized wearer id for --register")
    args = ap.parse_args()

    if args.mount:
        settings.camera_mount = args.mount
    data_root = settings.data_dir

    reports = []
    for src in args.videos:
        if not src.exists():
            print(f"missing: {src}"); continue
        rep = process(src.resolve(), settings.camera_mount,
                      args.height_cm, not args.no_vlm, data_root,
                      no_blur=args.no_blur, rotate=args.rotate,
                      hand_conf=args.hand_conf, gate=not args.no_gate)
        if args.register:
            register_clip(rep, src.resolve(), data_root, args.height_cm,
                          args.location, args.wearer_id)
        reports.append(rep)

    if reports and not args.no_export:
        from pipeline.lerobot_export import build_lerobot_dataset, validate_lerobot
        out = data_root / "exports" / "lerobot_demo"
        print(f"\n=== LeRobot export ({len(reports)} episode(s)) ===", flush=True)
        summary = build_lerobot_dataset([r["video_id"] for r in reports], out,
                                        data_root=data_root)
        problems = [p for p in validate_lerobot(out) if not p.startswith("INFO:")]
        print(f"  built: {summary}")
        print(f"  validation: {problems or 'clean'}")

    rep_path = data_root / "exports" / "shakedown_report.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(reports, indent=2))
    print(f"\nreport: {rep_path}")


if __name__ == "__main__":
    main()
