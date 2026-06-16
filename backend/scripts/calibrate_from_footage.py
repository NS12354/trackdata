#!/usr/bin/env python3
"""No-checkerboard calibration — derive a camera profile from ORDINARY footage.

Uses Structure-from-Motion (COLMAP) to self-calibrate: it matches features across
frames of a normal clip and bundle-adjusts the lens intrinsics + distortion. No
checkerboard, no special capture — just a clip with some camera movement and
texture (egocentric walking footage is ideal). Run once per camera model; writes
the same profile schema as scripts/calibrate_camera.py.

Requires COLMAP on PATH (one-time):  brew install colmap   (macOS)

Usage (from backend/):
  python scripts/calibrate_from_footage.py \\
      --input ../data/calibration/transcend_walkaround.mp4 \\
      --camera-model transcend_dpb30 \\
      --fps 2

Tips for a good clip: 30-90s, move/turn so the scene shifts between frames (not a
static stare), lots of texture and straight architectural edges, in focus. COLMAP
needs parallax — a clip where the camera barely moves will fail to reconstruct.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROFILES_DIR = BACKEND_DIR / "pipeline" / "calibration_profiles"

# COLMAP camera models we accept -> (our lens_model, how to read PARAMS).
# PARAMS order per COLMAP docs.
COLMAP_MODELS = {
    "SIMPLE_RADIAL": "standard",   # f, cx, cy, k
    "RADIAL": "standard",          # f, cx, cy, k1, k2
    "OPENCV": "standard",          # fx, fy, cx, cy, k1, k2, p1, p2
    "FULL_OPENCV": "standard",     # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
    "OPENCV_FISHEYE": "fisheye",   # fx, fy, cx, cy, k1, k2, k3, k4
}


def _require_colmap() -> str:
    exe = shutil.which("colmap")
    if not exe:
        sys.exit(
            "COLMAP not found on PATH. Install it once:\n"
            "  macOS:  brew install colmap\n"
            "  Linux:  apt-get install colmap  (or build from source)\n"
            "Then re-run. (Or use the checkerboard tool: scripts/calibrate_camera.py)"
        )
    return exe


def extract_frames(video: Path, out_dir: Path, fps: float, max_frames: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found — needed to sample frames.")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", f"fps={fps}", "-frames:v", str(max_frames),
         "-q:v", "2", str(out_dir / "frame_%05d.jpg")],
        check=True,
    )
    n = len(list(out_dir.glob("*.jpg")))
    if n < 12:
        sys.exit(f"only {n} frames extracted — need more. Use a longer clip or higher --fps.")
    return n


def run_colmap(colmap: str, work: Path, images: Path, camera_model_type: str, use_gpu: bool) -> Path:
    db = work / "database.db"
    sparse = work / "sparse"
    sparse.mkdir(exist_ok=True)
    gpu = "1" if use_gpu else "0"
    common_log = {"check": True}

    print("[colmap] feature extraction…")
    subprocess.run(
        [colmap, "feature_extractor", "--database_path", str(db),
         "--image_path", str(images),
         "--ImageReader.single_camera", "1",
         "--ImageReader.camera_model", camera_model_type,
         "--SiftExtraction.use_gpu", gpu],
        **common_log,
    )
    print("[colmap] sequential matching…")
    subprocess.run(
        [colmap, "sequential_matcher", "--database_path", str(db),
         "--SiftMatching.use_gpu", gpu],
        **common_log,
    )
    print("[colmap] sparse reconstruction (this is the slow part)…")
    subprocess.run(
        [colmap, "mapper", "--database_path", str(db),
         "--image_path", str(images), "--output_path", str(sparse)],
        **common_log,
    )
    models = sorted(p for p in sparse.glob("*") if p.is_dir())
    if not models:
        sys.exit("COLMAP could not reconstruct the scene — the clip likely lacks "
                 "movement/parallax or texture. Use a clip where the camera moves "
                 "through the space, or fall back to the checkerboard tool.")
    model_dir = models[0]
    # Convert the largest model to TXT so we can parse cameras.txt.
    subprocess.run(
        [colmap, "model_converter", "--input_path", str(model_dir),
         "--output_path", str(model_dir), "--output_type", "TXT"],
        **common_log,
    )
    return model_dir


def parse_cameras_txt(path: Path) -> dict:
    """Parse COLMAP cameras.txt -> {model, width, height, params:[...]}."""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
        parts = line.split()
        return {
            "model": parts[1],
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": [float(x) for x in parts[4:]],
        }
    raise ValueError(f"no camera found in {path}")


def mean_reproj_error(model_dir: Path, colmap: str) -> float | None:
    """Ask COLMAP for the reconstruction's mean reprojection error."""
    try:
        out = subprocess.run(
            [colmap, "model_analyzer", "--path", str(model_dir)],
            capture_output=True, text=True, check=True,
        ).stderr + subprocess.run(
            [colmap, "model_analyzer", "--path", str(model_dir)],
            capture_output=True, text=True, check=True,
        ).stdout
        for line in out.splitlines():
            if "reprojection error" in line.lower():
                # e.g. "Mean reprojection error: 0.73px"
                tok = line.split(":")[-1].strip().rstrip("px").strip()
                return round(float(tok), 4)
    except Exception:  # noqa: BLE001
        pass
    return None


def colmap_to_profile(cam: dict, camera_model: str, reproj: float | None) -> dict:
    """Convert a parsed COLMAP camera into our calibration-profile schema."""
    model = cam["model"]
    if model not in COLMAP_MODELS:
        sys.exit(f"unsupported COLMAP model {model}; re-run with --camera-model-type OPENCV")
    lens = COLMAP_MODELS[model]
    p = cam["params"]
    W, H = cam["width"], cam["height"]

    if model == "SIMPLE_RADIAL":
        f, cx, cy, k = p
        fx = fy = f
        D = [k, 0.0, 0.0, 0.0, 0.0]
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = p
        fx = fy = f
        D = [k1, k2, 0.0, 0.0, 0.0]
    elif model in ("OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        k1, k2, p1, p2 = p[4], p[5], p[6], p[7]
        k3 = p[8] if len(p) > 8 else 0.0
        D = [k1, k2, p1, p2, k3]
    elif model == "OPENCV_FISHEYE":
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        D = [p[4], p[5], p[6], p[7]]  # fisheye: k1..k4
    else:  # pragma: no cover
        sys.exit(f"unhandled model {model}")

    return {
        "camera_model": camera_model,
        "calibration_date": date.today().isoformat(),
        "lens_model": lens,
        "image_size": [W, H],
        "intrinsic_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        "distortion_coefficients": [round(c, 8) for c in D],
        "reprojection_error_pixels": reproj if reproj is not None else -1.0,
        "num_calibration_frames": 0,
        "pattern_size": [0, 0],
        "square_size_mm": 0,
        "notes": f"Self-calibrated from footage via COLMAP ({model}). No checkerboard.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="No-checkerboard calibration from ordinary footage (COLMAP SfM)")
    ap.add_argument("--input", type=Path, required=True, help="ordinary clip from the camera")
    ap.add_argument("--camera-model", required=True, help="profile key, e.g. transcend_dpb30")
    ap.add_argument("--fps", type=float, default=2.0, help="frames/sec to sample (default 2)")
    ap.add_argument("--max-frames", type=int, default=150, help="cap sampled frames (default 150)")
    ap.add_argument("--camera-model-type", default="OPENCV",
                    choices=list(COLMAP_MODELS.keys()), help="COLMAP camera model (default OPENCV)")
    ap.add_argument("--use-gpu", action="store_true", help="use GPU SIFT (faster; needs CUDA/Metal build)")
    ap.add_argument("--keep", action="store_true", help="keep the COLMAP working dir")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"input not found: {args.input}")
    colmap = _require_colmap()

    work = Path(tempfile.mkdtemp(prefix=f"colmap_{args.camera_model}_"))
    try:
        images = work / "images"
        n = extract_frames(args.input, images, args.fps, args.max_frames)
        print(f"extracted {n} frames -> {images}")
        model_dir = run_colmap(colmap, work, images, args.camera_model_type, args.use_gpu)
        cam = parse_cameras_txt(model_dir / "cameras.txt")
        reproj = mean_reproj_error(model_dir, colmap)
        profile = colmap_to_profile(cam, args.camera_model, reproj)
        profile["num_calibration_frames"] = n

        out = PROFILES_DIR / f"{args.camera_model}.json"
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(profile, indent=2))

        print("\n" + "=" * 60)
        print(f"✓ wrote {out}")
        print(f"  lens model  : {profile['lens_model']} (COLMAP {cam['model']})")
        print(f"  reproj error: {profile['reprojection_error_pixels']}px")
        print(f"  frames used : {n}")
        if reproj is not None and reproj > 1.5:
            print("  ⚠ reproj error is high — the reconstruction may be weak. Try a clip "
                  "with more movement/texture, or use the checkerboard tool for precision.")
        print("  The pipeline will undistort videos tagged with this camera model.")
    finally:
        if args.keep:
            print(f"(kept working dir: {work})")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
