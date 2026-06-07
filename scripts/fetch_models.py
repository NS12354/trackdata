#!/usr/bin/env python3
"""Download ML model weights into backend/ml_models/.

Run once after install (and in the Docker build):
    python scripts/fetch_models.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1] / "backend" / "ml_models"

MODELS = {
    # YuNet face detector (OpenCV Zoo, 2023mar). Small (~227KB), CPU-fast, robust.
    "face_detection_yunet_2023mar.onnx": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/"
               "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": None,  # set to pin integrity if desired
    },
    # EAST scene-text detector (~92MB) for text/PII blurring (unit numbers, mail,
    # plates, screens, whiteboards).
    "frozen_east_text_detection.pb": {
        "url": "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/"
               "frozen_east_text_detection.pb",
        "sha256": None,
    },
}


def main() -> int:
    ML_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in MODELS.items():
        dest = ML_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"✓ {name} already present ({dest.stat().st_size} bytes)")
            continue
        print(f"↓ downloading {name} ...")
        urllib.request.urlretrieve(spec["url"], dest)
        if spec.get("sha256"):
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            if digest != spec["sha256"]:
                dest.unlink()
                print(f"✗ checksum mismatch for {name}: {digest}")
                return 1
        print(f"✓ {name} ({dest.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
