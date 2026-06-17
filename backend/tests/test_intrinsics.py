"""Device-preset FOV tests (pure mapping; the ffprobe path is exercised by
process_clip on real footage).

Run from backend/:  python tests/test_intrinsics.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.intrinsics import fov_from_model  # noqa: E402


def test_fov_presets():
    fov, name = fov_from_model("iPhone 17 Pro")
    assert fov == 68.0 and "iPhone" in name
    fov, _ = fov_from_model("iphone 12 mini")
    assert fov == 68.0
    fov, name = fov_from_model("GoPro HERO12 Black")
    assert fov == 118.0
    assert fov_from_model("Some Unknown Cam 3000") is None
    assert fov_from_model(None) is None
    assert fov_from_model("") is None
    print("ok: device FOV presets")


if __name__ == "__main__":
    test_fov_presets()
    print("ALL TESTS PASSED")
