"""Tests for profile-driven lens undistortion.

Run from backend/:  python tests/test_undistortion.py   (or via pytest if installed)

Note on architecture: undistortion is applied INSIDE the anonymization re-encode
(not a separate phase/file), so "downstream uses the undistorted video" is verified
by checking the anonymized output is undistorted — every later stage reads that
file. Tests use tiny synthetic clips to stay fast.
"""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import undistort  # noqa: E402
from pipeline.undistort import (  # noqa: E402
    Undistorter, make_undistorter, load_profile, list_profiles, _is_identity,
)
from pipeline.anonymize import anonymize_video  # noqa: E402
from pipeline.video_meta import probe  # noqa: E402

PROFILES_DIR = undistort.PROFILES_DIR


def _gradient_frame(w=160, h=120):
    """A non-uniform frame so a remap visibly changes pixels."""
    gx = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    gy = np.tile(np.linspace(0, 255, h, dtype=np.uint8)[:, None], (1, w))
    return np.stack([gx, gy, (gx // 2 + gy // 2).astype(np.uint8)], axis=2)


def _make_clip(path: Path, n=12, w=160, h=120):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    base = _gradient_frame(w, h)
    for _ in range(n):
        vw.write(base.copy())
    vw.release()


def _write_profile(name: str, profile: dict) -> Path:
    p = PROFILES_DIR / f"{name}.json"
    p.write_text(json.dumps(profile))
    return p


def _calibrated_profile(w=160, h=120) -> dict:
    """A standard-model profile with real (non-zero) barrel distortion."""
    f = float(w)
    return {
        "camera_model": "_test_cal",
        "lens_model": "standard",
        "image_size": [w, h],
        "intrinsic_matrix": [[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]],
        "distortion_coefficients": [-0.25, 0.05, 0.0, 0.0, 0.0],
        "reprojection_error_pixels": 0.42,
    }


def test_default_profile_is_identity():
    prof = load_profile("default")
    assert _is_identity(prof)
    u = Undistorter(160, 120, prof)
    assert u.active is False
    frame = _gradient_frame()
    assert np.array_equal(u.apply(frame), frame)              # exact no-op
    assert make_undistorter(160, 120, "default") is None       # skipped entirely
    print("ok: default profile is identity (no-op)")


def test_undistortion_handles_missing_profile():
    prof = load_profile("does_not_exist_cam")                  # logs WARNING, falls back
    assert prof.get("camera_model") == "default"
    assert make_undistorter(160, 120, "does_not_exist_cam") is None
    print("ok: missing profile falls back to default (no-op)")


def test_list_profiles_includes_registry():
    names = {p["camera_model"] for p in list_profiles()}
    assert {"default", "transcend_dpb30", "gopro_hero"}.issubset(names)
    print("ok: registry lists default + placeholders")


def test_calibrated_profile_changes_frames_and_preserves_size():
    _write_profile("_test_cal", _calibrated_profile())
    try:
        u = make_undistorter(160, 120, "_test_cal")
        assert u is not None and u.active
        frame = _gradient_frame()
        out = u.apply(frame)
        assert out.shape == frame.shape                        # resolution preserved
        assert not np.array_equal(out, frame)                  # actually undistorted
        m = u.meta()
        assert m["applied"] and m["camera_model"] == "_test_cal" and m["lens_model"] == "standard"
        print("ok: calibrated profile undistorts and preserves resolution")
    finally:
        (PROFILES_DIR / "_test_cal.json").unlink(missing_ok=True)


def test_intrinsics_scale_to_frame_size():
    # Calibrated at 160x120 but applied to a 320x240 frame — must not error and
    # must produce a valid same-size remap.
    _write_profile("_test_cal", _calibrated_profile(160, 120))
    try:
        u = make_undistorter(320, 240, "_test_cal")
        assert u is not None and u.active
        out = u.apply(_gradient_frame(320, 240))
        assert out.shape == (240, 320, 3)
        print("ok: intrinsics scale from calibration size to frame size")
    finally:
        (PROFILES_DIR / "_test_cal.json").unlink(missing_ok=True)


def test_anonymize_marks_undistort_in_method_and_meta():
    _write_profile("_test_cal", _calibrated_profile())
    try:
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "src.mp4", Path(d) / "out.mp4"
            _make_clip(src)

            # default camera -> no undistortion
            r0 = anonymize_video(src, Path(d) / "out0.mp4", camera_model="default")
            assert "+undistort" not in r0.method
            assert (r0.undistort or {"applied": False})["applied"] is False

            # calibrated camera -> undistortion applied + recorded
            r1 = anonymize_video(src, out, camera_model="_test_cal")
            assert "+undistort" in r1.method
            assert r1.undistort and r1.undistort["applied"] is True
            assert r1.undistort["camera_model"] == "_test_cal"
            m = probe(out)
            assert (m.frame_count > 0) or (m.duration_seconds > 0)
            print("ok: anonymize records undistort in method + meta when calibrated")
    finally:
        (PROFILES_DIR / "_test_cal.json").unlink(missing_ok=True)


if __name__ == "__main__":
    test_default_profile_is_identity()
    test_undistortion_handles_missing_profile()
    test_list_profiles_includes_registry()
    test_calibrated_profile_changes_frames_and_preserves_size()
    test_intrinsics_scale_to_frame_size()
    test_anonymize_marks_undistort_in_method_and_meta()
    print("ALL TESTS PASSED")
