"""Body-pose assembly tests: mount-aware provenance (chest vs head/forehead),
grasp state in the output doc, and the LeRobot action vector (wrist + gripper).

Run from backend/:  python tests/test_body_pose.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from pipeline.body_pose import estimate_body_pose, body_pose_to_json  # noqa: E402
from pipeline import lerobot_export  # noqa: E402
from tests.test_grasp import open_hand, fist, _hand, _world_hand  # noqa: E402

IDENTITY_Q = [1.0, 0.0, 0.0, 0.0]
YAW90_Q = [float(np.cos(np.pi / 4)), 0.0, float(np.sin(np.pi / 4)), 0.0]


def _head_poses(n=10, quat=IDENTITY_Q, tracked=True):
    return [{"timestamp_ms": i * 100.0, "position": [0, 0, 0],
             "quaternion": list(quat), "tracked": tracked} for i in range(n)]


def _hand_frames(n=10):
    """Left hand open for the first half, fist for the second; right absent."""
    return [{"timestamp_ms": i * 100.0,
             "left_hand_landmarks": open_hand() if i < n // 2 else fist(),
             "right_hand_landmarks": None} for i in range(n)]


def test_chest_mount_provenance_and_grasp():
    prev = settings.camera_mount
    settings.camera_mount = "chest"
    try:
        frames = estimate_body_pose(_head_poses(), _hand_frames(), height_cm=170)
        f = frames[0]
        assert f.joints["head"].source == "inferred", "chest cam never sees the head"
        assert f.joints["l_wrist"].source == "measured", "left hand is in frame"
        assert f.joints["r_wrist"].source == "inferred", "right hand absent"
        assert f.left_grasp is not None and f.right_grasp is None
        assert f.left_grasp["aperture_norm"] > 0.6, "first half is an open hand"
        assert frames[-1].left_grasp["closed"] is True, "second half is a fist"

        doc = body_pose_to_json("vid", frames, 170)
        assert doc["camera_mount"] == "chest"
        assert doc["coverage"]["grasp_observed_fraction"] == 1.0
        assert doc["frames"][0]["left_grasp"]["aperture_norm"] > 0.6
        print("ok: chest mount provenance + grasp in doc")
    finally:
        settings.camera_mount = prev


def test_head_mount_provenance():
    prev = settings.camera_mount
    try:
        settings.camera_mount = "head"
        head_frames = estimate_body_pose(_head_poses(), _hand_frames(), height_cm=170)
        settings.camera_mount = "chest"
        chest_frames = estimate_body_pose(_head_poses(), _hand_frames(), height_cm=170)

        f = head_frames[0]
        # Camera rigid on the head: head/neck orientation is measured.
        assert f.joints["head"].source == "oriented"
        assert f.joints["neck"].source == "oriented"
        assert f.joints["head"].confidence > 0.4
        # Torso only gets a yaw proxy: lower confidence than on the chest rig.
        assert f.joints["spine"].confidence < chest_frames[0].joints["spine"].confidence
        assert f.joints["l_wrist"].source == "measured"

        settings.camera_mount = "head"
        doc = body_pose_to_json("vid", head_frames, 170)
        assert doc["camera_mount"] == "head"
        assert "head" in doc["provenance_legend"]["oriented"]
        print("ok: head mount provenance (head oriented, torso degraded)")
    finally:
        settings.camera_mount = prev


def test_head_mount_yaw_proxy_rotates_torso():
    """A 90-degree head yaw should carry the torso heading (yaw-only proxy):
    the chest joint's rest +Z offset rotates into +X."""
    prev = settings.camera_mount
    settings.camera_mount = "head"
    try:
        frames = estimate_body_pose(_head_poses(quat=YAW90_Q), _hand_frames(), height_cm=170)
        chest = np.array(frames[0].joints["chest"].pos)
        # rest chest = (0, 0.34, 0.034) for H=1.7m; yaw90 -> x ~= 0.034, z ~= 0
        assert chest[0] > 0.025, f"chest should rotate into +X, got {chest}"
        assert abs(chest[2]) < 0.01, f"chest +Z should vanish under yaw90, got {chest}"
        head = np.array(frames[0].joints["head"].pos)
        assert head[0] > 0.02, f"head offset should follow full VO rotation, got {head}"
        print("ok: yaw-only torso proxy rotates heading")
    finally:
        settings.camera_mount = prev


def test_lerobot_action_includes_gripper():
    prev = settings.camera_mount
    settings.camera_mount = "chest"
    try:
        frames = estimate_body_pose(_head_poses(), _hand_frames(), height_cm=170)
        doc = body_pose_to_json("vid", frames, 170)
        rows = list(lerobot_export._episode_rows(doc))
        assert len(lerobot_export.ACTION_NAMES) == 8
        assert all(len(r["action"]) == 8 for r in rows)
        # l_gripper (index 3): open at the start, closed at the end.
        assert rows[0]["action"][3] > 0.6, f"open hand -> open gripper: {rows[0]['action']}"
        assert rows[-1]["action"][3] <= 0.3, f"fist -> closed gripper: {rows[-1]['action']}"
        # r_gripper (index 7): never observed -> holds the open default.
        assert all(r["action"][7] == 1.0 for r in rows)
        # Backward compat: docs from before grasp existed still export (open default).
        for fr in doc["frames"]:
            fr.pop("left_grasp", None)
            fr.pop("right_grasp", None)
        rows_old = list(lerobot_export._episode_rows(doc))
        assert all(r["action"][3] == 1.0 for r in rows_old)
        print("ok: LeRobot action = [l_wrist xyz, l_gripper, r_wrist xyz, r_gripper]")
    finally:
        settings.camera_mount = prev


def test_wrist_depth_from_hand_scale():
    """A hand of known real size (world landmarks) at a known apparent size must
    place the wrist at the pinhole-predicted distance from the camera."""
    prev = settings.camera_mount
    settings.camera_mount = "chest"
    try:
        # Image hand: wrist at image center, middle MCP 0.05 image-heights below
        # -> palm_px = 50 px at 1000x1000. World palm = 0.07 m. fov 90 deg ->
        # f_px = 500. Predicted depth = 500 * 0.07 / 50 = 0.70 m.
        img = _hand({0: (0.5, 0.5), 4: (0.45, 0.45), 5: (0.48, 0.54), 8: (0.46, 0.40),
                     9: (0.50, 0.55), 12: (0.50, 0.40), 13: (0.52, 0.55), 16: (0.53, 0.41),
                     17: (0.54, 0.56), 20: (0.55, 0.43)})
        world = _world_hand(aperture_m=0.09, palm_m=0.07)
        frames = estimate_body_pose(
            _head_poses(n=3),
            [{"timestamp_ms": i * 100.0, "left_hand_landmarks": img,
              "left_world_landmarks": world, "right_hand_landmarks": None} for i in range(3)],
            height_cm=170, image_wh=(1000, 1000), fov_deg=90.0,
        )
        f = frames[0]
        assert f.wrist_depth_source == {"l": "scale", "r": "none"}
        H = 1.7
        cam_o = np.array([0.0, 0.22 * H, 0.10 * H])  # chest mount, identity pose
        dist = float(np.linalg.norm(np.array(f.joints["l_wrist"].pos) - cam_o))
        assert abs(dist - 0.70) < 0.02, f"expected ~0.70m from camera, got {dist:.3f}"
        doc = body_pose_to_json("vid", frames, 170)
        assert doc["coverage"]["wrist_depth_from_scale_fraction"] == 0.5  # left only
        print(f"ok: wrist depth from hand scale ({dist:.3f}m ~= 0.70m)")
    finally:
        settings.camera_mount = prev


def test_wearer_palm_length_override():
    """A measured palm length replaces MediaPipe's size prior: depth scales to
    the true hand, works even WITHOUT world landmarks, and corrects aperture_m."""
    prev_mount, prev_palm = settings.camera_mount, settings.wearer_palm_length_cm
    settings.camera_mount = "chest"
    settings.wearer_palm_length_cm = 7.5  # true palm; MediaPipe world says 7.0
    try:
        img = _hand({0: (0.5, 0.5), 4: (0.45, 0.45), 5: (0.48, 0.54), 8: (0.46, 0.40),
                     9: (0.50, 0.55), 12: (0.50, 0.40), 13: (0.52, 0.55), 16: (0.53, 0.41),
                     17: (0.54, 0.56), 20: (0.55, 0.43)})
        world = _world_hand(aperture_m=0.09, palm_m=0.07)
        frames = estimate_body_pose(
            _head_poses(n=3),
            [{"timestamp_ms": i * 100.0, "left_hand_landmarks": img,
              "left_world_landmarks": world, "right_hand_landmarks": None} for i in range(3)],
            height_cm=170, image_wh=(1000, 1000), fov_deg=90.0,
        )
        f = frames[0]
        cam_o = np.array([0.0, 0.22 * 1.7, 0.10 * 1.7])
        dist = float(np.linalg.norm(np.array(f.joints["l_wrist"].pos) - cam_o))
        # palm_px=50, f_px=500 -> depth = 500*0.075/50 = 0.75m (not 0.70 from the prior)
        assert abs(dist - 0.75) < 0.02, f"expected 0.75m with true palm, got {dist:.3f}"
        g = f.left_grasp
        assert g["scale_corrected"] is True and abs(g["hand_scale_m"] - 0.075) < 1e-6
        assert abs(g["aperture_m"] - 0.09 * (0.075 / 0.07)) < 1e-3

        # No world landmarks at all: override still enables scale depth.
        frames2 = estimate_body_pose(
            _head_poses(n=3),
            [{"timestamp_ms": i * 100.0, "left_hand_landmarks": img,
              "right_hand_landmarks": None} for i in range(3)],
            height_cm=170, image_wh=(1000, 1000), fov_deg=90.0,
        )
        assert frames2[0].wrist_depth_source["l"] == "scale"
        print("ok: wearer palm-length override (depth 0.75m, aperture corrected)")
    finally:
        settings.camera_mount = prev_mount
        settings.wearer_palm_length_cm = prev_palm


def test_wilor_measured_wrist_depth():
    """A row carrying wrist_cam (WiLoR absolute camera-frame position) must
    drive the wrist depth directly, labeled source 'wilor'."""
    prev = settings.camera_mount
    settings.camera_mount = "chest"
    try:
        img = _hand({0: (0.5, 0.5), 4: (0.45, 0.45), 5: (0.48, 0.54), 8: (0.46, 0.40),
                     9: (0.50, 0.55), 12: (0.50, 0.40), 13: (0.52, 0.55), 16: (0.53, 0.41),
                     17: (0.54, 0.56), 20: (0.55, 0.43)})
        frames = estimate_body_pose(
            _head_poses(n=3),
            [{"timestamp_ms": i * 100.0, "left_hand_landmarks": img,
              "right_hand_landmarks": None,
              "left_wrist_cam": [0.0, 0.0, 0.6]} for i in range(3)],  # 0.6m from camera
            height_cm=170, image_wh=(1000, 1000), fov_deg=90.0,
        )
        f = frames[0]
        assert f.wrist_depth_source["l"] == "wilor", f.wrist_depth_source
        cam_o = np.array([0.0, 0.22 * 1.7, 0.10 * 1.7])
        dist = float(np.linalg.norm(np.array(f.joints["l_wrist"].pos) - cam_o))
        assert abs(dist - 0.6) < 0.02, f"expected 0.6m measured depth, got {dist:.3f}"
        print(f"ok: wilor measured wrist depth ({dist:.3f}m ~= 0.60m)")
    finally:
        settings.camera_mount = prev


def test_head_mount_yaw_smoothing():
    """Rapid alternating head glances (+/-40 deg) must NOT swing the torso:
    the low-passed heading stays near zero while the raw head yaw oscillates."""
    prev_mount = settings.camera_mount
    prev_win = settings.ego_torso_yaw_smooth_seconds
    settings.camera_mount = "head"
    settings.ego_torso_yaw_smooth_seconds = 1.0
    try:
        a = 0.7  # rad, ~40 deg
        head = []
        for i in range(20):
            yaw = a if i % 2 == 0 else -a
            head.append({"timestamp_ms": i * 100.0, "position": [0, 0, 0],
                         "quaternion": [float(np.cos(yaw / 2)), 0.0, float(np.sin(yaw / 2)), 0.0],
                         "tracked": True})
        frames = estimate_body_pose(head, _hand_frames(20), height_cm=170)
        chest = np.array(frames[10].joints["chest"].pos)
        # Unsmoothed, |x| would be ~0.34*sin(40deg) ~= 0.22; smoothed ~ 0.
        assert abs(chest[0]) < 0.03, f"torso should ignore glances, chest={chest}"
        # The HEAD itself still follows the raw VO rotation (it is measured);
        # its forward offset from the neck (0.034m) swings to x ~= 0.022 at 40deg.
        head_x = abs(np.array(frames[10].joints["head"].pos)[0])
        assert head_x > 0.015, f"head should still swing with VO, head_x={head_x}"
        print("ok: torso yaw low-passed; head follows raw VO")
    finally:
        settings.camera_mount = prev_mount
        settings.ego_torso_yaw_smooth_seconds = prev_win


def test_lerobot_segment_episodes():
    """Segment mode: one episode per annotated segment, VLM description as the
    task string, timestamps re-based, per-episode video cut, and hands-free
    episodes filtered out (a manipulation dataset shouldn't ship them)."""
    import json
    import shutil as _shutil
    import subprocess
    import tempfile

    if not _shutil.which("ffmpeg"):
        print("skip: ffmpeg not available")
        return

    prev = settings.camera_mount
    settings.camera_mount = "chest"
    try:
        root = Path(tempfile.mkdtemp())
        vid = "segclip"
        (root / "processed" / vid).mkdir(parents=True)
        (root / "anonymized").mkdir()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                        "-i", "testsrc=duration=6:size=160x120:rate=10",
                        "-pix_fmt", "yuv420p",
                        str(root / "anonymized" / f"{vid}.mp4")], check=True)
        # 0-2s open hand, 2-4s fist, 4-6s NO hands (must be filtered out).
        hands = [{"timestamp_ms": i * 100.0,
                  "left_hand_landmarks": (open_hand() if i < 20 else
                                          fist() if i < 40 else None),
                  "right_hand_landmarks": None} for i in range(60)]
        frames = estimate_body_pose(_head_poses(n=60), hands, height_cm=170)
        doc = body_pose_to_json(vid, frames, 170)
        (root / "processed" / vid / "body_pose.json").write_text(json.dumps(doc))
        (root / "processed" / vid / "segments.json").write_text(json.dumps({
            "segments": [
                {"start_time": 0.0, "end_time": 2.0, "task_label": "opening jar",
                 "description": "The person unscrews the lid of a jar."},
                {"start_time": 2.0, "end_time": 4.0, "task_label": "pouring",
                 "description": "The person pours the contents into a bowl."},
                {"start_time": 4.0, "end_time": 6.0, "task_label": "looking around",
                 "description": "The person looks around the room."},
            ]}))

        out = root / "lerobot"
        summary = lerobot_export.build_lerobot_dataset([vid], out, data_root=root,
                                                       episode_mode="segment")
        assert summary["episodes"] == 2 and summary["tasks"] == 2, summary
        assert summary["skipped_episodes"] == 1, "hands-free episode must be filtered"
        problems = [p for p in lerobot_export.validate_lerobot(out)
                    if not p.startswith("INFO:")]
        assert not problems, problems

        tasks = [json.loads(l) for l in (out / "meta" / "tasks.jsonl").read_text().splitlines()]
        assert tasks[0]["task"].startswith("The person unscrews")
        assert tasks[1]["task"].startswith("The person pours")
        import pyarrow.parquet as pq
        t1 = pq.read_table(out / "data" / "chunk-000" / "episode_000001.parquet")
        ts = t1.column("timestamp").to_pylist()
        assert ts[0] < 0.15, f"episode 2 timestamps must re-base to ~0, got {ts[0]}"
        assert t1.column("task_index").to_pylist()[0] == 1
        assert (out / "videos" / "chunk-000" / "observation.images.ego" /
                "episode_000001.mp4").exists()
        # Clip mode still produces a single generic episode.
        summary2 = lerobot_export.build_lerobot_dataset([vid], root / "lr2", data_root=root,
                                                        episode_mode="clip")
        assert summary2["episodes"] == 1 and summary2["tasks"] == 1
        print("ok: segment-per-episode export (2 episodes, language tasks, re-based time)")
    finally:
        settings.camera_mount = prev


if __name__ == "__main__":
    test_chest_mount_provenance_and_grasp()
    test_head_mount_provenance()
    test_head_mount_yaw_proxy_rotates_torso()
    test_lerobot_action_includes_gripper()
    test_wrist_depth_from_hand_scale()
    test_wilor_measured_wrist_depth()
    test_head_mount_yaw_smoothing()
    test_lerobot_segment_episodes()
    print("ALL TESTS PASSED")
