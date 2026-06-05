"""Backend body-pose assembly — the lab-facing skeletal DATA product.

This promotes pose from a frontend visualization (Pose3D.tsx) to a first-class
backend output: a per-frame skeleton in a defined joint convention where EVERY
joint carries a confidence and a provenance flag. That honesty is what makes the
data usable to a robot-learning lab — they can filter/weight by confidence and
know exactly which joints are measured vs inferred.

Provenance, for a CHEST-mounted camera (the current rig):
  measured  — directly observed. Hands (MediaPipe, when in frame).
  oriented  — position templated, but ORIENTATION is measured. Torso/chest:
              the camera is on the torso, so VO measures torso orientation.
  ik        — solved from measured joints. Elbows (2-bone IK shoulder->wrist).
  inferred  — never observed by a chest cam; an anthropometric prior. Head/neck
              (above the camera) and the entire lower body (legs).

Nothing here claims to *measure* the unseen body. Legs and head are explicitly
low-confidence priors. The value is the measured hands + arms + torso motion.

Units: meters, body frame. Origin at the pelvis. +X = subject's right, +Y = up,
+Z = forward. Scaled by operator height.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from config import settings
except Exception:  # pragma: no cover - allows import without app config
    settings = None

log = logging.getLogger("revisent.body_pose")

SCHEMA_VERSION = "body_pose/1.0"

# Defined joint convention (SMPL-ordering-compatible names; a superset can be
# fit to SMPL-X downstream). Each entry: (name, template position as fractions of
# height H from the pelvis, default provenance, base confidence).
#   pos is the rest-pose template; orientation/measurement is applied per frame.
_T = [
    # name        x/H     y/H     z/H    provenance   base_conf
    ("pelvis",   ( 0.00,  0.00,  0.00), "inferred", 0.40),
    ("spine",    ( 0.00,  0.12,  0.01), "oriented", 0.55),
    ("chest",    ( 0.00,  0.20,  0.02), "oriented", 0.55),
    ("neck",     ( 0.00,  0.30,  0.00), "inferred", 0.25),
    ("head",     ( 0.00,  0.40,  0.02), "inferred", 0.20),
    ("l_shoulder", (-0.11, 0.28,  0.00), "oriented", 0.50),
    ("r_shoulder", ( 0.11, 0.28,  0.00), "oriented", 0.50),
    ("l_elbow",  (-0.16,  0.12,  0.05), "ik",       0.45),
    ("r_elbow",  ( 0.16,  0.12,  0.05), "ik",       0.45),
    ("l_wrist",  (-0.18, -0.02,  0.18), "measured", 0.90),
    ("r_wrist",  ( 0.18, -0.02,  0.18), "measured", 0.90),
    ("l_hip",    (-0.09,  0.00,  0.00), "inferred", 0.25),
    ("r_hip",    ( 0.09,  0.00,  0.00), "inferred", 0.25),
    ("l_knee",   (-0.09, -0.27,  0.03), "inferred", 0.12),
    ("r_knee",   ( 0.09, -0.27,  0.03), "inferred", 0.12),
    ("l_ankle",  (-0.09, -0.52,  0.05), "inferred", 0.10),
    ("r_ankle",  ( 0.09, -0.52,  0.05), "inferred", 0.10),
]

# Kinematic tree (parent -> child) for the defined skeleton, for retargeting/export.
SKELETON_EDGES = [
    ("pelvis", "spine"), ("spine", "chest"), ("chest", "neck"), ("neck", "head"),
    ("chest", "l_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("chest", "r_shoulder"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("pelvis", "l_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ("pelvis", "r_hip"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
]
JOINT_NAMES = [name for name, *_ in _T]
_UPPER = {"spine", "chest", "neck", "head", "l_shoulder", "r_shoulder"}


@dataclass
class Joint:
    pos: List[float]      # [x, y, z] meters, body frame
    confidence: float     # 0..1
    source: str           # measured | oriented | ik | inferred


@dataclass
class BodyFrame:
    timestamp_ms: float
    root_quaternion: List[float]   # torso orientation [w,x,y,z], measured by VO
    root_tracked: bool             # did VO have a fix this frame
    joints: Dict[str, Joint]
    # 21-pt measured hand landmarks (image-normalized x,y,z), null when no hand.
    left_hand: Optional[List[List[float]]]
    right_hand: Optional[List[List[float]]]


# ---------------------------------------------------------------------------
# math helpers
# ---------------------------------------------------------------------------
def _quat_to_R(q: List[float]) -> np.ndarray:
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-9:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z),     s * (x * z + w * y)],
        [s * (x * y + w * z),     1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y),     s * (y * z + w * x),     1 - s * (x * x + y * y)],
    ])


def _ik_elbow(S: np.ndarray, T: np.ndarray, Lu: float, Lf: float) -> np.ndarray:
    """2-bone IK: elbow position between shoulder S and (clamped) wrist target T."""
    d_vec = T - S
    d = float(np.linalg.norm(d_vec)) or 1e-6
    d = max(abs(Lu - Lf) + 1e-3, min(d, Lu + Lf - 1e-3))
    n = d_vec / (np.linalg.norm(d_vec) or 1e-6)
    a = (Lu * Lu + d * d - Lf * Lf) / (2 * d)
    h = np.sqrt(max(0.0, Lu * Lu - a * a))
    ref = np.array([0.0, -1.0, -0.3])           # bend elbow down-and-back
    perp = ref - n * float(np.dot(ref, n))
    perp = perp / (np.linalg.norm(perp) or 1e-6)
    return S + n * a + perp * h


def _cfg(name: str, default: float) -> float:
    return float(getattr(settings, name, default)) if settings is not None else default


@dataclass
class CameraModel:
    """Pinhole + chest-mount geometry to back-project image points into body space."""
    width: int
    height: int
    fov_deg: float
    mount_height: float   # body-frame Y of the camera (meters)
    mount_forward: float  # body-frame Z of the camera (meters)
    pitch_deg: float      # downward tilt

    @property
    def K_inv(self) -> np.ndarray:
        f = 0.5 * self.width / np.tan(np.radians(self.fov_deg) / 2.0)
        cx, cy = self.width / 2.0, self.height / 2.0
        return np.array([[1 / f, 0, -cx / f], [0, 1 / f, -cy / f], [0, 0, 1]])

    def origin(self, R: np.ndarray) -> np.ndarray:
        """Camera position in body frame (rotates with the torso)."""
        return R @ np.array([0.0, self.mount_height, self.mount_forward])

    def ray_dir(self, u_norm: float, v_norm: float, R: np.ndarray) -> np.ndarray:
        """Unit ray (body frame) through a normalized image point, incl. torso R."""
        u, v = u_norm * self.width, v_norm * self.height
        d_opt = self.K_inv @ np.array([u, v, 1.0])      # optical frame: x-right, y-down, z-fwd
        d = np.array([d_opt[0], -d_opt[1], d_opt[2]])   # to body-aligned (y up)
        d /= (np.linalg.norm(d) or 1e-6)
        p = np.radians(self.pitch_deg)                   # mount pitch about X (look down)
        Rp = np.array([[1, 0, 0],
                       [0, np.cos(p), -np.sin(p)],
                       [0, np.sin(p), np.cos(p)]])
        # camera looks down by pitch; image z(+fwd) maps onto a slightly-down fwd dir
        d = Rp @ d
        d = R @ d
        return d / (np.linalg.norm(d) or 1e-6)


def _ray_sphere_far(o: np.ndarray, d: np.ndarray, c: np.ndarray, r: float) -> Optional[float]:
    """Far positive intersection t of ray o+t*d with sphere(center c, radius r)."""
    oc = o - c
    b = 2 * float(np.dot(oc, d))
    cc = float(np.dot(oc, oc)) - r * r
    disc = b * b - 4 * cc
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    t = (-b + sq) / 2.0
    return t if t > 0 else None


def _wrist_from_image(hand: Optional[List[List[float]]], H: float, shoulder: np.ndarray,
                      R: np.ndarray, cam: CameraModel) -> Tuple[np.ndarray, bool]:
    """Image-consistent wrist placement.

    The 2D direction is MEASURED (back-projected through the real intrinsics, so it
    reprojects onto the detected hand pixel); the depth along that ray is the
    inferred part, bounded by arm reach from the shoulder. Returns (pos, measured).
    """
    if not hand:
        return shoulder + np.array([0.0, -0.38 * H, 0.12 * H]), False
    w = hand[0]  # MediaPipe wrist landmark, normalized image coords
    o = cam.origin(R)
    d = cam.ray_dir(w[0], w[1], R)
    reach = _cfg("ego_arm_reach_frac", 0.30) * H
    t = _ray_sphere_far(o, d, shoulder, reach)
    if t is None:
        # Ray doesn't reach the arm-reach sphere: use the closest point on the ray
        # to the shoulder (most plausible reachable wrist along the measured ray).
        t = max(0.05, float(np.dot(shoulder - o, d)))
    return o + t * d, True


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def estimate_body_pose(
    head_poses: List[dict],
    hand_frames: List[dict],
    height_cm: float = 170.0,
    image_wh: Tuple[int, int] = (1080, 1920),
    fov_deg: Optional[float] = None,
) -> List[BodyFrame]:
    """Fuse measured torso motion (VO) + measured hands into a per-frame skeleton.

    head_poses: list of {timestamp_ms, quaternion[w,x,y,z], tracked, ...} (VO).
    hand_frames: list of {timestamp_ms, left_hand_landmarks, right_hand_landmarks}.
    image_wh: anonymized-video resolution, for camera-intrinsic back-projection.
    Output is time-sampled on the hand frames (the manipulation-relevant clock);
    the nearest VO pose supplies torso orientation.
    """
    H = (height_cm or 170.0) / 100.0
    Lu, Lf = 0.17 * H, 0.15 * H
    cam = CameraModel(
        width=int(image_wh[0]), height=int(image_wh[1]),
        fov_deg=float(fov_deg if fov_deg is not None else _cfg("ego_camera_fov_deg", 90.0)),
        mount_height=_cfg("ego_chest_mount_height_frac", 0.22) * H,
        mount_forward=_cfg("ego_chest_mount_forward_frac", 0.10) * H,
        pitch_deg=_cfg("ego_chest_mount_pitch_deg", 12.0),
    )

    hp_times = np.array([h["timestamp_ms"] for h in head_poses]) if head_poses else np.array([])

    def nearest_head(ms: float) -> Optional[dict]:
        if not len(hp_times):
            return None
        return head_poses[int(np.argmin(np.abs(hp_times - ms)))]

    out: List[BodyFrame] = []
    for hf in hand_frames:
        ms = hf["timestamp_ms"]
        hp = nearest_head(ms)
        R = _quat_to_R(hp["quaternion"]) if hp else np.eye(3)
        tracked = bool(hp["tracked"]) if hp else False
        root_q = hp["quaternion"] if hp else [1.0, 0.0, 0.0, 0.0]

        pelvis = np.zeros(3)
        joints: Dict[str, Joint] = {}
        # 1) template joints; rotate the upper body by MEASURED torso orientation.
        tmpl: Dict[str, np.ndarray] = {}
        for name, frac, prov, conf in _T:
            p = np.array(frac) * H
            if name in _UPPER:
                p = pelvis + R @ (p - pelvis)        # orientation is measured
            tmpl[name] = p

        # 2) wrists from MEASURED hands (override template), then IK elbows.
        lw, l_meas = _wrist_from_image(hf.get("left_hand_landmarks"), H, tmpl["l_shoulder"], R, cam)
        rw, r_meas = _wrist_from_image(hf.get("right_hand_landmarks"), H, tmpl["r_shoulder"], R, cam)
        tmpl["l_wrist"], tmpl["r_wrist"] = lw, rw
        tmpl["l_elbow"] = _ik_elbow(tmpl["l_shoulder"], lw, Lu, Lf)
        tmpl["r_elbow"] = _ik_elbow(tmpl["r_shoulder"], rw, Lu, Lf)

        # 3) emit joints with provenance + confidence (degrade by VO tracking /
        #    hand presence).
        for name, frac, prov, base in _T:
            conf, source = base, prov
            if name in _UPPER:
                conf = base * (1.0 if tracked else 0.4)
            if name in ("l_wrist", "l_elbow"):
                conf = base if l_meas else base * 0.25
                source = prov if l_meas else "inferred"
            if name in ("r_wrist", "r_elbow"):
                conf = base if r_meas else base * 0.25
                source = prov if r_meas else "inferred"
            joints[name] = Joint(
                pos=[round(float(v), 4) for v in tmpl[name]],
                confidence=round(float(conf), 3),
                source=source,
            )

        out.append(BodyFrame(
            timestamp_ms=ms,
            root_quaternion=[round(float(v), 5) for v in root_q],
            root_tracked=tracked,
            joints=joints,
            left_hand=hf.get("left_hand_landmarks"),
            right_hand=hf.get("right_hand_landmarks"),
        ))
    return out


def body_pose_to_json(video_id: str, frames: List[BodyFrame], height_cm: float) -> dict:
    n = len(frames)
    measured_wrist = sum(
        1 for f in frames
        for k in ("l_wrist", "r_wrist") if f.joints[k].source == "measured"
    )
    return {
        "schema": SCHEMA_VERSION,
        "video_id": video_id,
        "units": "meters",
        "frame": "body: origin=pelvis, +X=right, +Y=up, +Z=forward",
        "camera_mount": "chest",
        "joint_names": JOINT_NAMES,
        "skeleton_edges": SKELETON_EDGES,
        "provenance_legend": {
            "measured": "directly observed (hands)",
            "oriented": "position templated, orientation measured by torso VO",
            "ik": "solved by inverse kinematics from measured joints",
            "inferred": "anthropometric prior; NOT observed by a chest cam",
        },
        "height_cm": height_cm,
        "frame_count": n,
        "coverage": {
            "wrist_measured_fraction": round(measured_wrist / (2 * n), 3) if n else 0.0,
        },
        "frames": [
            {
                "timestamp_ms": f.timestamp_ms,
                "root_quaternion": f.root_quaternion,
                "root_tracked": f.root_tracked,
                "joints": {k: asdict(v) for k, v in f.joints.items()},
                "left_hand": f.left_hand,
                "right_hand": f.right_hand,
            }
            for f in frames
        ],
    }


if __name__ == "__main__":  # quick manual check against a processed clip
    import sys
    from pathlib import Path
    from pipeline.hand_pose import load_hand_pose

    from pipeline.video_meta import probe
    vid = sys.argv[1]
    base = Path(__file__).resolve().parents[1].parent / "data"
    hp_json = base / "processed" / vid / "head_pose.json"
    hand_pq = base / "processed" / vid / "hand_pose.parquet"
    anon = base / "anonymized" / f"{vid}.mp4"
    head_poses = json.loads(hp_json.read_text())["frames"] if hp_json.exists() else []
    hand_frames = load_hand_pose(hand_pq) if hand_pq.exists() else []
    wh = (1080, 1920)
    if anon.exists():
        m = probe(anon)
        wh = (m.width, m.height)
    print(f"head_poses={len(head_poses)} hand_frames={len(hand_frames)} res={wh}")
    frames = estimate_body_pose(head_poses, hand_frames, height_cm=175, image_wh=wh)
    doc = body_pose_to_json(vid, frames, 175)
    print(f"assembled {doc['frame_count']} body frames; "
          f"wrist_measured={doc['coverage']['wrist_measured_fraction']}")
    if frames:
        f0 = doc["frames"][len(frames) // 2]
        print("sample joints (mid-clip):")
        for k in ("chest", "r_shoulder", "r_elbow", "r_wrist", "head", "r_knee"):
            j = f0["joints"][k]
            print(f"  {k:<11} pos={j['pos']} conf={j['confidence']} src={j['source']}")
