"""Backend body-pose assembly — the lab-facing skeletal DATA product.

This promotes pose from a frontend visualization (Pose3D.tsx) to a first-class
backend output: a per-frame skeleton in a defined joint convention where EVERY
joint carries a confidence and a provenance flag. That honesty is what makes the
data usable to a robot-learning lab — they can filter/weight by confidence and
know exactly which joints are measured vs inferred.

Provenance depends on where the camera is mounted (settings.camera_mount):

CHEST mount — the camera is rigid on the torso, so VO measures TORSO orientation:
  measured  — directly observed. Hands (MediaPipe, when in frame).
  oriented  — position templated, but ORIENTATION is measured (torso/chest).
  ik        — solved from measured joints. Elbows (2-bone IK shoulder->wrist).
  inferred  — never observed by a chest cam; an anthropometric prior. Head/neck
              (above the camera) and the entire lower body (legs).

HEAD (forehead) mount — the camera is rigid on the head, so VO measures HEAD
orientation. The head/neck become "oriented" (their orientation IS measured);
the torso only inherits a yaw-only heading proxy from the head, so torso joints
keep "oriented" provenance but at reduced confidence. The camera rides on the
neck pivot for hand back-projection.

Nothing here claims to *measure* the unseen body. Legs (and the head, on a chest
rig) are explicitly low-confidence priors. The value is the measured hands +
grasp state + arms + camera-rig motion.

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

from .grasp import grasp_features, hand_scale_m

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
_TORSO = {"spine", "chest", "l_shoulder", "r_shoulder"}
_HEAD_CHAIN = {"neck", "head"}
_PALM_MCP = 9  # MediaPipe middle-finger MCP; wrist->MCP = palm length


@dataclass
class Joint:
    pos: List[float]      # [x, y, z] meters, body frame
    confidence: float     # 0..1
    source: str           # measured | oriented | ik | inferred


@dataclass
class BodyFrame:
    timestamp_ms: float
    root_quaternion: List[float]   # camera-rig orientation [w,x,y,z], measured by VO
    root_tracked: bool             # did VO have a fix this frame
    joints: Dict[str, Joint]
    # 21-pt measured hand landmarks (image-normalized x,y,z), null when no hand.
    left_hand: Optional[List[List[float]]]
    right_hand: Optional[List[List[float]]]
    # Grasp state derived from the landmarks (aperture/curl/closed), null when
    # no hand. aperture_norm is the gripper channel in robot-format exports;
    # aperture_m/hand_scale_m (meters) appear when world landmarks exist.
    left_grasp: Optional[dict] = None
    right_grasp: Optional[dict] = None
    # How each wrist's depth was resolved: scale (measured from real hand size)
    # | reach (arm-reach bounded inference) | none (hand not detected).
    wrist_depth_source: Optional[Dict[str, str]] = None


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


def _yaw_angle(R: np.ndarray) -> float:
    """Heading angle (radians about +Y) of R's forward vector; 0 if vertical."""
    f = R @ np.array([0.0, 0.0, 1.0])
    x, z = float(f[0]), float(f[2])
    if np.hypot(x, z) < 1e-6:
        return 0.0
    return float(np.arctan2(x, z))


def _R_from_yaw(a: float) -> np.ndarray:
    c, s = float(np.cos(a)), float(np.sin(a))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _yaw_only(R: np.ndarray) -> np.ndarray:
    """Heading (rotation about +Y) component of R — the only part of a HEAD
    orientation that plausibly transfers to the torso. Identity if the forward
    vector is near-vertical (heading undefined)."""
    return _R_from_yaw(_yaw_angle(R))


def _smooth_yaw(yaws: List[float], times_ms: List[float], window_s: float) -> List[float]:
    """Circular moving average of heading angles (low-pass): heads glance, torsos
    don't. Averaged as unit vectors so the +/-pi wrap is handled correctly."""
    if window_s <= 0 or len(yaws) < 2:
        return list(yaws)
    half = window_s * 500.0  # half-window in ms
    t = np.asarray(times_ms, dtype=np.float64)
    vx, vz = np.sin(yaws), np.cos(yaws)
    out = []
    for i in range(len(yaws)):
        m = np.abs(t - t[i]) <= half
        sx, sz = float(vx[m].sum()), float(vz[m].sum())
        out.append(float(np.arctan2(sx, sz)) if (sx * sx + sz * sz) > 1e-12 else yaws[i])
    return out


def _cfg(name: str, default: float) -> float:
    return float(getattr(settings, name, default)) if settings is not None else default


def _camera_mount() -> str:
    """Configured mount ("chest" | "head"); defaults to chest."""
    m = str(getattr(settings, "camera_mount", "chest") or "chest") if settings is not None else "chest"
    return m if m in ("chest", "head") else "chest"


def _correct_grasp_scale(g: Optional[dict]) -> Optional[dict]:
    """Rescale metric grasp fields by the wearer's REAL palm length when
    configured — MediaPipe's world landmarks carry a generic hand-size prior,
    so all its metric distances are off by the same ratio."""
    palm_true = _cfg("wearer_palm_length_cm", 0.0) / 100.0
    if not g or palm_true <= 0 or not g.get("hand_scale_m"):
        return g
    corr = palm_true / g["hand_scale_m"]
    out = {**g, "hand_scale_m": round(palm_true, 4), "scale_corrected": True}
    if "aperture_m" in g:
        out["aperture_m"] = round(g["aperture_m"] * corr, 4)
    return out


@dataclass
class CameraModel:
    """Pinhole + wearable-mount geometry to back-project image points into body space."""
    width: int
    height: int
    fov_deg: float
    mount_height: float   # rest-pose body-frame Y of the camera (meters)
    mount_forward: float  # rest-pose body-frame Z of the camera (meters)
    pitch_deg: float      # downward tilt

    @property
    def rest_offset(self) -> np.ndarray:
        """Rest-pose camera position from the pelvis (meters)."""
        return np.array([0.0, self.mount_height, self.mount_forward])

    @property
    def f_px(self) -> float:
        """Focal length in pixels. The configured FOV is the camera's
        HORIZONTAL field of view in sensor orientation, i.e. it spans the LONG
        side — in portrait video that's the vertical axis. Using the long side
        keeps f correct in either orientation (square pixels assumed)."""
        return 0.5 * max(self.width, self.height) / np.tan(np.radians(self.fov_deg) / 2.0)

    @property
    def K_inv(self) -> np.ndarray:
        f = self.f_px
        cx, cy = self.width / 2.0, self.height / 2.0
        return np.array([[1 / f, 0, -cx / f], [0, 1 / f, -cy / f], [0, 0, 1]])

    def ray_dir(self, u_norm: float, v_norm: float, R: np.ndarray) -> np.ndarray:
        """Unit ray (body frame) through a normalized image point, incl. rig R."""
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


def _wrist_from_image(hand: Optional[List[List[float]]],
                      world: Optional[List[List[float]]],
                      H: float, shoulder: np.ndarray,
                      cam_o: np.ndarray, R_cam: np.ndarray, cam: CameraModel,
                      wrist_cam: Optional[List[float]] = None,
                      ) -> Tuple[np.ndarray, bool, str]:
    """Image-consistent wrist placement.

    The 2D direction is MEASURED (back-projected through the real intrinsics, so
    it reprojects onto the detected hand pixel). Depth along the ray, best first:
      "wilor" — the hand model's ABSOLUTE camera-frame position (WiLoR predicts
                metric translation): a direct depth measurement.
      "scale" — recovered from the hand's REAL size (world landmarks, meters)
                vs its apparent pixel size: depth = f_px * palm_m / palm_px.
      "reach" — bounded by arm reach from the shoulder (the old inference).
      "none"  — hand not detected; anthropometric fallback pose.
    cam_o/R_cam are the camera's body-frame position and orientation this frame
    (mount-dependent). Returns (pos, measured, depth_source).
    """
    if not hand:
        return shoulder + np.array([0.0, -0.38 * H, 0.12 * H]), False, "none"
    w = hand[0]  # wrist landmark, normalized image coords
    o = cam_o
    d = cam.ray_dir(w[0], w[1], R_cam)
    reach = _cfg("ego_arm_reach_frac", 0.30) * H

    t = None
    src = "reach"
    if wrist_cam is not None and len(wrist_cam) == 3:
        depth = float(np.linalg.norm(np.asarray(wrist_cam, dtype=np.float64)))
        if 0.05 < depth < 2.0:  # physical sanity for a wearable camera
            t_max = float(np.linalg.norm(shoulder - o)) + reach * 1.05
            t = float(min(max(depth, 0.08), t_max))
            src = "wilor"
    # Real measured palm length (config) beats MediaPipe's generic hand-size
    # prior — and lets depth-from-scale work even without world landmarks.
    palm_true = _cfg("wearer_palm_length_cm", 0.0) / 100.0
    palm_m = palm_true if palm_true > 0 else hand_scale_m(world)
    if t is None and palm_m:
        dx = (hand[_PALM_MCP][0] - hand[0][0]) * cam.width
        dy = (hand[_PALM_MCP][1] - hand[0][1]) * cam.height
        palm_px = float(np.hypot(dx, dy))
        if palm_px > 1e-6:
            t_scale = cam.f_px * palm_m / palm_px
            # Physical plausibility clamp: the wrist can't be farther from the
            # camera than the shoulder distance plus a full arm (small slack).
            t_max = float(np.linalg.norm(shoulder - o)) + reach * 1.05
            t = float(min(max(t_scale, 0.08), t_max))
            src = "scale"
    if t is None:
        t = _ray_sphere_far(o, d, shoulder, reach)
        if t is None:
            # Ray doesn't reach the arm-reach sphere: use the closest point on the
            # ray to the shoulder (most plausible reachable wrist along the ray).
            t = max(0.05, float(np.dot(shoulder - o, d)))
    return o + t * d, True, src


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
    """Fuse measured camera-rig motion (VO) + measured hands into a skeleton.

    head_poses: list of {timestamp_ms, quaternion[w,x,y,z], tracked, ...} (VO).
    hand_frames: list of {timestamp_ms, left_hand_landmarks, right_hand_landmarks}.
    image_wh: anonymized-video resolution, for camera-intrinsic back-projection.
    Output is time-sampled on the hand frames (the manipulation-relevant clock);
    the nearest VO pose supplies the rig orientation. What that orientation
    *means* depends on settings.camera_mount: "chest" = torso orientation,
    "head" = head orientation (torso gets a yaw-only heading proxy).
    """
    H = (height_cm or 170.0) / 100.0
    Lu, Lf = 0.17 * H, 0.15 * H
    mount = _camera_mount()
    if mount == "head":
        mount_h = _cfg("ego_head_mount_height_frac", 0.42)
        mount_f = _cfg("ego_head_mount_forward_frac", 0.05)
        mount_p = _cfg("ego_head_mount_pitch_deg", 20.0)
    else:
        mount_h = _cfg("ego_chest_mount_height_frac", 0.22)
        mount_f = _cfg("ego_chest_mount_forward_frac", 0.10)
        mount_p = _cfg("ego_chest_mount_pitch_deg", 12.0)
    cam = CameraModel(
        width=int(image_wh[0]), height=int(image_wh[1]),
        fov_deg=float(fov_deg if fov_deg is not None else _cfg("ego_camera_fov_deg", 90.0)),
        mount_height=mount_h * H,
        mount_forward=mount_f * H,
        pitch_deg=mount_p,
    )
    # Per-mount provenance/confidence overrides. On a head rig the head/neck
    # orientation IS measured (camera rigid on the head), while the torso only
    # inherits a yaw-only heading proxy — honest, lower confidence.
    overrides: Dict[str, Tuple[str, float]] = {}
    torso_factor = 1.0
    if mount == "head":
        overrides = {"head": ("oriented", 0.55), "neck": ("oriented", 0.45)}
        torso_factor = 0.75
    rest = {name: np.array(frac) * H for name, frac, _, _ in _T}

    hp_times = np.array([h["timestamp_ms"] for h in head_poses]) if head_poses else np.array([])

    def nearest_head(ms: float) -> Optional[dict]:
        if not len(hp_times):
            return None
        return head_poses[int(np.argmin(np.abs(hp_times - ms)))]

    # Head mount: precompute the torso heading as a LOW-PASSED head yaw — heads
    # glance around several times a second, torsos don't follow every glance.
    torso_yaw: Optional[List[float]] = None
    if mount == "head" and hand_frames:
        frame_ms = [hf["timestamp_ms"] for hf in hand_frames]
        raw = []
        for ms in frame_ms:
            hp = nearest_head(ms)
            raw.append(_yaw_angle(_quat_to_R(hp["quaternion"])) if hp else 0.0)
        torso_yaw = _smooth_yaw(raw, frame_ms, _cfg("ego_torso_yaw_smooth_seconds", 1.0))

    out: List[BodyFrame] = []
    for fi, hf in enumerate(hand_frames):
        ms = hf["timestamp_ms"]
        hp = nearest_head(ms)
        R_vo = _quat_to_R(hp["quaternion"]) if hp else np.eye(3)
        tracked = bool(hp["tracked"]) if hp else False
        root_q = hp["quaternion"] if hp else [1.0, 0.0, 0.0, 0.0]
        # What the VO rotation applies to depends on the mount.
        R_torso = R_vo if mount == "chest" else _R_from_yaw(torso_yaw[fi] if torso_yaw else 0.0)

        joints: Dict[str, Joint] = {}
        # 1) template joints; rotate by what is actually MEASURED for this mount.
        tmpl: Dict[str, np.ndarray] = {}
        for name, frac, prov, conf in _T:
            p = rest[name]
            if name in _TORSO:
                p = R_torso @ p
            elif name in _HEAD_CHAIN:
                if mount == "chest":
                    p = R_torso @ p                  # rides the torso (prior)
                elif name == "neck":
                    p = R_torso @ p                  # neck pivots with heading
                else:  # head, head mount: orientation measured by VO
                    p = (R_torso @ rest["neck"]) + R_vo @ (rest["head"] - rest["neck"])
            tmpl[name] = p

        # Camera pose this frame: rigid on the torso (chest) or on the head
        # (forehead strap riding the neck pivot).
        if mount == "head":
            neck_p = R_torso @ rest["neck"]
            cam_o = neck_p + R_vo @ (cam.rest_offset - rest["neck"])
            R_cam = R_vo
        else:
            cam_o = R_torso @ cam.rest_offset
            R_cam = R_torso

        # 2) wrists from MEASURED hands (override template), then IK elbows.
        lw, l_meas, l_src = _wrist_from_image(
            hf.get("left_hand_landmarks"), hf.get("left_world_landmarks"),
            H, tmpl["l_shoulder"], cam_o, R_cam, cam,
            wrist_cam=hf.get("left_wrist_cam"))
        rw, r_meas, r_src = _wrist_from_image(
            hf.get("right_hand_landmarks"), hf.get("right_world_landmarks"),
            H, tmpl["r_shoulder"], cam_o, R_cam, cam,
            wrist_cam=hf.get("right_wrist_cam"))
        tmpl["l_wrist"], tmpl["r_wrist"] = lw, rw
        tmpl["l_elbow"] = _ik_elbow(tmpl["l_shoulder"], lw, Lu, Lf)
        tmpl["r_elbow"] = _ik_elbow(tmpl["r_shoulder"], rw, Lu, Lf)

        # 3) emit joints with provenance + confidence (degrade by VO tracking /
        #    hand presence; torso degraded further on a head rig — yaw proxy only).
        for name, frac, prov, base in _T:
            source, base = overrides.get(name, (prov, base))
            conf = base
            if name in _TORSO or name in _HEAD_CHAIN:
                conf = base * (1.0 if tracked else 0.4)
                if name in _TORSO:
                    conf *= torso_factor
            # Measured wrists/elbows are further weighted by the hand's shape
            # plausibility (bone-length consistency): low = the landmark model
            # was hallucinating occluded joints (fingers wrapped around an
            # object), so the position is suspect even though it was "seen".
            l_p = hf.get("left_shape_plausibility")
            r_p = hf.get("right_shape_plausibility")
            if name in ("l_wrist", "l_elbow"):
                conf = base * (0.5 + 0.5 * l_p if l_meas and l_p is not None
                               else (1.0 if l_meas else 0.25))
                source = prov if l_meas else "inferred"
            if name in ("r_wrist", "r_elbow"):
                conf = base * (0.5 + 0.5 * r_p if r_meas and r_p is not None
                               else (1.0 if r_meas else 0.25))
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
            left_grasp=_correct_grasp_scale(grasp_features(
                hf.get("left_hand_landmarks"), hf.get("left_world_landmarks"))),
            right_grasp=_correct_grasp_scale(grasp_features(
                hf.get("right_hand_landmarks"), hf.get("right_world_landmarks"))),
            wrist_depth_source={"l": l_src, "r": r_src},
        ))
    return out


def body_pose_to_json(video_id: str, frames: List[BodyFrame], height_cm: float) -> dict:
    n = len(frames)
    mount = _camera_mount()
    measured_wrist = sum(
        1 for f in frames
        for k in ("l_wrist", "r_wrist") if f.joints[k].source == "measured"
    )
    grasp_frames = sum(1 for f in frames if f.left_grasp or f.right_grasp)
    scale_depth = sum(
        1 for f in frames
        for k in ("l", "r") if (f.wrist_depth_source or {}).get(k) in ("scale", "wilor")
    )
    if mount == "head":
        oriented_legend = ("position templated, orientation measured by head VO "
                           "(head/neck) or its yaw-only heading proxy (torso)")
        inferred_legend = "anthropometric prior; NOT observed by a head-mounted cam"
    else:
        oriented_legend = "position templated, orientation measured by torso VO"
        inferred_legend = "anthropometric prior; NOT observed by a chest cam"
    return {
        "schema": SCHEMA_VERSION,
        "video_id": video_id,
        "units": "meters",
        "frame": "body: origin=pelvis, +X=right, +Y=up, +Z=forward",
        "camera_mount": mount,
        "joint_names": JOINT_NAMES,
        "skeleton_edges": SKELETON_EDGES,
        "provenance_legend": {
            "measured": "directly observed (hands)",
            "oriented": oriented_legend,
            "ik": "solved by inverse kinematics from measured joints",
            "inferred": inferred_legend,
        },
        "height_cm": height_cm,
        "frame_count": n,
        "coverage": {
            "wrist_measured_fraction": round(measured_wrist / (2 * n), 3) if n else 0.0,
            "grasp_observed_fraction": round(grasp_frames / n, 3) if n else 0.0,
            # Fraction of wrist placements whose DEPTH was recovered from real
            # hand size (world landmarks) rather than reach-bounded inference.
            "wrist_depth_from_scale_fraction": round(scale_depth / (2 * n), 3) if n else 0.0,
        },
        "wrist_depth_legend": {
            "wilor": "absolute metric camera-frame position predicted by the hand model (measured)",
            "scale": "depth from real hand size vs apparent size (measured, up to intrinsics)",
            "reach": "depth bounded by arm reach (inferred)",
            "none": "hand not detected (anthropometric fallback)",
        },
        "frames": [
            {
                "timestamp_ms": f.timestamp_ms,
                "root_quaternion": f.root_quaternion,
                "root_tracked": f.root_tracked,
                "joints": {k: asdict(v) for k, v in f.joints.items()},
                "left_hand": f.left_hand,
                "right_hand": f.right_hand,
                "left_grasp": f.left_grasp,
                "right_grasp": f.right_grasp,
                "wrist_depth_source": f.wrist_depth_source,
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
