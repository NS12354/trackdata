"""Application configuration loaded from environment / .env file."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ directory and repo root
BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Environment ---
    # "development" | "production" — gates auth enforcement, debug behavior, etc.
    environment: str = "development"

    # --- Storage ---
    # Root of the local data tree. Subdirs: uploads/ anonymized/ processed/ exports/
    data_dir: Path = REPO_ROOT / "data"
    storage_backend: str = "local"  # "local" (S3 backend is a future swap)

    # --- Database ---
    # SQLite for local dev; set DATABASE_URL to a postgresql:// URL in production
    # (docker-compose provides one). The engine adapts pool settings per dialect.
    database_url: str = f"sqlite:///{REPO_ROOT / 'data' / 'revisent.db'}"
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_pre_ping: bool = True

    # --- Job queue (Celery + Redis) ---
    redis_url: str = "redis://localhost:6379/0"
    # Broker/result default to redis_url but can be overridden independently.
    celery_broker_url: str = ""
    celery_result_backend: str = ""
    # Run tasks synchronously in-process (tests / no-broker dev).
    celery_task_always_eager: bool = False
    # Max task retries on transient failure, with exponential backoff.
    task_max_retries: int = 3

    # --- Uploads ---
    max_upload_bytes: int = 5 * 1024 * 1024 * 1024  # 5 GB

    # --- Privacy ---
    # If True, keep the raw upload after successful anonymization (dev/debug only).
    # Per privacy principle, raw uploads are deleted once anonymization succeeds.
    retain_raw_uploads: bool = True

    # --- Anonymization ---
    # Master switch for face blurring on UPLOADED clips. With it off, uploads
    # get a fast orientation-corrected passthrough (GPU NVENC when available)
    # instead of the CPU-heavy detect+blur+encode pass. ONLY for footage with
    # no bystanders — re-enable before processing anyone else's face.
    blur_enabled: bool = True
    # Candidate detection threshold — deliberately low so weak frames of a real
    # face are still captured and can be rescued by a confirmed track. Stray
    # false positives that clear this bar are filtered out by track confirmation.
    face_detection_confidence: float = 0.5
    # A face *track* is only blurred if at least one of its detections scores at
    # or above this. Measured separation: false positives (hands/objects) top out
    # ~0.71, real faces reach 0.90 with many detections >=0.75. So a track that
    # never produces a strong detection is treated as a false positive and dropped
    # — this is what stops hands/objects from being blurred. Lower it if you have
    # only small/distant faces that never score high (trades in more false blur).
    face_confirm_confidence: float = 0.74
    # Detection model: 0 = short-range (within ~2m, faces large), 1 = full-range.
    face_detection_model: int = 1
    # Gaussian blur kernel multiplier relative to detected face box size.
    blur_strength: float = 0.6
    # Pad each detected face box outward by this fraction before blurring.
    face_box_padding: float = 0.2

    # --- Temporal persistence (anti-flicker) ---
    # Detection misses on individual frames (motion blur, profile, partial face)
    # would otherwise leave a face un-blurred for a moment. We build face tracks
    # across the whole clip and fill short gaps so the blur stays put.
    # Max gap (seconds) to bridge between two detections of the same face.
    temporal_max_gap_seconds: float = 0.8
    # Hold the blur this long before a face first appears / after it last appears.
    temporal_hold_seconds: float = 0.4
    # Grow held/interpolated boxes by this fraction per held frame (covers motion
    # while the face is undetected), capped internally.
    track_dilation_per_frame: float = 0.06
    # IoU above which a detection is matched to an existing track.
    track_match_iou: float = 0.1
    # Run face detection on a copy downscaled so its longest side is at most this
    # many pixels (boxes are normalized, so they map back to full res). Speeds up
    # detection on HD video; 0 disables downscaling. Verified recall-safe at 640
    # for near faces; 960 is a conservative default that better preserves small /
    # distant faces on real footage. Raise (or set 0) if you have tiny faces.
    face_detection_max_dim: int = 960
    # Run detection only every Nth frame; the temporal track gap-fill bridges the
    # skipped frames (their gap is far below temporal_max_gap_seconds). At 30fps a
    # stride of 2 halves detection cost — the dominant cost, doubly so in union
    # mode — with negligible recall impact (a face barely moves in 1 frame). The
    # blur pass still runs on every frame. 1 disables striding.
    face_detection_stride: int = 2

    # --- Hardware video acceleration (Apple VideoToolbox) ---
    # "auto" uses ffmpeg VideoToolbox hardware decode + H.264 encode when present
    # (macOS) — a large no-quality-loss speedup; falls back to software (OpenCV +
    # libx264) elsewhere (Docker/Linux/CI). "on"/"off" force it.
    use_hardware_video: str = "auto"  # auto | on | off

    # --- Orientation auto-correction ---
    # Detect "which way is up" from content (upright faces) rather than trusting
    # only the rotation metadata, which is sometimes missing or wrong — so a clip
    # uploaded upside-down is corrected. Falls back to metadata when no faces are
    # found (e.g. chest-cam clips with none in frame).
    auto_orient: bool = True
    orientation_sample_frames: int = 12
    # Minimum summed face-detection score for a candidate orientation to be
    # trusted over the metadata rotation.
    orientation_min_score: float = 1.0

    # --- Face detector strategy (privacy robustness) ---
    # "union" runs BOTH YuNet (cv2.FaceDetectorYN) and MediaPipe and takes the
    # union of detections for redundancy — a face missed by one may be caught by
    # the other. "yunet" or "mediapipe" use a single detector. Union is the
    # privacy-conservative default.
    face_detector: str = "union"  # union | yunet | mediapipe
    # YuNet candidate score threshold (its own confidence scale) — low to catch
    # weak frames; confirmation below gates what actually gets blurred.
    yunet_score_threshold: float = 0.5
    yunet_nms_threshold: float = 0.3
    # A YuNet face track is confirmed (blurred) if one detection clears this.
    # Measured separation: false positives top out ~0.67, real faces reach 0.92
    # with many detections >=0.80. 0.80 maximizes recall with a wide FP margin.
    yunet_confirm_score: float = 0.80
    # Path to the YuNet ONNX weights (downloaded by scripts/fetch_models.py).
    yunet_model_path: str = str(BACKEND_DIR / "ml_models" / "face_detection_yunet_2023mar.onnx")

    # --- Hand pose (Phase 2, MediaPipe Hands) ---
    # Frames per second to sample for hand-keypoint extraction. 10fps keeps CPU
    # cost down and is smooth enough for the dashboard overlay; recorded in the
    # output metadata so consumers know the sampling rate.
    hand_pose_sample_fps: float = 10.0
    hand_pose_max_hands: int = 2
    # Model complexity: 0 = "lite", 1 = "full". TUNED to 0 — on blurry/fast/
    # occluded egocentric footage the lite palm detector is far more permissive and
    # recovers ~3x more hands (AssemblyHands: 10.8% -> 36.3% detection) at ~no
    # accuracy cost (~22mm PA-MPJPE either way). For blur/speed, catching the hand
    # at all beats a few mm of precision. (Measured via scripts/tune_hand.py.)
    hand_pose_model_complexity: int = 0
    # Lowered to 0.1: blurry/fast hands and hands partly out of frame produce weak
    # detections; a low bar recovers them. Stray hand keypoints are harmless (not a
    # privacy issue, unlike faces), so we bias hard toward recall on messy footage.
    hand_pose_min_detection_confidence: float = 0.1
    hand_pose_min_tracking_confidence: float = 0.1
    # MediaPipe Hands reports handedness assuming a mirrored (selfie) image. A
    # chest-worn forward camera is not mirrored, so the raw Left/Right labels are
    # swapped relative to the worker's actual hands. Set True to swap them back.
    hand_pose_swap_handedness: bool = True
    # Temporal robustness for blurry/fast footage. Bridge detection dropouts up to
    # this many seconds by interpolating between the surrounding detections (a few
    # blurred frames no longer break the track). Filled frames are marked with low
    # confidence so consumers can tell measured from interpolated.
    hand_pose_gap_fill_seconds: float = 0.3
    # Light One-Euro smoothing to de-jitter noisy blurry detections WITHOUT lagging
    # fast motion (it adapts: smooths when slow, stays responsive when fast).
    hand_pose_smooth: bool = True
    # One-Euro tuning: higher min_cutoff = less low-speed smoothing (less lag).
    # Retuned for 30fps sampling on clean footage; the old 1.5/0.04 visibly
    # trailed fast hands at 30fps.
    hand_pose_smooth_min_cutoff: float = 2.0
    hand_pose_smooth_beta: float = 0.06
    # MediaPipe classifies handedness PER FRAME (no memory): a lone right hand
    # occasionally flips to "left" for a few frames. When a single detected hand
    # is spatially continuous with the OTHER side's recent track, relabel it.
    hand_pose_handedness_continuity: bool = True
    # Hand tracking backend:
    #   "mediapipe" — CPU, fast, landmark regression (hallucinates occluded
    #                 grasps; bone-consistency 0.711 on real grasp frames).
    #   "wilor"     — GPU (dedicated env), transformer mesh regression with a
    #                 MANO prior: occlusion-robust (0.994), and predicts the
    #                 hand's ABSOLUTE metric camera-frame position = measured
    #                 wrist depth. ~62 ms/frame on an RTX 5080.
    hand_pose_backend: str = "mediapipe"  # mediapipe | wilor
    # Python interpreter of the WiLoR GPU env ("" = data/tmp_wilor_env default).
    wilor_python: str = ""
    # Temporal smoother: "oneeuro" (adaptive low-pass) | "kalman" (constant-
    # velocity prediction — best for per-frame models like WiLoR) | "none".
    hand_pose_smoother: str = "oneeuro"

    # --- Security ---
    # When set, the API requires this key via the `X-API-Key` header. Always
    # enforced in production; optional in development. Generate a strong value.
    api_key: str = ""
    # Comma-separated list of allowed CORS origins (frontend).
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    # Reject uploads whose sniffed content type isn't an allowed video container.
    validate_upload_content: bool = True

    # --- Observability ---
    log_level: str = "INFO"
    # Emit structured JSON logs (production) vs human-readable (dev).
    log_json: bool = False
    # Expose Prometheus metrics at /metrics.
    metrics_enabled: bool = True

    # --- Event extraction (Phase 4) ---
    # An idle/waiting segment at or above this length is flagged as "downtime".
    idle_downtime_seconds: float = 30.0

    # --- Task segmentation (Phase 3) ---
    # Mode: "open" = describe what the person is doing in natural language +
    # commentary (general-purpose, feeds the chatbot); "taxonomy" = classify into
    # the fixed waste-services task list. Open is the default.
    segmentation_mode: str = "open"  # open | taxonomy
    # Provider: "ollama" runs a LOCAL vision model (free, no data egress);
    # "claude" uses the Anthropic API (paid, opt-in). Default is the free local
    # path so processing never costs money.
    segmentation_provider: str = "ollama"  # ollama | claude
    # Sample this many frames per second to classify.
    segmentation_sample_fps: float = 1.0
    # Merge/discard segments shorter than this (seconds) to reduce flicker.
    segmentation_min_segment_seconds: float = 1.5

    # --- Boundary detection strategy (Phase 3) ---
    # How segment *boundaries* are decided:
    #   "perframe" — classify every sampled frame with the VLM, then cut wherever
    #                adjacent labels differ (original behavior).
    #   "fused"    — find cuts from cheap dense signals (frame-diff + the Phase-2
    #                hand-pose & ego-motion products), then label each segment with
    #                the VLM ONCE. ~10x fewer VLM calls, sub-second cut precision,
    #                no flicker. Falls back to whatever signals are present.
    segmentation_boundary_mode: str = "perframe"  # perframe | fused
    # Common time-grid resolution (Hz) the fused signals are resampled onto — this
    # caps achievable cut precision. 4 Hz is a good cost/precision balance.
    boundary_sample_fps: float = 4.0
    # No two cuts closer than this (seconds); also the minimum segment length.
    boundary_min_segment_seconds: float = 2.0
    # Smoothing applied to each signal before scoring (seconds of moving average).
    boundary_smooth_seconds: float = 0.75
    # Half-width (seconds) of the before/after comparison window for the step score.
    boundary_window_seconds: float = 1.5
    # LOCAL cut threshold (late fusion): a candidate must clear k std above its
    # sliding local baseline. Higher k = fewer cuts. ~1.5-2.5 is typical.
    boundary_threshold_k: float = 1.5
    # Width (seconds) of the sliding window the local threshold is measured over.
    # Larger = more global behavior; smaller = more sensitive to local context.
    boundary_local_window_seconds: float = 4.0
    # Hard cap on segments per video (keeps the strongest cuts if exceeded).
    boundary_max_segments: int = 60
    # Per-signal toggles (a signal is also skipped if its Phase-2 product is absent).
    boundary_use_frame_diff: bool = True
    boundary_use_hand_pose: bool = True
    boundary_use_ego_motion: bool = True
    # Per-signal fusion weights (tune to trust one signal more on your footage).
    boundary_weight_frame_diff: float = 1.0
    boundary_weight_hand_pose: float = 1.0
    boundary_weight_ego_motion: float = 1.0
    # Frames sampled per segment in fused mode. With a multi-image-capable
    # provider (Ollama qwen-VL, Claude) they go in ONE call, ordered, so the
    # model reasons over what CHANGES across the segment — actions are motion,
    # not single frames ("massage the neck" vs "dry your hair").
    boundary_label_samples: int = 3
    # Local Ollama VLM. qwen2.5vl:3b gives sharper, more accurate descriptions and
    # runs on ~16GB machines; moondream is the lighter/faster fallback; qwen2.5vl:7b
    # is best but too heavy for ~16GB. Set ollama_use_json=true for capable models
    # in taxonomy mode.
    ollama_base_url: str = "http://localhost:11434"
    ollama_vlm_model: str = "qwen2.5vl:3b"
    ollama_timeout_seconds: int = 240
    # Critical for usable speed: downscale the frame sent to the VLM (fewer vision
    # tokens) and cap the context window (avoids a multi-GB KV cache). Together
    # these take qwen2.5vl:3b from >4min/frame to ~2-4s/frame on a 16GB Mac.
    segmentation_frame_max_dim: int = 512
    ollama_num_ctx: int = 4096
    # Capable models answer the structured-JSON classification prompt directly;
    # tiny models (moondream) do better with a caption prompt mapped to the
    # taxonomy locally. Set true for capable models (qwen2.5vl, llava).
    ollama_use_json: bool = False
    # Claude VLM (only used when segmentation_provider == "claude").
    claude_vlm_model: str = "claude-sonnet-4-6"

    # --- Head pose / ego-body (visual odometry) ---
    # Approximate horizontal field of view of the camera (degrees) — the
    # fallback when no calibration or device preset applies. Wrist depth divides
    # by focal length, so FOV error becomes proportional depth error.
    ego_camera_fov_deg: float = 90.0
    # Infer FOV from the recording device's metadata (iPhone/GoPro presets in
    # pipeline/intrinsics.py) when no checkerboard calibration exists. A real
    # calibration (scripts/calibrate_camera.py) always takes precedence.
    ego_camera_fov_auto: bool = True
    # The wearer's REAL palm length in cm (wrist crease to the base knuckle of
    # the middle finger; measure once with a ruler). Replaces MediaPipe's
    # generic hand-size prior in depth-from-scale, removing its ~5-10% depth
    # error, and corrects metric grasp aperture. 0 = use MediaPipe's estimate.
    wearer_palm_length_cm: float = 0.0
    ego_vo_sample_fps: float = 6.0
    ego_vo_orb_features: int = 2000

    # --- Camera mount (body-pose geometry & provenance) ---
    # Where the wearable camera is mounted. Decides the back-projection geometry
    # AND what the visual odometry actually measures:
    #   "chest" — VO measures TORSO orientation (camera rigid on the torso).
    #   "head"  — forehead-mounted; VO measures HEAD orientation. The torso only
    #             gets a yaw-only heading proxy from the head (lower confidence),
    #             and the camera rides on the neck pivot for hand back-projection.
    camera_mount: str = "chest"  # chest | head

    # --- Chest-mount camera geometry (body-pose back-projection) ---
    # Where the camera sits relative to the pelvis, in the body frame (fractions of
    # operator height H; +Y up, +Z forward). A shirt-clipped chest cam sits a bit
    # above and in front of the chest joint. Used to back-project the measured 2D
    # hand position into a 3D body-space ray (image-consistent wrist placement).
    ego_chest_mount_height_frac: float = 0.22   # above pelvis
    ego_chest_mount_forward_frac: float = 0.10  # in front of pelvis
    # Downward tilt of the mounted camera (degrees); a clipped chest cam usually
    # points slightly down toward the hands/work area.
    ego_chest_mount_pitch_deg: float = 12.0

    # --- Head (forehead) mount camera geometry ---
    # Same convention as the chest mount: rest-pose position as fractions of
    # operator height H from the pelvis. A forehead strap sits at ~0.42H, slightly
    # forward, and is usually tilted further down so the hands stay in frame.
    ego_head_mount_height_frac: float = 0.42
    ego_head_mount_forward_frac: float = 0.05
    ego_head_mount_pitch_deg: float = 20.0
    # Heads swing much faster than torsos: the torso heading proxy derived from
    # head VO is low-passed over this window (seconds) so a glance left doesn't
    # spin the whole estimated body. 0 disables smoothing.
    ego_torso_yaw_smooth_seconds: float = 1.0
    # Shoulder->wrist reach radius (fraction of H) used to resolve depth along the
    # back-projected image ray. Monocular depth is ambiguous; the 2D ray direction
    # is MEASURED, the depth along it is bounded by arm reach (documented as such).
    ego_arm_reach_frac: float = 0.30

    # --- LeRobot export (Phase 7) ---
    # "segment": one episode per annotated activity segment, with the VLM
    #            description as the episode's task string (language-conditioned
    #            VLA training: pi0/GR00T/OpenVLA consume exactly this).
    # "clip":    one episode per whole video with a single generic task label.
    # Segment mode falls back to clip mode for videos without segments.
    lerobot_episode_mode: str = "segment"  # segment | clip
    # Drop episodes whose frames almost never contain a MEASURED hand — a
    # manipulation dataset shouldn't ship hands-free episodes (walking past
    # things, staring at a screen). Fraction of frames with a measured wrist;
    # 0 disables filtering. Skips are logged, never silent.
    lerobot_min_hand_fraction: float = 0.05

    # --- Chatbot (Phase 6) ---
    # Answers questions grounded in the activity commentary + events. Local LLM by
    # default (free, no data egress). "claude" uses the Anthropic API (paid).
    chat_provider: str = "ollama"  # ollama | claude
    chat_model: str = "llama3.1:8b"  # a text LLM you already have in Ollama
    chat_max_segments: int = 400      # cap timeline lines fed to the model
    chat_temperature: float = 0.2

    # --- Anthropic API (Phases 3 & 6) ---
    anthropic_api_key: str = ""

    # ---- Derived helpers ----
    @property
    def is_production(self) -> bool:
        return self.environment.lower() in ("production", "prod")

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def auth_required(self) -> bool:
        # Enforce auth whenever a key is configured, and always in production.
        return bool(self.api_key) or self.is_production

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def anonymized_dir(self) -> Path:
        return self.data_dir / "anonymized"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"


settings = Settings()
