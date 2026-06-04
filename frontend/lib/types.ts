export type VideoStatus =
  | "uploaded"
  | "processing"
  | "anonymized"
  | "processed"
  | "failed";

export interface Video {
  id: string;
  original_filename: string;
  uploaded_at: string | null;
  operator_id: string;
  worker_id_anonymized: string | null;
  property_tag: string | null;
  operator_height_cm: number | null;
  scene: string | null;
  status: VideoStatus;
  file_size: number | null;
  duration_seconds: number | null;
  anonymized_at: string | null;
  anonymization_coverage: number | null;
  anonymization_method: string | null;
  hand_pose_extracted: boolean;
  segmented: boolean;
  segmentation_cost_usd: number | null;
  error_message: string | null;
}

export interface Segment {
  start_time: number;
  end_time: number;
  task_label: string;
  confidence: number;
  description: string;
  duration_seconds: number;
}

export interface SegmentsResponse {
  video_id: string;
  provider: string;
  model: string;
  sample_fps: number;
  frames_classified: number;
  cost_usd: number;
  duration_seconds: number;
  segments: Segment[];
}

export type Landmark = [number, number, number]; // normalized x, y, z

export interface HandPoseFrame {
  frame_number: number;
  timestamp_ms: number;
  left_hand_landmarks: Landmark[] | null;
  right_hand_landmarks: Landmark[] | null;
  left_confidence: number | null;
  right_confidence: number | null;
}

export interface HandPoseResponse {
  video_id: string;
  metadata: Record<string, string>;
  frames: HandPoseFrame[];
}

export interface HeadPoseFrameT {
  timestamp_ms: number;
  position: [number, number, number];
  quaternion: [number, number, number, number]; // w, x, y, z
  tracked: boolean;
}

export interface HeadPoseResponse {
  video_id: string;
  method: string;
  frames: HeadPoseFrameT[];
}

export interface OpEvent {
  id: number;
  video_id: string;
  type: string;
  label: string;
  start_time: number;
  end_time: number;
  duration_seconds: number;
  property_tag: string | null;
  description: string | null;
}

export interface VideoSummary {
  video_id: string;
  event_count: number;
  total_event_seconds: number;
  time_per_task: Record<string, number>;
  time_per_type: Record<string, number>;
  service_event_count: number;
  idle_seconds: number;
  downtime_seconds: number;
  downtime_event_count: number;
  contamination_event_count: number;
  active_hand_seconds: number | null;
  events: OpEvent[];
}

export interface PropertyStats {
  service_events: number;
  downtime_seconds: number;
  contamination_flags: number;
  total_seconds: number;
}

export interface Overview {
  operator_id: string;
  video_count: number;
  total_hours: number;
  total_service_events: number;
  total_downtime_seconds: number;
  total_contamination_flags: number;
  per_property: Record<string, PropertyStats>;
}
