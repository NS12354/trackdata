import type {
  Video,
  SegmentsResponse,
  HandPoseResponse,
  VideoSummary,
  Overview,
} from "./types";

const BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";

function authHeaders(): Record<string, string> {
  // ngrok-skip-browser-warning bypasses the free-tier interstitial page.
  const h: Record<string, string> = { "ngrok-skip-browser-warning": "true" };
  if (API_KEY) h["X-API-Key"] = API_KEY;
  return h;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: authHeaders(),
    // Always fetch fresh status/metrics.
    cache: "no-store",
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`GET ${path} -> ${res.status} ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listVideos: () => get<Video[]>("/api/videos"),
  getVideo: (id: string) => get<Video>(`/api/videos/${id}`),
  getSegments: (id: string) => get<SegmentsResponse>(`/api/videos/${id}/segments`),
  getHandPose: (id: string) => get<HandPoseResponse>(`/api/videos/${id}/hand-pose`),
  getEvents: (id: string) => get<VideoSummary>(`/api/videos/${id}/events`),
  getOverview: () => get<Overview>("/api/metrics/overview"),
};

export interface ChatReply {
  answer: string;
  provider: string;
  model: string;
}

export async function chat(question: string, videoId?: string): Promise<ChatReply> {
  const res = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ question, video_id: videoId || null }),
  });
  if (!res.ok) {
    throw new Error(`${res.status}: ${(await res.text().catch(() => "")) || "chat failed"}`);
  }
  return res.json() as Promise<ChatReply>;
}

/** Direct media URL for the <video> element (auth via query param). */
export function anonymizedVideoUrl(id: string): string {
  const q = API_KEY ? `?api_key=${encodeURIComponent(API_KEY)}` : "";
  return `${BASE}/api/videos/${id}/anonymized${q}`;
}

/** Export bundle (.zip) download URL (auth via query param for <a download>). */
export function exportBundleUrl(id: string): string {
  const q = API_KEY ? `?api_key=${encodeURIComponent(API_KEY)}` : "";
  return `${BASE}/api/videos/${id}/export${q}`;
}

export interface UploadFields {
  operator_id?: string;
  property_tag?: string;
  worker_id_anonymized?: string;
  operator_height_cm?: string;
}

/**
 * Upload a video straight to the backend (multipart) with progress.
 * Uses XHR for upload-progress events. The browser talks to the backend
 * directly (not through Next.js), so the backend's CORS must allow this origin.
 */
export function uploadVideo(
  file: File,
  fields: UploadFields,
  onProgress?: (pct: number) => void
): Promise<{ id: string }> {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append("file", file);
    if (fields.operator_id) fd.append("operator_id", fields.operator_id);
    if (fields.property_tag) fd.append("property_tag", fields.property_tag);
    if (fields.worker_id_anonymized)
      fd.append("worker_id_anonymized", fields.worker_id_anonymized);
    if (fields.operator_height_cm)
      fd.append("operator_height_cm", fields.operator_height_cm);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/api/videos`);
    xhr.setRequestHeader("ngrok-skip-browser-warning", "true");
    if (API_KEY) xhr.setRequestHeader("X-API-Key", API_KEY);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress)
        onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new Error("upload succeeded but response was not JSON"));
        }
      } else {
        reject(new Error(`${xhr.status}: ${xhr.responseText || "upload failed"}`));
      }
    };
    xhr.onerror = () =>
      reject(new Error("network error — is the backend reachable and is this origin allowed by CORS?"));
    xhr.send(fd);
  });
}

export { BASE as API_BASE };
