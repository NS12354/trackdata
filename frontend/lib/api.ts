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

function authHeaders(): HeadersInit {
  return API_KEY ? { "X-API-Key": API_KEY } : {};
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

export { BASE as API_BASE };
