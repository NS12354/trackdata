"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { Panel, StatusBadge } from "@/components/ui";

// NOTE: 3D rig panel removed for now (kept in repo: HumanModel3D.tsx procedural,
// HumanModelFBX.tsx Mixamo Y Bot WIP). Re-add when pose accuracy is validated.
import { fmtDuration, fmtPct, fmtDate } from "@/lib/format";
import { exportBundleUrl } from "@/lib/api";
import HandPoseOverlay from "@/components/HandPoseOverlay";
import SegmentTimeline from "@/components/SegmentTimeline";
import EventMetrics from "@/components/EventMetrics";
import Pose3D from "@/components/Pose3D";
import DewarpControls from "@/components/DewarpControls";
import type {
  Video, SegmentsResponse, HandPoseResponse, HeadPoseResponse, VideoSummary,
} from "@/lib/types";

/** seconds -> HH:MM:SS.mmm */
function fmtClock(s?: number): string {
  if (s == null || isNaN(s)) return "—";
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60), ms = Math.floor((s % 1) * 1000);
  return `${pad(h)}:${pad(m)}:${pad(sec)}.${pad(ms, 3)}`;
}

export default function VideoDetail({
  video,
  segments,
  handpose,
  headpose,
  summary,
  videoUrl,
}: {
  video: Video;
  segments: SegmentsResponse | null;
  handpose: HandPoseResponse | null;
  headpose: HeadPoseResponse | null;
  summary: VideoSummary | null;
  videoUrl: string;
}) {
  const router = useRouter();
  // Live copy of the video row: polled while it's still processing so the
  // anonymization percentage animates without a manual refresh.
  const [vid, setVid] = useState(video);
  useEffect(() => setVid(video), [video]);
  useEffect(() => {
    if (vid.status !== "uploaded" && vid.status !== "processing") return;
    let alive = true;
    const tick = async () => {
      try {
        const next = await api.getVideo(video.id);
        if (!alive) return;
        setVid(next);
        // Once it leaves the processing states, reload to fetch the now-ready
        // anonymized video + pose/segments (server-rendered on the page).
        if (next.status !== "uploaded" && next.status !== "processing") router.refresh();
      } catch {
        /* transient — keep polling */
      }
    };
    const h = setInterval(tick, 1500);
    return () => {
      alive = false;
      clearInterval(h);
    };
  }, [vid.status, video.id, router]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(video.duration_seconds || 0);
  const [overlay, setOverlay] = useState(true);
  const [rot, setRot] = useState(0); // viewer rotation: 0 / 90 / 180 / 270

  // Display size of the video (computed from intrinsic dims + the 60vh cap), so
  // we can resize the frame box when rotated 90°/270° and keep it from clipping.
  const [box, setBox] = useState({ w: 0, h: 0 });
  const measure = () => {
    const v = videoRef.current;
    if (!v || !v.videoWidth || !v.videoHeight) return;
    const h = Math.min(window.innerHeight * 0.6, v.videoHeight);
    setBox({ w: h * (v.videoWidth / v.videoHeight), h });
  };
  useEffect(() => {
    const onResize = () => measure();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const rotated = rot % 180 !== 0;
  const outerW = rotated ? box.h : box.w;
  const outerH = rotated ? box.w : box.h;

  const seek = (t: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = t;
    v.play().catch(() => {});
  };

  const [deleting, setDeleting] = useState(false);
  const isProcessing = vid.status === "uploaded" || vid.status === "processing";
  const onDelete = async () => {
    const msg = isProcessing
      ? "Cancel processing and delete this video? The upload and all derived data will be removed."
      : "Delete this video and all its data? This cannot be undone.";
    if (!window.confirm(msg)) return;
    setDeleting(true);
    try {
      await api.deleteVideo(video.id);
      router.push("/videos");
      router.refresh();
    } catch (e) {
      setDeleting(false);
      window.alert("Delete failed: " + (e as Error).message);
    }
  };

  const hasHands = !!handpose && handpose.frames.length > 0;
  const ready = vid.status === "processed" || vid.status === "anonymized";
  const operatorHeight = video.operator_height_cm;
  const scene = video.scene;
  const activeSeg =
    segments?.segments.find((s) => currentTime >= s.start_time && currentTime < s.end_time) ||
    segments?.segments[0];

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-3">
        <Link href="/videos" className="text-sm text-muted hover:text-text">
          ← Videos
        </Link>
        <h1 className="text-lg font-semibold">{video.original_filename}</h1>
        <StatusBadge status={vid.status} />
        <div className="ml-auto flex items-center gap-3 text-xs text-muted">
          {video.property_tag && <span>{video.property_tag}</span>}
          <span>{fmtDate(video.uploaded_at)}</span>
          <span>{fmtDuration(video.duration_seconds)}</span>
          {ready && (
            <a
              href={exportBundleUrl(video.id)}
              className="rounded border border-border bg-panel2 px-3 py-1.5 text-xs font-medium text-text hover:border-accent hover:text-accent"
            >
              ↓ Export bundle
            </a>
          )}
          <button
            type="button"
            onClick={onDelete}
            disabled={deleting}
            className="rounded border border-danger/40 bg-danger/10 px-3 py-1.5 text-xs font-medium text-danger hover:bg-danger/20 disabled:opacity-50"
          >
            {deleting ? "Deleting…" : isProcessing ? "✕ Cancel processing" : "🗑 Delete"}
          </button>
        </div>
      </div>

      {video.error_message && (
        <Panel className="border-danger/40 bg-danger/10 p-3 text-sm text-danger">
          {video.error_message}
        </Panel>
      )}

      {/* Egocentric-pose layout: footage + commentary (left), 3D pose (right) */}
      <div className="grid gap-5 lg:grid-cols-[1.5fr_1fr]">
        <div className="space-y-3">
          <Panel className="overflow-hidden">
            <div
              className="relative mx-auto bg-black"
              style={box.w ? { width: outerW, height: outerH } : undefined}
            >
              {ready ? (
                // The video + pose overlay rotate together as one unit so they stay aligned.
                <div className="absolute inset-0 flex items-center justify-center overflow-hidden">
                  <div
                    className="relative shrink-0 transition-transform duration-200"
                    style={{ transform: `rotate(${rot}deg)` }}
                  >
                    <video
                      ref={videoRef}
                      src={videoUrl}
                      controls
                      playsInline
                      onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
                      onLoadedMetadata={(e) => {
                        setDuration(e.currentTarget.duration);
                        measure();
                      }}
                      onLoadedData={measure}
                      className="block max-h-[60vh] w-auto"
                    />
                    {hasHands && (
                      <HandPoseOverlay videoRef={videoRef} frames={handpose!.frames} enabled={overlay} />
                    )}
                  </div>
                </div>
              ) : (
                <div className="flex h-64 w-[480px] max-w-full flex-col items-center justify-center gap-3 px-8 text-sm text-muted">
                  {vid.status === "failed" ? (
                    <span className="text-danger">Processing failed.</span>
                  ) : (
                    (() => {
                      const stage = vid.processing_stage || "queued";
                      const isAnon = stage === "anonymizing";
                      // Only the anonymization stage reports a true fraction; later
                      // stages show an indeterminate (animated) bar.
                      const pct = isAnon
                        ? Math.round((vid.processing_progress || 0) * 100)
                        : null;
                      const label =
                        stage === "anonymizing"
                          ? "Anonymizing (blurring faces)"
                          : stage === "hand pose"
                          ? "Extracting hand pose"
                          : stage === "segmenting"
                          ? "Segmenting tasks"
                          : "Queued for processing";
                      return (
                        <div className="w-full max-w-[360px]">
                          <div className="mb-1.5 flex items-baseline justify-between">
                            <span className="font-medium text-text">{label}…</span>
                            {pct !== null && (
                              <span className="font-mono text-lg font-semibold tabular-nums text-accent">
                                {pct}%
                              </span>
                            )}
                          </div>
                          <div className="h-2.5 w-full overflow-hidden rounded bg-panel2">
                            {pct !== null ? (
                              <div
                                className="h-full rounded bg-accent transition-all duration-300"
                                style={{ width: `${pct}%` }}
                              />
                            ) : (
                              <div className="h-full w-1/3 animate-pulse rounded bg-accent" />
                            )}
                          </div>
                          <p className="mt-2 text-xs text-muted">
                            This runs automatically — anonymize → hand pose → segment.
                          </p>
                        </div>
                      );
                    })()
                  )}
                </div>
              )}
              {/* Active-skill overlay card */}
              {ready && activeSeg && (
                <div className="absolute left-3 top-3 max-w-[62%] rounded-lg bg-black/65 px-3 py-2 shadow-lg backdrop-blur-sm">
                  <div className="text-[13px] font-semibold text-white">
                    Skill: <span className="capitalize">{activeSeg.task_label || "activity"}</span>
                  </div>
                  {activeSeg.description && (
                    <div className="mt-0.5 text-xs leading-snug text-slate-200">{activeSeg.description}</div>
                  )}
                  <div className="mt-1 font-mono text-[10px] text-slate-400">
                    {fmtClock(activeSeg.start_time)} → {fmtClock(activeSeg.end_time)}
                  </div>
                </div>
              )}
            </div>
            {/* Live commentary caption */}
            {activeSeg?.description && (
              <div className="border-t border-border px-4 py-3 text-center text-sm">
                {activeSeg.description}
              </div>
            )}
            {/* Environment + scene + operator height */}
            <div className="flex gap-10 border-t border-border px-4 py-3 text-sm">
              <div>
                <div className="text-[11px] uppercase tracking-wide text-muted">⌖ Environment</div>
                <div className="mt-0.5">{video.property_tag || "—"}</div>
              </div>
              <div>
                <div className="text-[11px] uppercase tracking-wide text-muted">⬚ Scene</div>
                <div className="mt-0.5 capitalize">{scene || "—"}</div>
              </div>
              <div>
                <div className="text-[11px] uppercase tracking-wide text-muted">⬓ Operator height</div>
                <div className="mt-0.5">{operatorHeight ? `${operatorHeight}cm` : "—"}</div>
              </div>
            </div>
          </Panel>

          <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
            <span>Anonymized · blur {fmtPct(video.anonymization_coverage)}</span>
            {hasHands && (
              <label className="flex cursor-pointer items-center gap-2">
                <input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} />
                Hand-pose overlay
              </label>
            )}
            {ready && (
              <div className="flex items-center gap-1">
                <span className="mr-1">Rotate</span>
                <button
                  type="button"
                  title="Rotate left 90°"
                  onClick={() => setRot((r) => (r + 270) % 360)}
                  className="rounded border border-border bg-panel2 px-2 py-1 text-text hover:border-accent hover:text-accent"
                >
                  ⟲
                </button>
                <button
                  type="button"
                  title="Rotate right 90°"
                  onClick={() => setRot((r) => (r + 90) % 360)}
                  className="rounded border border-border bg-panel2 px-2 py-1 text-text hover:border-accent hover:text-accent"
                >
                  ⟳
                </button>
                {rot !== 0 && (
                  <button
                    type="button"
                    onClick={() => setRot(0)}
                    className="rounded border border-border bg-panel2 px-2 py-1 text-text hover:border-accent hover:text-accent"
                  >
                    Reset · {rot}°
                  </button>
                )}
              </div>
            )}
          </div>

          {ready && <DewarpControls videoId={video.id} />}
        </div>

        {/* 3D pose, synced to the playhead */}
        <Pose3D
          videoRef={videoRef}
          frames={handpose?.frames ?? []}
          headFrames={headpose?.frames ?? []}
          operatorHeightCm={operatorHeight || 170}
        />
      </div>

      {/* Timeline + metrics */}
      <div className="grid gap-5 lg:grid-cols-[1fr_1fr]">
        <Panel className="p-4">
          <div className="mb-3 flex items-center justify-between">
            <div className="text-sm font-medium">Task timeline</div>
            {segments && (
              <div className="text-xs text-muted">
                {segments.provider}/{segments.model} · ${segments.cost_usd.toFixed(2)}
              </div>
            )}
          </div>
          {segments && segments.segments.length > 0 ? (
            <SegmentTimeline
              segments={segments.segments}
              duration={duration}
              currentTime={currentTime}
              onSeek={seek}
            />
          ) : (
            <div className="py-8 text-center text-sm text-muted">No segments yet.</div>
          )}
        </Panel>
        {summary && <EventMetrics summary={summary} />}
      </div>
    </div>
  );
}
