"use client";

import { useRef, useState } from "react";
import Link from "next/link";
import { Panel, StatusBadge } from "@/components/ui";
import { fmtDuration, fmtPct, fmtDate } from "@/lib/format";
import { exportBundleUrl } from "@/lib/api";
import HandPoseOverlay from "@/components/HandPoseOverlay";
import SegmentTimeline from "@/components/SegmentTimeline";
import EventMetrics from "@/components/EventMetrics";
import Pose3D from "@/components/Pose3D";
import type { Video, SegmentsResponse, HandPoseResponse, VideoSummary } from "@/lib/types";

export default function VideoDetail({
  video,
  segments,
  handpose,
  summary,
  videoUrl,
}: {
  video: Video;
  segments: SegmentsResponse | null;
  handpose: HandPoseResponse | null;
  summary: VideoSummary | null;
  videoUrl: string;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(video.duration_seconds || 0);
  const [overlay, setOverlay] = useState(true);

  const seek = (t: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = t;
    v.play().catch(() => {});
  };

  const hasHands = !!handpose && handpose.frames.length > 0;
  const ready = video.status === "processed" || video.status === "anonymized";
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
        <StatusBadge status={video.status} />
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
            <div className="relative mx-auto w-fit bg-black">
              {ready ? (
                <video
                  ref={videoRef}
                  src={videoUrl}
                  controls
                  playsInline
                  onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
                  onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
                  className="block max-h-[60vh] w-auto"
                />
              ) : (
                <div className="flex h-64 w-[480px] max-w-full items-center justify-center text-sm text-muted">
                  Anonymized video not ready yet.
                </div>
              )}
              {hasHands && (
                <HandPoseOverlay videoRef={videoRef} frames={handpose!.frames} enabled={overlay} />
              )}
            </div>
            {/* Live commentary caption */}
            {activeSeg?.description && (
              <div className="border-t border-border px-4 py-3 text-center text-sm">
                {activeSeg.description}
              </div>
            )}
            {/* Scene + operator height */}
            <div className="flex gap-10 border-t border-border px-4 py-3 text-sm">
              <div>
                <div className="text-[11px] uppercase tracking-wide text-muted">⬚ Scene</div>
                <div className="mt-0.5">{scene || "—"}</div>
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
          </div>
        </div>

        {/* 3D pose, synced to the playhead */}
        <Pose3D
          videoRef={videoRef}
          frames={handpose?.frames ?? []}
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
