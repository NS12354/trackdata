"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { Panel, StatusBadge } from "@/components/ui";

// NOTE: 3D rig panel removed for now (kept in repo: HumanModel3D.tsx procedural,
// HumanModelFBX.tsx Mixamo Y Bot WIP). Re-add when pose accuracy is validated.
import { fmtClock, fmtDuration, fmtPct, fmtDate, labelColor } from "@/lib/format";
import { exportBundleUrl, patchSegments } from "@/lib/api";
import { graspAperture, graspApertureMeters, nearestByTime } from "@/lib/pose";
import HandPoseOverlay from "@/components/HandPoseOverlay";
import SegmentTimeline from "@/components/SegmentTimeline";
import EventMetrics from "@/components/EventMetrics";
import Pose3D from "@/components/Pose3D";
import type {
  Video, SegmentsResponse, HandPoseResponse, HandPoseFrame, HeadPoseResponse,
  Landmark, VideoSummary,
} from "@/lib/types";

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
  const videoRef = useRef<HTMLVideoElement>(null);
  const router = useRouter();
  // Live processing tracker: poll the clip's row while the pipeline runs and
  // reload the page data the moment it completes.
  const [live, setLive] = useState(video);
  const [prog, setProg] = useState<{ stage?: string; pct?: number | null; detail?: string }>({});
  useEffect(() => {
    if (live.status === "processed" || live.status === "failed") return;
    const iv = setInterval(async () => {
      try {
        const [v, p] = await Promise.all([
          api.getVideo(video.id),
          api.getProgress(video.id).catch(() => ({})),
        ]);
        setLive(v);
        setProg(p || {});
        if (v.status === "processed" || v.status === "failed") {
          clearInterval(iv);
          router.refresh(); // pull segments/hand data into the page
        }
      } catch {
        /* backend busy mid-stage; try again next tick */
      }
    }, 3000);
    return () => clearInterval(iv);
  }, [video.id, live.status, router]);
  const fallbackStage =
    !live.anonymized_at && (live.status === "uploaded" || live.status === "processing")
      ? "ingesting video"
    : !live.hand_pose_extracted ? "hand tracking"
    : !live.segmented ? "annotating activities"
    : "deriving metrics";
  const progFresh =
    (prog as any).ts != null && Date.now() / 1000 - (prog as any).ts < 45;
  const stage =
    live.status === "processed" || live.status === "failed"
      ? null
      : !progFresh && live.status === "uploaded"
        ? "queued — waiting for worker"
      : progFresh
        ? `${prog.stage || fallbackStage}` +
          (prog.pct != null ? ` ${Math.round(prog.pct)}%` : "") +
          (prog.detail ? ` — ${prog.detail}` : "")
        : fallbackStage;
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(video.duration_seconds || 0);
  const [overlay, setOverlay] = useState(true);
  const [segs, setSegs] = useState(segments?.segments ?? []);
  const [editIdx, setEditIdx] = useState<number | null>(null);
  const [editMode, setEditMode] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);

  // Boundary drags STAGE locally; Save commits them all in one PATCH, Cancel
  // discards — like an editing app, no accidental writes from a stray drag.
  const [pendingSegs, setPendingSegs] = useState<typeof segs | null>(null);
  const [pendingEdits, setPendingEdits] = useState<object[]>([]);

  const moveBoundary = (i: number, t: number) => {
    const base = pendingSegs ?? segs;
    setPendingEdits((e) => [...e, {
      op: "move_boundary", segment_index: i,
      field: "boundary", before: base[i].end_time, after: t,
    }]);
    setPendingSegs(base.map((s, j) => {
      if (j === i) return { ...s, end_time: t, human_verified: true };
      if (j === i + 1) return { ...s, start_time: t, human_verified: true };
      return s;
    }));
  };

  const commitBoundaries = async () => {
    if (!pendingSegs) return;
    try {
      const doc = await patchSegments(video.id, pendingSegs, pendingEdits);
      setSegs(doc.segments);
      setPendingSegs(null);
      setPendingEdits([]);
      setSaveMsg(`${pendingEdits.length} cut(s) saved ✓ (logged; events re-derived)`);
      setTimeout(() => setSaveMsg(null), 3500);
    } catch (e) {
      setSaveMsg(`save failed: ${(e as Error).message}`);
    }
  };

  const cancelBoundaries = () => {
    setPendingSegs(null);
    setPendingEdits([]);
    setSaveMsg("changes discarded");
    setTimeout(() => setSaveMsg(null), 2000);
  };

  const saveSegment = async (idx: number, upd: Partial<(typeof segs)[number]>) => {
    const base = pendingSegs ?? segs;  // include any staged cut drags
    const before = base[idx];
    const next = base.map((s, i) => (i === idx ? { ...s, ...upd, human_verified: true } : s));
    const edits = [
      ...pendingEdits,
      ...Object.entries(upd)
        .filter(([k, v]) => (before as any)[k] !== v)
        .map(([field, after]) => ({
          op: "update", segment_index: idx, field,
          before: (before as any)[field], after,
        })),
    ];
    try {
      const doc = await patchSegments(video.id, next, edits);
      setSegs(doc.segments);
      setPendingSegs(null);
      setPendingEdits([]);
      setEditIdx(null);
      setSaveMsg("saved ✓ (logged to audit trail; events re-derived)");
      setTimeout(() => setSaveMsg(null), 3500);
    } catch (e) {
      setSaveMsg(`save failed: ${(e as Error).message}`);
    }
  };

  const seek = (t: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = t;
    v.play().catch(() => {});
  };

  const hasHands = !!handpose && handpose.frames.length > 0;
  const handCoverage = hasHands
    ? handpose!.frames.filter((f) => f.left_hand_landmarks || f.right_hand_landmarks).length /
      handpose!.frames.length
    : null;
  const ready = live.status === "processed" || live.status === "anonymized";
  const operatorHeight = video.operator_height_cm;
  const scene = video.scene;
  const activeSeg =
    segs.find((s) => currentTime >= s.start_time && currentTime < s.end_time) || segs[0];

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-3">
        <Link href="/videos" className="text-sm text-muted hover:text-text">
          ← Videos
        </Link>
        <h1 className="text-lg font-semibold">{video.original_filename}</h1>
        <StatusBadge status={live.status} />
        {stage && (
          <span className="flex items-center gap-1.5 rounded-full border border-warn/50 px-2.5 py-0.5 text-xs text-warn">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-warn" />
            {stage}
          </span>
        )}
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
                <div className="text-[11px] uppercase tracking-wide text-muted">⬓ Wearer height</div>
                <div className="mt-0.5">{operatorHeight ? `${operatorHeight}cm` : "—"}</div>
              </div>
            </div>
          </Panel>

          <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
            <span>
              {video.anonymization_method?.startsWith("none")
                ? "Privacy blur off (no bystanders)"
                : `Anonymized · blur ${fmtPct(video.anonymization_coverage)}`}
            </span>
            {hasHands && (
              <label className="flex cursor-pointer items-center gap-2">
                <input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} />
                Hand-pose overlay
              </label>
            )}
            {hasHands && (
              <GraspChips frames={handpose!.frames} currentTime={currentTime} />
            )}
          </div>
        </div>

        {/* 3D pose, synced to the playhead */}
        <Pose3D
          videoRef={videoRef}
          frames={handpose?.frames ?? []}
          headFrames={headpose?.frames ?? []}
          operatorHeightCm={operatorHeight || 170}
        />
      </div>

      {/* Timeline (full width) */}
      <Panel className="p-4">
        <div className="mb-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="text-sm font-medium">Activity timeline</div>
            <button
              onClick={() => {
                if (editMode && pendingSegs) cancelBoundaries();
                setEditMode(!editMode);
              }}
              className={`rounded border px-2 py-0.5 text-xs ${
                editMode
                  ? "border-warn/70 text-warn"
                  : "border-border text-muted hover:border-accent hover:text-accent"
              }`}
            >
              {editMode ? "✂ editing — drag cuts on the bar" : "✂ edit cuts"}
            </button>
            {pendingSegs && (
              <>
                <button
                  onClick={commitBoundaries}
                  className="rounded bg-accent px-3 py-0.5 text-xs font-medium text-white"
                >
                  Save {pendingEdits.length} cut{pendingEdits.length > 1 ? "s" : ""} ✓
                </button>
                <button
                  onClick={cancelBoundaries}
                  className="rounded border border-border px-2 py-0.5 text-xs text-muted hover:border-danger hover:text-danger"
                >
                  Cancel
                </button>
              </>
            )}
          </div>
          {segments && (
            <div className="text-xs text-muted">
              annotated by {segments.provider}/{segments.model} · ${segments.cost_usd.toFixed(2)}
            </div>
          )}
        </div>
        {segs.length > 0 ? (
          <>
            <SegmentTimeline
              segments={pendingSegs ?? segs}
              duration={duration}
              currentTime={currentTime}
              onSeek={seek}
              editable={editMode}
              onMoveBoundary={moveBoundary}
              onScrub={(t) => {
                const v = videoRef.current;
                if (v) { v.pause(); v.currentTime = t; }
              }}
            />
            <ul className="mt-3 max-h-72 space-y-1 overflow-auto pr-1">
              {(pendingSegs ?? segs).map((s, i) => {
                const active = currentTime >= s.start_time && currentTime < s.end_time;
                return (
                  <li key={i} className={`rounded px-2 py-1.5 text-sm ${active ? "bg-panel2 ring-1 ring-border" : ""}`}>
                    {editIdx === i ? (
                      <SegmentEditor seg={s} onSave={(u) => saveSegment(i, u)}
                                     onCancel={() => setEditIdx(null)} />
                    ) : (
                      <div className="flex items-start gap-2">
                        <button onClick={() => seek(s.start_time)}
                                className="flex min-w-0 flex-1 items-start gap-2 text-left hover:opacity-80">
                          <span className="mt-1 h-2.5 w-2.5 shrink-0 rounded-sm"
                                style={{ backgroundColor: labelColor(s.task_label) }} />
                          <span className="w-24 shrink-0 tabular-nums text-xs text-muted">
                            {fmtClock(s.start_time)}–{fmtClock(s.end_time)}
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className="block">
                              {s.task_label}
                              {s.human_verified && (
                                <span className="ml-1 text-ok" title="human verified">✓</span>
                              )}
                            </span>
                            {s.description && (
                              <span className="block truncate text-xs text-muted">{s.description}</span>
                            )}
                          </span>
                        </button>
                        <button onClick={() => setEditIdx(i)}
                                className="shrink-0 rounded border border-border px-2 py-0.5 text-xs text-muted hover:border-accent hover:text-accent">
                          edit
                        </button>
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
            {saveMsg && <div className="mt-2 text-xs text-muted">{saveMsg}</div>}
          </>
        ) : (
          <div className="py-8 text-center text-sm text-muted">No segments yet.</div>
        )}
      </Panel>

      {/* Metrics (full width) */}
      {summary && <EventMetrics summary={summary} handCoverage={handCoverage} />}
    </div>
  );
}

/** Inline correction form: fix the label/description/boundaries the model got
 * wrong. Saving marks the segment human-verified and logs the edit. */
function SegmentEditor({
  seg,
  onSave,
  onCancel,
}: {
  seg: { start_time: number; end_time: number; task_label: string; description: string };
  onSave: (u: { start_time: number; end_time: number; task_label: string; description: string }) => void;
  onCancel: () => void;
}) {
  const [label, setLabel] = useState(seg.task_label);
  const [desc, setDesc] = useState(seg.description);
  const [start, setStart] = useState(String(seg.start_time));
  const [end, setEnd] = useState(String(seg.end_time));
  const cls = "rounded border border-border bg-panel2 px-2 py-1 text-sm outline-none focus:border-accent";
  return (
    <div className="flex flex-wrap items-center gap-2 py-1">
      <input className={`${cls} w-20 tabular-nums`} value={start} onChange={(e) => setStart(e.target.value)} title="start (s)" />
      <input className={`${cls} w-20 tabular-nums`} value={end} onChange={(e) => setEnd(e.target.value)} title="end (s)" />
      <input className={`${cls} w-44`} value={label} onChange={(e) => setLabel(e.target.value)} placeholder="task label" />
      <input className={`${cls} min-w-0 flex-1`} value={desc} onChange={(e) => setDesc(e.target.value)} placeholder="imperative description, e.g. 'pick up the towel roll'" />
      <button
        onClick={() => onSave({ start_time: parseFloat(start), end_time: parseFloat(end), task_label: label, description: desc })}
        className="rounded bg-accent px-3 py-1 text-xs font-medium text-white"
      >
        Save ✓
      </button>
      <button onClick={onCancel} className="rounded border border-border px-2 py-1 text-xs text-muted">
        cancel
      </button>
    </div>
  );
}

/** Live gripper-channel readout synced to the playhead — the same aperture
 * signal exported as the robot action's gripper dimension. */
function GraspChips({ frames, currentTime }: { frames: HandPoseFrame[]; currentTime: number }) {
  const times = useMemo(() => frames.map((f) => f.timestamp_ms), [frames]);
  const fr = nearestByTime(frames, times, currentTime * 1000);
  if (!fr) return null;
  return (
    <span className="flex items-center gap-1.5">
      <span className="text-muted">Grasp:</span>
      <GraspChip side="L" hand={fr.left_hand_landmarks} world={fr.left_world_landmarks} />
      <GraspChip side="R" hand={fr.right_hand_landmarks} world={fr.right_world_landmarks} />
    </span>
  );
}

function GraspChip({
  side,
  hand,
  world,
}: {
  side: string;
  hand: Landmark[] | null;
  world?: Landmark[] | null;
}) {
  const a = graspAperture(hand);
  if (a == null) {
    return (
      <span className="rounded-full border border-border px-2 py-0.5 text-[11px] text-muted/70">
        {side} not in view
      </span>
    );
  }
  const meters = graspApertureMeters(world);
  const closed = a <= 0.3;
  return (
    <span
      className={`rounded-full border px-2 py-0.5 text-[11px] tabular-nums ${
        closed ? "border-warn/60 text-warn" : "border-ok/60 text-ok"
      }`}
    >
      {side} {closed ? "✊ closed" : "✋ open"}{" "}
      {meters != null ? `${(meters * 100).toFixed(1)}cm` : a.toFixed(2)}
    </span>
  );
}
