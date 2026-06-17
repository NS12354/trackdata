"use client";

import { useRef, useState } from "react";
import { fmtClock, labelColor } from "@/lib/format";
import type { Segment } from "@/lib/types";

const MIN_SEG_S = 0.3; // a segment can't be trimmed shorter than this

export default function SegmentTimeline({
  segments,
  duration,
  currentTime,
  onSeek,
  editable = false,
  onMoveBoundary,
  onScrub,
}: {
  segments: Segment[];
  duration: number;
  currentTime: number;
  onSeek: (t: number) => void;
  /** Edit mode: show draggable trim handles at each interior boundary. */
  editable?: boolean;
  /** Commit a boundary move: segments[i].end_time = t = segments[i+1].start_time. */
  onMoveBoundary?: (i: number, t: number) => void;
  /** Frame-scrub during drag (seek without playing); falls back to onSeek. */
  onScrub?: (t: number) => void;
}) {
  const total = duration || (segments.length ? segments[segments.length - 1].end_time : 1);
  const barRef = useRef<HTMLDivElement>(null);
  // Live preview while dragging handle i (seconds), null when not dragging.
  const [drag, setDrag] = useState<{ i: number; t: number } | null>(null);

  const shown = segments.map((s) => ({ ...s }));
  if (drag) {
    shown[drag.i].end_time = drag.t;
    shown[drag.i + 1].start_time = drag.t;
  }
  const activeIdx = shown.findIndex(
    (s) => currentTime >= s.start_time && currentTime < s.end_time
  );

  const timeFromClientX = (clientX: number) => {
    const r = barRef.current!.getBoundingClientRect();
    return Math.min(total, Math.max(0, ((clientX - r.left) / r.width) * total));
  };

  const startDrag = (i: number) => (e: React.PointerEvent) => {
    e.preventDefault();
    e.stopPropagation();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    const lo = segments[i].start_time + MIN_SEG_S;
    const hi = segments[i + 1].end_time - MIN_SEG_S;
    const move = (ev: PointerEvent) => {
      const t = Math.min(hi, Math.max(lo, timeFromClientX(ev.clientX)));
      setDrag({ i, t });
      (onScrub ?? onSeek)(t); // show the cut frame, editing-software style
    };
    const up = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      const t = Math.min(hi, Math.max(lo, timeFromClientX(ev.clientX)));
      setDrag(null);
      onMoveBoundary?.(i, Math.round(t * 1000) / 1000);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  return (
    <div className="space-y-3">
      {/* Proportional bar */}
      <div
        ref={barRef}
        className={`relative w-full overflow-visible rounded border border-border bg-panel2 ${
          editable ? "h-9" : "h-7"
        }`}
        role="group"
        aria-label="activity timeline"
      >
        {shown.map((s, i) => {
          const left = (s.start_time / total) * 100;
          const width = ((s.end_time - s.start_time) / total) * 100;
          return (
            <button
              key={i}
              onClick={() => onSeek(s.start_time)}
              title={`${s.task_label} (${fmtClock(s.start_time)}–${fmtClock(s.end_time)})`}
              className="absolute top-0 h-full border-r border-black/30 transition-opacity hover:opacity-80"
              style={{
                left: `${left}%`,
                width: `${width}%`,
                backgroundColor: labelColor(s.task_label),
                opacity: activeIdx === i ? 1 : 0.55,
              }}
            />
          );
        })}
        {/* Trim handles at interior boundaries (edit mode) */}
        {editable &&
          shown.slice(0, -1).map((s, i) => (
            <div
              key={`h${i}`}
              onPointerDown={startDrag(i)}
              title={`drag to move cut (${fmtClock(s.end_time)})`}
              className="absolute top-[-4px] z-10 flex h-[calc(100%+8px)] w-3 -translate-x-1/2 cursor-ew-resize items-center justify-center"
              style={{ left: `${(s.end_time / total) * 100}%`, touchAction: "none" }}
            >
              <div
                className={`h-full w-1.5 rounded-sm border border-black/40 ${
                  drag?.i === i ? "bg-warn" : "bg-white/90"
                }`}
              />
            </div>
          ))}
        {/* Playhead */}
        <div
          className="pointer-events-none absolute top-0 h-full w-0.5 bg-white"
          style={{ left: `${(currentTime / total) * 100}%` }}
        />
      </div>
      {drag && (
        <div className="text-xs tabular-nums text-warn">
          cut at {drag.t.toFixed(2)}s — release, then Save/Cancel above
        </div>
      )}
    </div>
  );
}
