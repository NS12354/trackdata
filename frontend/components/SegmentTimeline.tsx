"use client";

import { fmtClock, labelColor } from "@/lib/format";
import type { Segment } from "@/lib/types";

export default function SegmentTimeline({
  segments,
  duration,
  currentTime,
  onSeek,
}: {
  segments: Segment[];
  duration: number;
  currentTime: number;
  onSeek: (t: number) => void;
}) {
  const total = duration || (segments.length ? segments[segments.length - 1].end_time : 1);
  const activeIdx = segments.findIndex(
    (s) => currentTime >= s.start_time && currentTime < s.end_time
  );

  return (
    <div className="space-y-3">
      {/* Proportional bar */}
      <div
        className="relative h-7 w-full overflow-hidden rounded border border-border bg-panel2"
        role="group"
        aria-label="task timeline"
      >
        {segments.map((s, i) => {
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
        {/* Playhead */}
        <div
          className="pointer-events-none absolute top-0 h-full w-0.5 bg-white"
          style={{ left: `${(currentTime / total) * 100}%` }}
        />
      </div>

      {/* Segment list */}
      <ul className="max-h-72 space-y-1 overflow-auto pr-1">
        {segments.map((s, i) => (
          <li key={i}>
            <button
              onClick={() => onSeek(s.start_time)}
              className={`flex w-full items-start gap-2 rounded px-2 py-1.5 text-left text-sm hover:bg-panel2 ${
                activeIdx === i ? "bg-panel2 ring-1 ring-border" : ""
              }`}
            >
              <span
                className="mt-1 h-2.5 w-2.5 shrink-0 rounded-sm"
                style={{ backgroundColor: labelColor(s.task_label) }}
              />
              <span className="w-20 shrink-0 tabular-nums text-xs text-muted">
                {fmtClock(s.start_time)}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block">{s.task_label}</span>
                {s.description && (
                  <span className="block truncate text-xs text-muted">{s.description}</span>
                )}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
