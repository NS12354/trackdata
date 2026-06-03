"use client";

import { fmtDuration, labelColor } from "@/lib/format";
import { Panel, StatCard } from "@/components/ui";
import type { VideoSummary } from "@/lib/types";

export default function EventMetrics({ summary }: { summary: VideoSummary }) {
  const tasks = Object.entries(summary.time_per_task).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...tasks.map(([, v]) => v));

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatCard label="Service events" value={summary.service_event_count} tone="ok" />
        <StatCard label="Downtime" value={fmtDuration(summary.downtime_seconds)} tone="warn" />
        <StatCard
          label="Contamination"
          value={summary.contamination_event_count}
          tone={summary.contamination_event_count > 0 ? "danger" : "default"}
        />
        <StatCard
          label="Active hands"
          value={summary.active_hand_seconds != null ? fmtDuration(summary.active_hand_seconds) : "—"}
          hint="time with detected hands"
        />
      </div>

      <Panel className="p-4">
        <div className="mb-3 text-sm font-medium">Time per task</div>
        {tasks.length === 0 ? (
          <div className="text-sm text-muted">No events.</div>
        ) : (
          <div className="space-y-2">
            {tasks.map(([label, secs]) => (
              <div key={label} className="flex items-center gap-3">
                <div className="w-48 shrink-0 truncate text-sm text-muted">{label}</div>
                <div className="h-4 flex-1 overflow-hidden rounded bg-panel2">
                  <div
                    className="h-full rounded"
                    style={{ width: `${(secs / max) * 100}%`, backgroundColor: labelColor(label) }}
                  />
                </div>
                <div className="w-16 shrink-0 text-right text-sm tabular-nums">
                  {fmtDuration(secs)}
                </div>
              </div>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}
