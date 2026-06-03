export function fmtDuration(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  if (m < 60) return rem ? `${m}m ${rem}s` : `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export function fmtClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

export function fmtDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function fmtPct(x: number | null | undefined): string {
  if (x == null) return "—";
  return `${Math.round(x * 100)}%`;
}

export function fmtBytes(n: number | null | undefined): string {
  if (!n) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

// Stable color per task/event label for the timeline + chips.
const PALETTE: Record<string, string> = {
  service: "#22c55e",
  "moving container": "#22c55e",
  "loading/unloading": "#16a34a",
  "opening gate/enclosure": "#10b981",
  "manipulating lock or latch": "#14b8a6",
  contamination: "#ef4444",
  "handling overflow/contamination": "#ef4444",
  downtime: "#f59e0b",
  idle: "#a16207",
  "idle/waiting": "#a16207",
  transit: "#3b82f6",
  "transit/walking": "#3b82f6",
  "approaching property": "#6366f1",
  task: "#8b5cf6",
};

export function labelColor(label: string): string {
  return PALETTE[label] || "#64748b";
}
