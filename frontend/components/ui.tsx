import Link from "next/link";
import clsx from "clsx";
import type { VideoStatus } from "@/lib/types";

export function Panel({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={clsx(
        "rounded-lg border border-border bg-panel",
        className
      )}
    >
      {children}
    </div>
  );
}

export function StatCard({
  label,
  value,
  hint,
  tone = "default",
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
  tone?: "default" | "ok" | "warn" | "danger";
}) {
  const toneColor = {
    default: "text-text",
    ok: "text-ok",
    warn: "text-warn",
    danger: "text-danger",
  }[tone];
  return (
    <Panel className="p-4">
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      <div className={clsx("mt-1 text-2xl font-semibold tabular-nums", toneColor)}>
        {value}
      </div>
      {hint && <div className="mt-1 text-xs text-muted">{hint}</div>}
    </Panel>
  );
}

const STATUS_STYLES: Record<VideoStatus, string> = {
  uploaded: "bg-slate-600/30 text-slate-300",
  processing: "bg-blue-500/20 text-blue-300",
  anonymized: "bg-teal-500/20 text-teal-300",
  processed: "bg-green-500/20 text-green-300",
  failed: "bg-red-500/20 text-red-300",
};

export function StatusBadge({ status }: { status: VideoStatus }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium",
        STATUS_STYLES[status] || "bg-slate-600/30 text-slate-300"
      )}
    >
      {status === "processing" && (
        <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue-400" />
      )}
      {status}
    </span>
  );
}

export function Chip({ color, children }: { color: string; children: React.ReactNode }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs"
      style={{ backgroundColor: `${color}22`, color }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color }} />
      {children}
    </span>
  );
}

export function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <Link
      href={href}
      className="rounded px-3 py-1.5 text-sm text-muted hover:bg-panel2 hover:text-text"
    >
      {children}
    </Link>
  );
}
