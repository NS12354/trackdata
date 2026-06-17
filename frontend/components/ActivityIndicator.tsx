"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { Video } from "@/lib/types";

const TERMINAL = new Set(["processed", "failed"]);

/** Global pipeline heartbeat (lives in the nav bar, every page): which clip is
 * being processed right now, what stage, how far along — plus queue depth.
 * "I just want to know what's happening at all times." */
export default function ActivityIndicator() {
  const [active, setActive] = useState<Video[]>([]);
  const [prog, setProg] = useState<{ stage?: string; pct?: number | null; detail?: string; ts?: number }>({});

  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const vids = await api.listVideos();
        if (stop) return;
        const act = vids.filter((v) => !TERMINAL.has(v.status));
        setActive(act);
        if (act.length) {
          const p = await api.getProgress(act[0].id).catch(() => ({}));
          if (!stop) setProg(p || {});
        }
      } catch {
        /* backend briefly busy — keep last reading */
      }
    };
    tick();
    const iv = setInterval(tick, 4000);
    return () => { stop = true; clearInterval(iv); };
  }, []);

  if (!active.length) {
    return <span className="text-xs text-muted">● pipeline idle</span>;
  }
  const v = active[0];
  const fresh = prog.ts != null && Date.now() / 1000 - prog.ts < 45;
  const label = fresh
    ? `${prog.stage}${prog.pct != null ? ` ${Math.round(prog.pct)}%` : ""}` +
      (prog.detail ? ` — ${prog.detail}` : "")
    : "queued — waiting for worker";
  return (
    <Link href={`/videos/${v.id}`}
          className="flex max-w-[44rem] items-center gap-2 text-xs text-warn hover:opacity-80">
      <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-warn" />
      <span className="truncate">
        <span className="font-medium">{v.original_filename}</span> · {label}
        {active.length > 1 && <span className="text-muted"> (+{active.length - 1} queued)</span>}
      </span>
    </Link>
  );
}
