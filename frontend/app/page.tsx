import Link from "next/link";
import { api } from "@/lib/api";
import { fmtDate, fmtDuration } from "@/lib/format";
import { Panel, StatCard, StatusBadge } from "@/components/ui";
import type { Overview, Video } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function OverviewPage() {
  let overview: Overview | null = null;
  let videos: Video[] = [];
  let error: string | null = null;
  try {
    [overview, videos] = await Promise.all([api.getOverview(), api.listVideos()]);
  } catch (e) {
    error = (e as Error).message;
  }

  if (error) {
    return <BackendDown error={error} />;
  }

  const ov = overview!;
  const recent = videos.slice(0, 6);
  const properties = Object.entries(ov.per_property);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Overview</h1>
        <p className="text-sm text-muted">Operational summary across processed footage.</p>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatCard label="Footage processed" value={`${ov.total_hours.toFixed(2)} h`} hint={`${ov.video_count} videos`} />
        <StatCard label="Service events" value={ov.total_service_events} tone="ok" />
        <StatCard label="Downtime" value={fmtDuration(ov.total_downtime_seconds)} tone="warn" />
        <StatCard
          label="Contamination flags"
          value={ov.total_contamination_flags}
          tone={ov.total_contamination_flags > 0 ? "danger" : "default"}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <Panel className="overflow-hidden">
          <div className="border-b border-border px-4 py-3 text-sm font-medium">
            Recent videos
          </div>
          {recent.length === 0 ? (
            <Empty>No videos uploaded yet.</Empty>
          ) : (
            <ul className="divide-y divide-border">
              {recent.map((v) => (
                <li key={v.id}>
                  <Link
                    href={`/videos/${v.id}`}
                    className="flex items-center gap-3 px-4 py-3 hover:bg-panel2"
                  >
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm">{v.original_filename}</div>
                      <div className="text-xs text-muted">
                        {fmtDate(v.uploaded_at)} · {v.property_tag || "untagged"}
                      </div>
                    </div>
                    <div className="text-xs text-muted">{fmtDuration(v.duration_seconds)}</div>
                    <StatusBadge status={v.status} />
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </Panel>

        <Panel className="overflow-hidden">
          <div className="border-b border-border px-4 py-3 text-sm font-medium">
            By property
          </div>
          {properties.length === 0 ? (
            <Empty>No property metrics yet.</Empty>
          ) : (
            <ul className="divide-y divide-border">
              {properties.map(([name, p]) => (
                <li key={name} className="px-4 py-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">{name}</span>
                    <span className="text-xs text-muted">{fmtDuration(p.total_seconds)}</span>
                  </div>
                  <div className="mt-1 flex gap-4 text-xs text-muted">
                    <span className="text-ok">{p.service_events} service</span>
                    <span className="text-warn">{fmtDuration(p.downtime_seconds)} idle</span>
                    {p.contamination_flags > 0 && (
                      <span className="text-danger">{p.contamination_flags} contam.</span>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <div className="px-4 py-8 text-center text-sm text-muted">{children}</div>;
}

function BackendDown({ error }: { error: string }) {
  return (
    <Panel className="p-8 text-center">
      <h1 className="text-lg font-semibold">Can’t reach the backend</h1>
      <p className="mt-2 text-sm text-muted">
        Start the API (default <code>http://localhost:8000</code>) and refresh.
      </p>
      <pre className="mt-4 overflow-auto rounded bg-panel2 p-3 text-left text-xs text-muted">
        {error}
      </pre>
    </Panel>
  );
}
