import Link from "next/link";
import { api } from "@/lib/api";
import { fmtDate, fmtDuration, fmtPct } from "@/lib/format";
import { Panel, StatusBadge } from "@/components/ui";
import type { Video } from "@/lib/types";

export const metadata = { title: "Videos — Revisent" };

export const dynamic = "force-dynamic";

export default async function VideosPage() {
  let videos: Video[] = [];
  let error: string | null = null;
  try {
    videos = await api.listVideos();
  } catch (e) {
    error = (e as Error).message;
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Videos</h1>
          <p className="text-sm text-muted">{videos.length} uploaded</p>
        </div>
        <Link
          href="/upload"
          className="rounded bg-accent px-3 py-1.5 text-sm font-medium text-white hover:opacity-90"
        >
          + Upload video
        </Link>
      </div>

      {error ? (
        <Panel className="p-6 text-sm text-muted">Can’t reach the backend: {error}</Panel>
      ) : videos.length === 0 ? (
        <Panel className="p-8 text-center text-sm text-muted">
          No videos yet. Upload one via <code>POST /api/videos</code>.
        </Panel>
      ) : (
        <Panel className="overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs uppercase tracking-wide text-muted">
                <th className="px-4 py-2 font-medium">File</th>
                <th className="px-4 py-2 font-medium">Location</th>
                <th className="px-4 py-2 font-medium">Uploaded</th>
                <th className="px-4 py-2 font-medium">Duration</th>
                <th className="px-4 py-2 font-medium">Privacy</th>
                <th className="px-4 py-2 font-medium">Pipeline</th>
                <th className="px-4 py-2 font-medium">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {videos.map((v) => (
                <tr key={v.id} className="hover:bg-panel2">
                  <td className="px-4 py-2">
                    <Link href={`/videos/${v.id}`} className="text-text hover:text-accent">
                      {v.original_filename}
                    </Link>
                  </td>
                  <td className="px-4 py-2 text-muted">{v.property_tag || "—"}</td>
                  <td className="px-4 py-2 text-muted">{fmtDate(v.uploaded_at)}</td>
                  <td className="px-4 py-2 tabular-nums text-muted">{fmtDuration(v.duration_seconds)}</td>
                  <td className="px-4 py-2 tabular-nums text-muted">
                    {v.anonymization_method?.startsWith("none")
                      ? "blur off"
                      : fmtPct(v.anonymization_coverage)}
                  </td>
                  <td className="px-4 py-2">
                    <PipelineDots video={v} />
                  </td>
                  <td className="px-4 py-2">
                    <StatusBadge status={v.status} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
      )}
    </div>
  );
}

function PipelineDots({ video }: { video: Video }) {
  const steps: [string, boolean][] = [
    ["anonymized", ["anonymized", "processed"].includes(video.status)],
    ["hand pose", video.hand_pose_extracted],
    ["segmented", video.segmented],
    ["events", video.status === "processed"],
  ];
  return (
    <div className="flex items-center gap-1.5">
      {steps.map(([name, done]) => (
        <span
          key={name}
          title={name}
          className="h-2 w-2 rounded-full"
          style={{ backgroundColor: done ? "#22c55e" : "#27313f" }}
        />
      ))}
    </div>
  );
}
