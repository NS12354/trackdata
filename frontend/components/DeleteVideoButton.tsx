"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { VideoStatus } from "@/lib/types";

/** Cancel (if processing) or delete a video, from the videos list. */
export default function DeleteVideoButton({
  id,
  status,
  filename,
}: {
  id: string;
  status: VideoStatus;
  filename?: string;
}) {
  const router = useRouter();
  const [deleting, setDeleting] = useState(false);
  const processing = status === "uploaded" || status === "processing";

  const onClick = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const name = filename ? ` “${filename}”` : "";
    const msg = processing
      ? `Cancel processing and delete${name}? The upload and all derived data will be removed.`
      : `Delete${name} and all its data? This cannot be undone.`;
    if (!window.confirm(msg)) return;
    setDeleting(true);
    try {
      await api.deleteVideo(id);
      router.refresh();
    } catch (err) {
      setDeleting(false);
      window.alert("Delete failed: " + (err as Error).message);
    }
  };

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={deleting}
      title={processing ? "Cancel processing & delete" : "Delete video"}
      className="rounded border border-danger/40 bg-danger/10 px-2 py-1 text-xs font-medium text-danger hover:bg-danger/20 disabled:opacity-50"
    >
      {deleting ? "…" : processing ? "✕ Cancel" : "🗑"}
    </button>
  );
}
