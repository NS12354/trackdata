import { api } from "@/lib/api";
import { Panel } from "@/components/ui";
import ChatPanel from "@/components/ChatPanel";
import type { Video } from "@/lib/types";

export const dynamic = "force-dynamic";
export const metadata = { title: "Chat — Revisent" };

export default async function ChatPage() {
  let videos: Video[] = [];
  let error: string | null = null;
  try {
    videos = await api.listVideos();
  } catch (e) {
    error = (e as Error).message;
  }

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold">Chat</h1>
        <p className="text-sm text-muted">
          Ask questions about the processed footage. Grounded in the activity
          commentary + metrics — runs on a local LLM ($0).
        </p>
      </div>
      {error ? (
        <Panel className="p-6 text-sm text-muted">Can’t reach the backend: {error}</Panel>
      ) : (
        <ChatPanel videos={videos} />
      )}
    </div>
  );
}
