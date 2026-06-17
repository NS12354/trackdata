"use client";

import { useState, useRef, useEffect } from "react";
import { chat } from "@/lib/api";
import { Panel } from "@/components/ui";
import type { Video } from "@/lib/types";

interface Msg {
  role: "user" | "assistant";
  text: string;
}

const EXAMPLES = [
  "What was the person doing in this video?",
  "When did they pick something up?",
  "Which activities took the longest?",
  "Summarize the activity timeline.",
];

export default function ChatPanel({ videos }: { videos: Video[] }) {
  const [videoId, setVideoId] = useState<string>("");
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  const send = async (q: string) => {
    const question = q.trim();
    if (!question || busy) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: question }]);
    setBusy(true);
    try {
      const reply = await chat(question, videoId || undefined);
      setMessages((m) => [...m, { role: "assistant", text: reply.answer }]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: "assistant", text: `⚠️ ${(e as Error).message}` },
      ]);
    } finally {
      setBusy(false);
    }
  };

  const processed = videos.filter((v) => v.segmented);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-sm">
        <span className="text-muted">Ask about:</span>
        <select
          value={videoId}
          onChange={(e) => setVideoId(e.target.value)}
          className="rounded border border-border bg-panel2 px-2 py-1 text-sm outline-none focus:border-accent"
        >
          <option value="">All footage</option>
          {processed.map((v) => (
            <option key={v.id} value={v.id}>
              {v.original_filename}
            </option>
          ))}
        </select>
      </div>

      <Panel className="flex h-[60vh] flex-col">
        <div className="flex-1 space-y-3 overflow-auto p-4">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
              <p className="text-sm text-muted">
                Ask about what happened in the footage. Answers are grounded in the
                activity commentary — the model doesn’t re-watch video.
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {EXAMPLES.map((ex) => (
                  <button
                    key={ex}
                    onClick={() => send(ex)}
                    className="rounded-full border border-border px-3 py-1 text-xs text-muted hover:border-accent hover:text-text"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((m, i) => (
              <div key={i} className={m.role === "user" ? "text-right" : ""}>
                <div
                  className={`inline-block max-w-[85%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm ${
                    m.role === "user"
                      ? "bg-accent/20 text-text"
                      : "bg-panel2 text-text"
                  }`}
                >
                  {m.text}
                </div>
              </div>
            ))
          )}
          {busy && <div className="text-sm text-muted">thinking…</div>}
          <div ref={endRef} />
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
          className="flex gap-2 border-t border-border p-3"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about the footage…"
            disabled={busy}
            className="flex-1 rounded border border-border bg-panel2 px-3 py-2 text-sm outline-none focus:border-accent"
          />
          <button
            type="submit"
            disabled={busy || !input.trim()}
            className="rounded bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            Send
          </button>
        </form>
      </Panel>
    </div>
  );
}
