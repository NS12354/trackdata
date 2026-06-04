"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { uploadVideo } from "@/lib/api";
import { Panel } from "@/components/ui";

export default function UploadForm() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [propertyTag, setPropertyTag] = useState("");
  const [operatorId, setOperatorId] = useState("default-operator");
  const [workerId, setWorkerId] = useState("");
  const [heightCm, setHeightCm] = useState("");
  const [pct, setPct] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setError(null);
    setBusy(true);
    setPct(0);
    try {
      const res = await uploadVideo(
        file,
        {
          operator_id: operatorId || undefined,
          property_tag: propertyTag || undefined,
          worker_id_anonymized: workerId || undefined,
          operator_height_cm: heightCm || undefined,
        },
        setPct
      );
      setPct(100);
      // Processing runs in the background; the detail page shows live status.
      router.push(`/videos/${res.id}`);
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
      setPct(null);
    }
  };

  return (
    <Panel className="max-w-xl p-6">
      <form onSubmit={onSubmit} className="space-y-4">
        <Field label="Video file (.mp4 / .mov, up to 5GB)">
          <input
            type="file"
            accept="video/mp4,video/quicktime,.mp4,.mov"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            required
            className="block w-full text-sm text-muted file:mr-3 file:rounded file:border file:border-border file:bg-panel2 file:px-3 file:py-1.5 file:text-sm file:text-text hover:file:border-accent"
          />
          {file && (
            <p className="mt-1 text-xs text-muted">
              {file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB
            </p>
          )}
        </Field>

        <div className="grid grid-cols-2 gap-3">
          <Field label="Property tag (optional)">
            <Input value={propertyTag} onChange={setPropertyTag} placeholder="e.g. Greystar-Maple" />
          </Field>
          <Field label="Operator height cm (for body pose)">
            <Input value={heightCm} onChange={setHeightCm} placeholder="e.g. 162" />
          </Field>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <Field label="Operator ID">
            <Input value={operatorId} onChange={setOperatorId} />
          </Field>
          <Field label="Worker ID (anonymized, optional)">
            <Input value={workerId} onChange={setWorkerId} placeholder="e.g. W-001" />
          </Field>
        </div>

        {pct !== null && (
          <div>
            <div className="h-2 w-full overflow-hidden rounded bg-panel2">
              <div className="h-full rounded bg-accent transition-all" style={{ width: `${pct}%` }} />
            </div>
            <p className="mt-1 text-xs text-muted">
              {pct < 100 ? `Uploading… ${pct}%` : "Uploaded — starting processing…"}
            </p>
          </div>
        )}

        {error && (
          <div className="rounded border border-danger/40 bg-danger/10 p-3 text-sm text-danger">
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={!file || busy}
          className="rounded bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {busy ? "Uploading…" : "Upload & process"}
        </button>
        <p className="text-xs text-muted">
          After upload, the clip is anonymized → hand-pose → segmented → events
          automatically. You’ll be taken to its page to watch progress.
        </p>
      </form>
    </Panel>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs uppercase tracking-wide text-muted">{label}</span>
      {children}
    </label>
  );
}

function Input({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <input
      type="text"
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded border border-border bg-panel2 px-3 py-1.5 text-sm text-text outline-none focus:border-accent"
    />
  );
}
