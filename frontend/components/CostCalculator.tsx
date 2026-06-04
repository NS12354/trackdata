"use client";

import { useState } from "react";
import { Panel, StatCard } from "@/components/ui";

const GPU_PRESETS: { label: string; rate: number }[] = [
  { label: "RTX 4090 (spot)", rate: 0.4 },
  { label: "NVIDIA L4", rate: 0.8 },
  { label: "A10G", rate: 1.2 },
  { label: "A100", rate: 2.5 },
];

const money = (n: number) =>
  n >= 1000
    ? `$${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k`
    : `$${n.toFixed(n < 10 ? 2 : 0)}`;

export default function CostCalculator() {
  const [footageHours, setFootageHours] = useState(1000);
  const [gpuRate, setGpuRate] = useState(0.8);
  const [throughput, setThroughput] = useState(2); // hrs footage per GPU-hour
  const [gbPerHour, setGbPerHour] = useState(3); // 1080p ≈ 3GB/hr
  const [includeStorage, setIncludeStorage] = useState(true);
  const [storageMonths, setStorageMonths] = useState(1);
  const [downloadPct, setDownloadPct] = useState(100);
  const [segFps, setSegFps] = useState(1);

  // --- compute ---
  const gpuHours = throughput > 0 ? footageHours / throughput : 0;
  const computeCost = gpuHours * gpuRate;

  const sourceGB = footageHours * gbPerHour;
  const storageCost = includeStorage ? sourceGB * 0.023 * storageMonths : 0; // ~$0.023/GB/mo
  const egressGB = includeStorage ? sourceGB * (downloadPct / 100) : 0;
  const egressCost = egressGB * 0.09; // ~$0.09/GB

  const total = computeCost + storageCost + egressCost;
  const perHour = footageHours > 0 ? total / footageHours : 0;

  // --- Claude API comparison (segmentation only) ---
  const frames = footageHours * 3600 * segFps;
  const apiCost = frames * 0.002; // ~$0.002 per sampled frame (small image)

  return (
    <div className="grid gap-5 lg:grid-cols-[1fr_1.1fr]">
      {/* Inputs */}
      <Panel className="space-y-5 p-5">
        <NumberField label="Footage to process (hours)" value={footageHours} onChange={setFootageHours} min={1} step={10} />

        <div>
          <Label>GPU</Label>
          <div className="mb-2 flex flex-wrap gap-2">
            {GPU_PRESETS.map((g) => (
              <button
                key={g.label}
                onClick={() => setGpuRate(g.rate)}
                className={`rounded border px-2.5 py-1 text-xs ${
                  gpuRate === g.rate ? "border-accent text-accent" : "border-border text-muted hover:text-text"
                }`}
              >
                {g.label} · ${g.rate.toFixed(2)}/hr
              </button>
            ))}
          </div>
          <NumberField label="GPU rate ($/hour)" value={gpuRate} onChange={setGpuRate} min={0.1} step={0.1} />
        </div>

        <SliderField
          label={`Throughput: ${throughput}× real-time`}
          hint={`processes ${throughput} hr of footage per GPU-hour`}
          value={throughput}
          onChange={setThroughput}
          min={0.5}
          max={6}
          step={0.5}
        />

        <label className="flex items-center gap-2 text-sm text-muted">
          <input type="checkbox" checked={includeStorage} onChange={(e) => setIncludeStorage(e.target.checked)} />
          Include storage &amp; egress
        </label>
        {includeStorage && (
          <div className="space-y-4 rounded border border-border bg-panel2/40 p-3">
            <NumberField label="Video size (GB per hour, 1080p≈3)" value={gbPerHour} onChange={setGbPerHour} min={0.5} step={0.5} />
            <NumberField label="Storage duration (months)" value={storageMonths} onChange={setStorageMonths} min={0} step={1} />
            <SliderField
              label={`Download results back: ${downloadPct}%`}
              hint="share of data egressed (anonymized video + extracts)"
              value={downloadPct}
              onChange={setDownloadPct}
              min={0}
              max={100}
              step={10}
            />
          </div>
        )}

        <SliderField
          label={`Segmentation sampling: ${segFps} fps`}
          hint="for the API-cost comparison"
          value={segFps}
          onChange={setSegFps}
          min={0.25}
          max={2}
          step={0.25}
        />
      </Panel>

      {/* Results */}
      <div className="space-y-5">
        <div className="grid grid-cols-2 gap-3">
          <StatCard label="Total (GPU)" value={money(total)} hint={`${money(perHour)} per hour of footage`} tone="ok" />
          <StatCard label="GPU-hours" value={Math.round(gpuHours).toLocaleString()} hint={`at ${throughput}× real-time`} />
        </div>

        <Panel className="p-5">
          <div className="mb-3 text-sm font-medium">Breakdown</div>
          <Row label={`Compute (${Math.round(gpuHours).toLocaleString()} GPU-hr × $${gpuRate.toFixed(2)})`} value={money(computeCost)} />
          {includeStorage && (
            <>
              <Row label={`Storage (${(sourceGB / 1000).toFixed(1)} TB × ${storageMonths} mo)`} value={money(storageCost)} />
              <Row label={`Egress (${(egressGB / 1000).toFixed(1)} TB × $0.09/GB)`} value={money(egressCost)} />
            </>
          )}
          <div className="mt-2 flex justify-between border-t border-border pt-2 text-sm font-semibold">
            <span>Total</span>
            <span className="tabular-nums">{money(total)}</span>
          </div>
        </Panel>

        <Panel className="p-5">
          <div className="mb-2 text-sm font-medium">vs. Claude API (segmentation only)</div>
          <Row label={`${(frames / 1e6).toFixed(1)}M frames × ~$0.002`} value={money(apiCost)} />
          <p className="mt-2 text-xs text-muted">
            Running the VLM yourself on a GPU is{" "}
            <span className="text-ok">
              ~{apiCost > 0 && computeCost > 0 ? Math.round(apiCost / computeCost) : "—"}× cheaper
            </span>{" "}
            than the API at this volume — which is why you self-host at scale.
          </p>
        </Panel>

        <p className="text-xs text-muted">
          Estimates. Throughput is the biggest unknown — benchmark on a real GPU to firm it up.
          Spot/reserved instances and buying a GPU lower cost for recurring volume.
        </p>
      </div>
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <div className="mb-1 text-xs uppercase tracking-wide text-muted">{children}</div>;
}

function NumberField({
  label,
  value,
  onChange,
  min,
  step,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  step?: number;
}) {
  return (
    <label className="block">
      <Label>{label}</Label>
      <input
        type="number"
        value={value}
        min={min}
        step={step}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        className="w-full rounded border border-border bg-panel2 px-3 py-1.5 text-sm tabular-nums text-text outline-none focus:border-accent"
      />
    </label>
  );
}

function SliderField({
  label,
  hint,
  value,
  onChange,
  min,
  max,
  step,
}: {
  label: string;
  hint?: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
}) {
  return (
    <label className="block">
      <Label>{label}</Label>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-accent"
      />
      {hint && <div className="text-xs text-muted">{hint}</div>}
    </label>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between py-1 text-sm">
      <span className="text-muted">{label}</span>
      <span className="tabular-nums">{value}</span>
    </div>
  );
}
