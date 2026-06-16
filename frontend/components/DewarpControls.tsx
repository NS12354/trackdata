"use client";

import { useEffect, useMemo, useState } from "react";
import { Panel } from "@/components/ui";
import {
  getUndistortSettings,
  putUndistortSettings,
  undistortPreviewUrl,
  type UndistortSettings,
} from "@/lib/api";

/** Live de-fisheye tuning: before/after preview on this clip's cached sample
 * frame, with sliders that save the global default applied to new uploads. */
export default function DewarpControls({ videoId }: { videoId: string }) {
  const [loaded, setLoaded] = useState(false);
  const [enabled, setEnabled] = useState(true);
  const [strength, setStrength] = useState(0.35);
  const [zoom, setZoom] = useState(0);
  const [fov, setFov] = useState(120);
  const [calibrated, setCalibrated] = useState(false);
  const [saved, setSaved] = useState<UndistortSettings | null>(null);
  const [saving, setSaving] = useState(false);
  const [noFrame, setNoFrame] = useState(false);

  useEffect(() => {
    getUndistortSettings()
      .then((s) => {
        setEnabled(s.enabled);
        setStrength(s.strength);
        setZoom(s.zoom);
        setFov(s.fov_deg);
        setCalibrated(s.calibrated);
        setSaved(s);
        setLoaded(true);
      })
      .catch(() => setLoaded(true));
  }, []);

  // Debounce the corrected-preview URL so dragging a slider doesn't spam requests.
  const [debounced, setDebounced] = useState({ strength: 0.35, zoom: 0, fov: 120 });
  useEffect(() => {
    const h = setTimeout(() => setDebounced({ strength, zoom, fov }), 150);
    return () => clearTimeout(h);
  }, [strength, zoom, fov]);

  const rawUrl = useMemo(() => undistortPreviewUrl(videoId, { raw: true }), [videoId]);
  const corrUrl = useMemo(
    () =>
      undistortPreviewUrl(videoId, {
        strength: debounced.strength,
        zoom: debounced.zoom,
        fov: debounced.fov,
      }),
    [videoId, debounced]
  );

  const dirty =
    !!saved &&
    (saved.strength !== strength || saved.zoom !== zoom || saved.fov_deg !== fov || saved.enabled !== enabled);

  const onSave = async () => {
    setSaving(true);
    try {
      const s = await putUndistortSettings({ enabled, strength, zoom, fov_deg: fov });
      setSaved(s);
    } catch (e) {
      window.alert("Save failed: " + (e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  if (!loaded) return null;

  return (
    <Panel className="p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="text-sm font-medium">Lens correction (de-fisheye)</div>
        <span
          className={
            "rounded px-2 py-0.5 text-[11px] font-medium " +
            (calibrated ? "bg-accent/15 text-accent" : "bg-panel2 text-muted")
          }
        >
          {calibrated ? "Calibrated ✓" : "Estimate mode"}
        </span>
      </div>

      {noFrame ? (
        <p className="text-xs text-muted">
          No preview frame cached for this clip yet (older upload). Sliders below still set the global default.
        </p>
      ) : (
        <div className="grid grid-cols-2 gap-2">
          <figure className="space-y-1">
            <img
              src={rawUrl}
              alt="original"
              onError={() => setNoFrame(true)}
              className="w-full rounded border border-border"
            />
            <figcaption className="text-center text-[11px] text-muted">Original (fisheye)</figcaption>
          </figure>
          <figure className="space-y-1">
            <img src={corrUrl} alt="corrected" className="w-full rounded border border-accent/40" />
            <figcaption className="text-center text-[11px] text-accent">Corrected</figcaption>
          </figure>
        </div>
      )}

      <div className="mt-4 space-y-3">
        <label className="flex items-center gap-2 text-xs text-muted">
          <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
          Apply de-fisheye to new uploads
        </label>

        <Slider
          label="Strength"
          value={strength}
          min={0}
          max={1}
          step={0.01}
          disabled={!enabled || calibrated}
          fmt={(v) => `${Math.round(v * 100)}%`}
          onChange={setStrength}
        />
        <Slider
          label="Zoom / crop"
          value={zoom}
          min={0}
          max={1}
          step={0.01}
          disabled={!enabled}
          fmt={(v) => (v === 0 ? "crop (no border)" : v === 1 ? "keep all" : `${Math.round(v * 100)}%`)}
          onChange={setZoom}
        />
        <Slider
          label="Lens FOV"
          value={fov}
          min={60}
          max={160}
          step={1}
          disabled={!enabled || calibrated}
          fmt={(v) => `${Math.round(v)}°`}
          onChange={setFov}
        />
      </div>

      <div className="mt-4 flex items-center gap-3">
        <button
          type="button"
          onClick={onSave}
          disabled={saving || !dirty}
          className="rounded bg-accent px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
        >
          {saving ? "Saving…" : "Save as default"}
        </button>
        <span className="text-[11px] text-muted">
          {calibrated
            ? "Using calibrated intrinsics — strength/FOV are ignored."
            : "Applies to new uploads. This clip is already processed; re-upload to apply."}
        </span>
      </div>
    </Panel>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  disabled,
  fmt,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  fmt: (v: number) => string;
  onChange: (v: number) => void;
}) {
  return (
    <label className={"block " + (disabled ? "opacity-50" : "")}>
      <div className="mb-1 flex justify-between text-xs">
        <span className="text-muted">{label}</span>
        <span className="font-mono tabular-nums text-text">{fmt(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-accent"
      />
    </label>
  );
}
