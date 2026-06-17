"use client";

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { graspAperture, interpolateHandsAt, type HandsAt } from "@/lib/pose";
import type { HandPoseFrame, HeadPoseFrameT, Landmark } from "@/lib/types";

const Plot = dynamic(() => import("./plotly-client"), { ssr: false });

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

// (The old hidden body-rig renderer lived here; it duplicated lib/pose.ts
// and was unused. Recover from git history / lib/pose.ts if ever needed.)

// Camera-view hand rendering: x = image left->right, z(up) = image bottom->top
// (matches what you see in the video player), y = depth toward the viewer. The
// hand tints amber when the grasp closes — the same gripper signal we export.
//
// When METRIC world landmarks exist, the hand SHAPE comes from them (meters,
// drawn at constant real size — no perspective shrink), anchored at the image
// wrist position. Without them we fall back to image landmarks (shape only,
// apparent size).
const WORLD_SCALE = 2.2; // plot units per meter for the metric hand shape

function handTraces(lms: Landmark[] | null, world: Landmark[] | null | undefined,
                    base: string, ar: number) {
  if (!lms || lms.length < 21) return [];
  const aperture = graspAperture(lms);
  const color = aperture != null && aperture <= 0.3 ? "#f59e0b" : base;

  let P: [number, number, number][];
  if (world && world.length >= 21) {
    // Anchor = image wrist; shape = metric world offsets (x right, y down -> up,
    // z away-from-camera -> depth). The scene box is stretched horizontally by
    // the video aspect ratio, so metric x-offsets divide by it to keep the
    // hand's REAL proportions on wide and portrait clips alike.
    const ax = lms[0][0], au = 1 - lms[0][1];
    P = world.map((p) => [
      ax + ((p[0] - world[0][0]) * WORLD_SCALE) / ar,
      (p[2] - world[0][2]) * WORLD_SCALE,
      au - (p[1] - world[0][1]) * WORLD_SCALE,
    ]);
  } else {
    P = lms.map((p) => [p[0], p[2] * 2, 1 - p[1]]);
  }

  const px: (number | null)[] = [], py: (number | null)[] = [], pz: (number | null)[] = [];
  for (const [a, b] of HAND_CONNECTIONS) {
    px.push(P[a][0], P[b][0], null);
    py.push(P[a][1], P[b][1], null);
    pz.push(P[a][2], P[b][2], null);
  }
  return [
    { x: px, y: py, z: pz, type: "scatter3d", mode: "lines", line: { color, width: 5 }, hoverinfo: "skip", showlegend: false },
    {
      x: P.map((p) => p[0]), y: P.map((p) => p[1]), z: P.map((p) => p[2]),
      type: "scatter3d", mode: "markers", marker: { color, size: 2.5 },
      hoverinfo: "skip", showlegend: false,
    },
  ];
}

// Fixed frame so the view doesn't rescale as hands move: x/up span the camera
// image [0,1]; ticks hidden (normalized units carry no meaning for the viewer).
const AXIS = {
  color: "#64748b", gridcolor: "#27313f", showspikes: false,
  showticklabels: false, title: "", zeroline: false,
};
// Scene box mirrors the video frame's proportions: x spans the image width,
// z the height, so the panel matches what you see in the player whether the
// clip is portrait (9:16) or wide (16:9).
const makeScene = (ar: number) => ({
  bgcolor: "#0b0e13",
  xaxis: { ...AXIS, range: [0, 1] },
  yaxis: { ...AXIS, range: [-0.4, 0.4] },
  zaxis: { ...AXIS, range: [0, 1] },
  aspectmode: "manual" as const,
  aspectratio: { x: ar, y: 0.6, z: 1 },
  camera: { eye: { x: 0, y: -1.9, z: 0.15 }, up: { x: 0, y: 0, z: 1 } },
});
const LAYOUT_BASE = {
  paper_bgcolor: "#141a22",
  margin: { l: 0, r: 0, t: 0, b: 0 },
  showlegend: false,
  uirevision: "keep", // preserve camera angle across updates
};

export default function Pose3D({
  videoRef,
  frames,
  headFrames,
  operatorHeightCm,
}: {
  videoRef: React.RefObject<HTMLVideoElement>;
  frames: HandPoseFrame[];
  headFrames: HeadPoseFrameT[];
  operatorHeightCm: number;
}) {
  const [hands, setHands] = useState<HandsAt | null>(null);
  // Video aspect ratio (w/h), read from the element once metadata loads —
  // drives the 3D box proportions. Portrait default until known.
  const [ar, setAr] = useState(9 / 16);
  const lastTs = useRef(-1);

  useEffect(() => {
    const hTimes = frames.map((f) => f.timestamp_ms);
    let raf = 0;
    const tick = () => {
      const v = videoRef.current;
      if (v) {
        if (v.videoWidth > 0 && v.videoHeight > 0) {
          const a = Math.min(2.5, Math.max(0.4, v.videoWidth / v.videoHeight));
          setAr((prev) => (Math.abs(prev - a) > 0.01 ? a : prev));
        }
        const ms = v.currentTime * 1000;
        if (Math.abs(ms - lastTs.current) > 50) {
          lastTs.current = ms;
          // Interpolated between pose samples so the panel tracks the playhead
          // instead of stepping at the 10fps sampling rate.
          setHands(interpolateHandsAt(frames, hTimes, ms));
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [videoRef, frames, headFrames]);

  // Only hands present at the CURRENT playhead — no stale ghosts. An absent
  // hand simply isn't drawn (the legend dims to show which is which).
  const left = hands?.left ?? null;
  const right = hands?.right ?? null;
  const metric = !!(hands?.leftWorld || hands?.rightWorld);
  const handData = [
    ...handTraces(left, hands?.leftWorld, "#22c55e", ar),
    ...handTraces(right, hands?.rightWorld, "#38bdf8", ar),
  ];
  const hasHands = handData.length > 0;

  return (
    <div className="space-y-3">
      <Panel3D
        title={
          <span className="flex items-center gap-3">
            <span>3D hands · {metric ? "metric shape" : "camera view"}</span>
            <LegendDot color="#22c55e" label="left" active={!!left} />
            <LegendDot color="#38bdf8" label="right" active={!!right} />
            <LegendDot color="#f59e0b" label="closed grasp" active />
          </span>
        }
      >
        {hasHands ? (
          <Plot
            data={handData as any}
            layout={{ ...LAYOUT_BASE, scene: makeScene(ar) } as any}
            config={{ displayModeBar: false, responsive: true } as any}
            style={{ width: "100%", height: "240px" }}
            useResizeHandler
          />
        ) : (
          <Empty>No hand detected at this moment</Empty>
        )}
      </Panel3D>
    </div>
  );
}

function Panel3D({ title, children }: { title: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-panel">
      <div className="flex items-center justify-between border-b border-border px-3 py-1.5 text-xs text-muted">
        <span>{title}</span>
        <span className="text-[10px]">drag to rotate</span>
      </div>
      {children}
    </div>
  );
}

function LegendDot({ color, label, active }: { color: string; label: string; active: boolean }) {
  return (
    <span className={`flex items-center gap-1 text-[10px] ${active ? "" : "opacity-35"}`}>
      <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
      {label}
    </span>
  );
}
function Empty({ children }: { children: React.ReactNode }) {
  return <div className="flex h-[240px] items-center justify-center text-xs text-muted">{children}</div>;
}
