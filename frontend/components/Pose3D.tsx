"use client";

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import type { HandPoseFrame, Landmark } from "@/lib/types";

const Plot = dynamic(() => import("./plotly-client"), { ssr: false });

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: V3) => Math.hypot(a[0], a[1], a[2]);
const norm = (a: V3): V3 => {
  const l = len(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

// 2-bone IK: from shoulder S to target T, with bone lengths Lu, Lf. Returns elbow.
function ik(S: V3, T: V3, Lu: number, Lf: number): { elbow: V3; wrist: V3 } {
  let dir = sub(T, S);
  let d = len(dir);
  const dmax = Lu + Lf - 0.001;
  const dmin = Math.abs(Lu - Lf) + 0.001;
  d = Math.max(dmin, Math.min(d, dmax));
  const n = norm(dir);
  const wrist = add(S, mul(n, d));
  const a = (Lu * Lu + d * d - Lf * Lf) / (2 * d);
  const h = Math.sqrt(Math.max(0, Lu * Lu - a * a));
  // bend the elbow down-and-back
  let ref: V3 = [0, -1, -0.3];
  let perp = sub(ref, mul(n, dot(ref, n)));
  perp = norm(perp);
  const elbow = add(add(S, mul(n, a)), mul(perp, h));
  return { elbow, wrist };
}

// Map a hand wrist landmark (normalized image coords) into body space (meters).
function wristTarget(w: Landmark | undefined, H: number, side: number, sh: V3): V3 {
  if (!w) return add(sh, [side * 0.05, -0.38 * H, 0.12 * H]);
  const x = (w[0] - 0.5) * 0.9 * H;
  const y = 0.85 * H - (0.12 + w[1] * 0.5) * H;
  const z = (0.28 + (1 - w[1]) * 0.12) * H;
  return [x, y, z];
}

function bodyTraces(H: number, frame: HandPoseFrame | null) {
  const lSh: V3 = [-0.11 * H, 0.83 * H, 0];
  const rSh: V3 = [0.11 * H, 0.83 * H, 0];
  const joints: Record<string, V3> = {
    headTop: [0, H, 0],
    nose: [0, 0.95 * H, 0.06 * H],
    neck: [0, 0.85 * H, 0],
    chest: [0, 0.72 * H, 0.02 * H],
    pelvis: [0, 0.55 * H, 0],
    lSh, rSh,
    lHip: [-0.09 * H, 0.55 * H, 0],
    rHip: [0.09 * H, 0.55 * H, 0],
    lKnee: [-0.09 * H, 0.28 * H, 0.03 * H],
    rKnee: [0.09 * H, 0.28 * H, 0.03 * H],
    lAnk: [-0.09 * H, 0.03 * H, 0.05 * H],
    rAnk: [0.09 * H, 0.03 * H, 0.05 * H],
  };
  const Lu = 0.17 * H,
    Lf = 0.15 * H;
  const lArm = ik(lSh, wristTarget(frame?.left_hand_landmarks?.[0], H, -1, lSh), Lu, Lf);
  const rArm = ik(rSh, wristTarget(frame?.right_hand_landmarks?.[0], H, 1, rSh), Lu, Lf);

  const torso: [V3, V3][] = [
    [joints.headTop, joints.neck], [joints.neck, joints.chest], [joints.chest, joints.pelvis],
    [joints.neck, lSh], [joints.neck, rSh],
    [joints.pelvis, joints.lHip], [joints.pelvis, joints.rHip],
    [joints.lHip, joints.lKnee], [joints.lKnee, joints.lAnk],
    [joints.rHip, joints.rKnee], [joints.rKnee, joints.rAnk],
  ];
  const seg = (segs: [V3, V3][]) => {
    const x: (number | null)[] = [], y: (number | null)[] = [], z: (number | null)[] = [];
    for (const [a, b] of segs) {
      // plotly: x=right, y=forward(depth), z=up
      x.push(a[0], b[0], null); y.push(a[2], b[2], null); z.push(a[1], b[1], null);
    }
    return { x, y, z };
  };
  const line = (s: ReturnType<typeof seg>, color: string, w = 5) => ({
    ...s, type: "scatter3d", mode: "lines", line: { color, width: w }, hoverinfo: "skip", showlegend: false,
  });
  return [
    line(seg(torso), "#94a3b8", 5),
    line(seg([[lSh, lArm.elbow], [lArm.elbow, lArm.wrist]]), "#22c55e", 6),
    line(seg([[rSh, rArm.elbow], [rArm.elbow, rArm.wrist]]), "#ef4444", 6),
    {
      x: [joints.headTop[0]], y: [joints.headTop[2]], z: [joints.headTop[1]],
      type: "scatter3d", mode: "markers", marker: { color: "#3b82f6", size: 6 },
      hoverinfo: "skip", showlegend: false,
    },
  ];
}

function handTraces(lms: Landmark[] | null, color: string) {
  if (!lms) return [];
  const px: (number | null)[] = [], py: (number | null)[] = [], pz: (number | null)[] = [];
  for (const [a, b] of HAND_CONNECTIONS) {
    px.push(lms[a][0], lms[b][0], null);
    py.push(lms[a][2], lms[b][2], null);
    pz.push(-lms[a][1], -lms[b][1], null);
  }
  return [
    { x: px, y: py, z: pz, type: "scatter3d", mode: "lines", line: { color, width: 4 }, hoverinfo: "skip", showlegend: false },
    {
      x: lms.map((p) => p[0]), y: lms.map((p) => p[2]), z: lms.map((p) => -p[1]),
      type: "scatter3d", mode: "markers", marker: { color: "#ef4444", size: 3 },
      hoverinfo: "skip", showlegend: false,
    },
  ];
}

const SCENE = {
  bgcolor: "#0b0e13",
  xaxis: { color: "#64748b", gridcolor: "#27313f", showspikes: false, title: "X" },
  yaxis: { color: "#64748b", gridcolor: "#27313f", showspikes: false, title: "Y" },
  zaxis: { color: "#64748b", gridcolor: "#27313f", showspikes: false, title: "Z" },
  aspectmode: "data" as const,
};
const LAYOUT_BASE = {
  paper_bgcolor: "#141a22",
  margin: { l: 0, r: 0, t: 0, b: 0 },
  showlegend: false,
  uirevision: "keep", // preserve camera angle across updates
};

export default function Pose3D({
  videoRef,
  frames,
  operatorHeightCm,
}: {
  videoRef: React.RefObject<HTMLVideoElement>;
  frames: HandPoseFrame[];
  operatorHeightCm: number;
}) {
  const [frame, setFrame] = useState<HandPoseFrame | null>(frames[0] ?? null);
  const lastTs = useRef(-1);

  useEffect(() => {
    const times = frames.map((f) => f.timestamp_ms);
    const nearest = (ms: number): HandPoseFrame | null => {
      if (!frames.length) return null;
      let lo = 0, hi = times.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (times[mid] < ms) lo = mid + 1;
        else hi = mid;
      }
      if (lo > 0 && Math.abs(times[lo - 1] - ms) < Math.abs(times[lo] - ms)) lo -= 1;
      return frames[lo];
    };
    let raf = 0;
    const tick = () => {
      const v = videoRef.current;
      if (v) {
        const ms = v.currentTime * 1000;
        if (Math.abs(ms - lastTs.current) > 80) {
          lastTs.current = ms;
          setFrame(nearest(ms));
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [videoRef, frames]);

  const H = (operatorHeightCm || 170) / 100;
  const handData = [
    ...handTraces(frame?.left_hand_landmarks ?? null, "#22c55e"),
    ...handTraces(frame?.right_hand_landmarks ?? null, "#38bdf8"),
  ];
  const hasHands = handData.length > 0;

  return (
    <div className="space-y-3">
      <Panel3D title="3D hand pose">
        {hasHands ? (
          <Plot
            data={handData as any}
            layout={{ ...LAYOUT_BASE, scene: SCENE } as any}
            config={{ displayModeBar: false, responsive: true } as any}
            style={{ width: "100%", height: "240px" }}
            useResizeHandler
          />
        ) : (
          <Empty>No hand detected at this moment</Empty>
        )}
      </Panel3D>
      <Panel3D title="Body pose (estimated)">
        <Plot
          data={bodyTraces(H, frame) as any}
          layout={{ ...LAYOUT_BASE, scene: SCENE } as any}
          config={{ displayModeBar: false, responsive: true } as any}
          style={{ width: "100%", height: "260px" }}
          useResizeHandler
        />
      </Panel3D>
    </div>
  );
}

function Panel3D({ title, children }: { title: string; children: React.ReactNode }) {
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
function Empty({ children }: { children: React.ReactNode }) {
  return <div className="flex h-[240px] items-center justify-center text-xs text-muted">{children}</div>;
}
