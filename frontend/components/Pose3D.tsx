"use client";

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import type { HandPoseFrame, HeadPoseFrameT, Landmark } from "@/lib/types";

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

// Rotate point p around a pivot c — X axis (lean fwd/back) and Y axis (turn).
const rotX = (p: V3, c: V3, a: number): V3 => {
  const y = p[1] - c[1], z = p[2] - c[2], ca = Math.cos(a), sa = Math.sin(a);
  return [p[0], c[1] + y * ca - z * sa, c[2] + y * sa + z * ca];
};
const rotY = (p: V3, c: V3, a: number): V3 => {
  const x = p[0] - c[0], z = p[2] - c[2], ca = Math.cos(a), sa = Math.sin(a);
  return [c[0] + x * ca + z * sa, p[1], c[2] - x * sa + z * ca];
};
export function quatToEuler(q: [number, number, number, number]) {
  const [w, x, y, z] = q;
  const pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x))));
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  return { yaw, pitch, roll };
}

// Render a hand at a body-space wrist anchor. Uses the live 21-point landmarks
// when the hand is tracked; otherwise draws a static "resting" hand so BOTH
// hands are always shown — an untracked/idle hand stays stagnant at the posed
// body's side. plotly axes here: x=right, y=depth(z), z=up(y).
function handAtWrist(wrist: V3, lms: Landmark[] | null, color: string) {
  const lineTrace = (pairs: [V3, V3][]) => {
    const x: (number | null)[] = [], y: (number | null)[] = [], z: (number | null)[] = [];
    for (const [a, b] of pairs) { x.push(a[0], b[0], null); y.push(a[2], b[2], null); z.push(a[1], b[1], null); }
    return { x, y, z, type: "scatter3d", mode: "lines", line: { color, width: 4 }, hoverinfo: "skip", showlegend: false };
  };
  if (lms && lms.length >= 21) {
    // Attach the measured hand at the body wrist: positions relative to the
    // hand's own wrist (landmark 0), scaled from normalized-image to body meters.
    const w0 = lms[0], s = 0.7;
    const P: V3[] = lms.map((p) => [
      wrist[0] + (p[0] - w0[0]) * s,
      wrist[1] - (p[1] - w0[1]) * s,
      wrist[2] + (p[2] - w0[2]) * s,
    ]);
    const bones = HAND_CONNECTIONS.map(([a, b]) => [P[a], P[b]] as [V3, V3]);
    return [
      lineTrace(bones),
      { x: P.map((p) => p[0]), y: P.map((p) => p[2]), z: P.map((p) => p[1]),
        type: "scatter3d", mode: "markers", marker: { color, size: 2.5 }, hoverinfo: "skip", showlegend: false },
    ];
  }
  // Static resting hand: a small fan of fingers hanging from the wrist.
  const rest: [V3, V3][] = [-0.045, -0.022, 0, 0.022, 0.045].map((dx) => {
    const knuckle: V3 = [wrist[0] + dx, wrist[1] - 0.02, wrist[2] + 0.02];
    const tip: V3 = [wrist[0] + dx * 1.1, wrist[1] - 0.1, wrist[2] + 0.05];
    return [knuckle, tip] as [V3, V3];
  });
  rest.push([wrist, [wrist[0], wrist[1] - 0.04, wrist[2] + 0.02]]);
  return [lineTrace(rest)];
}

function bodyTraces(
  H: number,
  leftHand: Landmark[] | null,
  rightHand: Landmark[] | null,
  head: { yaw: number; pitch: number; roll: number } | null
) {
  const joints: Record<string, V3> = {
    headTop: [0, H, 0],
    nose: [0, 0.95 * H, 0.06 * H],
    neck: [0, 0.85 * H, 0],
    chest: [0, 0.72 * H, 0.02 * H],
    pelvis: [0, 0.55 * H, 0],
    lSh: [-0.11 * H, 0.83 * H, 0],
    rSh: [0.11 * H, 0.83 * H, 0],
    lHip: [-0.09 * H, 0.55 * H, 0],
    rHip: [0.09 * H, 0.55 * H, 0],
    lKnee: [-0.09 * H, 0.28 * H, 0.03 * H],
    rKnee: [0.09 * H, 0.28 * H, 0.03 * H],
    lAnk: [-0.09 * H, 0.03 * H, 0.05 * H],
    rAnk: [0.09 * H, 0.03 * H, 0.05 * H],
  };
  // Drive the upper body with the MEASURED head pose: lean the torso by head
  // pitch (look down → lean forward), turn by head yaw, around the pelvis; the
  // head/neck get extra rotation on top.
  if (head) {
    const pelvis = joints.pelvis;
    const lean = head.pitch * 0.6;
    const turn = head.yaw * 0.7;
    for (const k of ["neck", "chest", "lSh", "rSh", "nose", "headTop"]) {
      joints[k] = rotX(rotY(joints[k], pelvis, turn), pelvis, lean);
    }
    const neck = joints.neck;
    for (const k of ["nose", "headTop"]) {
      joints[k] = rotX(rotY(joints[k], neck, head.yaw * 0.3), neck, head.pitch * 0.4);
    }
  }
  const lSh = joints.lSh, rSh = joints.rSh;
  const Lu = 0.17 * H,
    Lf = 0.15 * H;
  const lArm = ik(lSh, wristTarget(leftHand?.[0], H, -1, lSh), Lu, Lf);
  const rArm = ik(rSh, wristTarget(rightHand?.[0], H, 1, rSh), Lu, Lf);

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
  const dots = (pts: V3[], color: string, size = 5) => ({
    x: pts.map((p) => p[0]), y: pts.map((p) => p[2]), z: pts.map((p) => p[1]),
    type: "scatter3d", mode: "markers", marker: { color, size }, hoverinfo: "skip", showlegend: false,
  });
  return [
    line(seg(torso), "#64748b", 5),
    line(seg([[lSh, lArm.elbow], [lArm.elbow, lArm.wrist]]), "#22c55e", 6),
    line(seg([[rSh, rArm.elbow], [rArm.elbow, rArm.wrist]]), "#ef4444", 6),
    // Per-joint markers, colored by side: left green, right red, centre gray, head blue.
    dots([joints.neck, joints.chest, joints.pelvis], "#64748b", 5),
    dots([lSh, lArm.elbow, joints.lHip, joints.lKnee, joints.lAnk], "#22c55e", 5),
    dots([rSh, rArm.elbow, joints.rHip, joints.rKnee, joints.rAnk], "#ef4444", 5),
    dots([joints.headTop], "#3b82f6", 7),
    // Both hands, always: live when tracked, static-resting at the side otherwise.
    ...handAtWrist(lArm.wrist, leftHand, "#22c55e"),
    ...handAtWrist(rArm.wrist, rightHand, "#ef4444"),
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
      type: "scatter3d", mode: "markers", marker: { color, size: 3 },
      hoverinfo: "skip", showlegend: false,
    },
  ];
}

// Light mode (matches the reference): white scene, light grid, dark labels.
const AXIS = {
  color: "#475569", gridcolor: "#e5e7eb", zerolinecolor: "#cbd5e1",
  showspikes: false, showbackground: false,
};
const SCENE = {
  bgcolor: "#ffffff",
  xaxis: { ...AXIS, title: "X" },
  yaxis: { ...AXIS, title: "Y" },
  zaxis: { ...AXIS, title: "Z" },
  // "data" keeps true proportions (manual/equal aspect over-stretched the hand's
  // shallow depth and looked distorted). Use the camera angle to show depth.
  aspectmode: "data" as const,
};
const LAYOUT_BASE = {
  paper_bgcolor: "#ffffff",
  margin: { l: 0, r: 0, t: 0, b: 0 },
  showlegend: false,
  uirevision: "keep", // preserve camera angle across updates
};

// Fixed 3/4 camera angle (matches the reference) + rotation disabled.
const HAND_CAM = { eye: { x: -1.25, y: -1.6, z: 0.65 }, up: { x: 0, y: 0, z: 1 }, center: { x: 0, y: 0, z: 0 } };
const BODY_CAM = { eye: { x: -1.25, y: -1.7, z: 0.45 }, up: { x: 0, y: 0, z: 1 }, center: { x: 0, y: 0, z: 0 } };
// Rotation re-enabled + a live camera readout so you can dial in the exact view
// and hand me the numbers to lock. (uirevision keeps your rotation across frames.)
const SCENE_HAND = { ...SCENE, camera: HAND_CAM };
const SCENE_BODY = { ...SCENE, camera: BODY_CAM };

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
  const [frame, setFrame] = useState<HandPoseFrame | null>(frames[0] ?? null);
  const [head, setHead] = useState<HeadPoseFrameT | null>(headFrames[0] ?? null);
  // --- Interactive camera capture + lock (no guessing) ---
  // Free rotation by default. On every rotation the FULL camera object is shown
  // on-screen and logged to console. "Lock view" captures the exact camera and
  // disables rotation; it persists across refreshes via localStorage.
  const [handLock, setHandLock] = useState<any>(null);
  const [bodyLock, setBodyLock] = useState<any>(null);
  const [handLive, setHandLive] = useState<any>(HAND_CAM);
  const [bodyLive, setBodyLive] = useState<any>(BODY_CAM);
  useEffect(() => {
    try {
      const h = localStorage.getItem("pose_handcam");
      if (h) { const c = JSON.parse(h); setHandLock(c); setHandLive(c); }
      const b = localStorage.getItem("pose_bodycam");
      if (b) { const c = JSON.parse(b); setBodyLock(c); setBodyLive(c); }
    } catch { /* ignore */ }
  }, []);
  const onRelay = (e: any, setLive: (c: any) => void) => {
    const c = e?.["scene.camera"];
    if (c?.eye) {
      const cam = { eye: c.eye, center: c.center, up: c.up };
      setLive(cam);
      // eslint-disable-next-line no-console
      console.log("[Pose3D camera]", JSON.stringify(cam));
    }
  };
  const lockView = (live: any, setLock: (c: any) => void, key: string) => {
    setLock(live);
    try { localStorage.setItem(key, JSON.stringify(live)); } catch { /* ignore */ }
  };
  const unlockView = (setLock: (c: any) => void, key: string) => {
    setLock(null);
    try { localStorage.removeItem(key); } catch { /* ignore */ }
  };
  const camControls = (live: any, locked: any, setLock: (c: any) => void, key: string) => (
    <div className="flex items-center gap-2 font-mono text-[10px] text-slate-400">
      <span>{live?.eye ? `eye(${live.eye.x.toFixed(2)}, ${live.eye.y.toFixed(2)}, ${live.eye.z.toFixed(2)})` : "—"}</span>
      {locked ? (
        <button onClick={() => unlockView(setLock, key)}
          className="rounded border border-slate-300 px-1.5 py-0.5 text-slate-600 hover:bg-slate-50">🔓 unlock</button>
      ) : (
        <button onClick={() => lockView(live, setLock, key)}
          className="rounded border border-emerald-400 bg-emerald-50 px-1.5 py-0.5 text-emerald-700 hover:bg-emerald-100">📌 lock view</button>
      )}
    </div>
  );
  const lastTs = useRef(-1);
  // Last-seen landmarks per hand, so a hand that stops being detected stays
  // stagnant (resting at the body's side) instead of flickering out.
  const lastLeft = useRef<Landmark[] | null>(null);
  const lastRight = useRef<Landmark[] | null>(null);

  useEffect(() => {
    const hTimes = frames.map((f) => f.timestamp_ms);
    const kTimes = headFrames.map((f) => f.timestamp_ms);
    const nearest = <T,>(arr: T[], times: number[], ms: number): T | null => {
      if (!arr.length) return null;
      let lo = 0, hi = times.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (times[mid] < ms) lo = mid + 1;
        else hi = mid;
      }
      if (lo > 0 && Math.abs(times[lo - 1] - ms) < Math.abs(times[lo] - ms)) lo -= 1;
      return arr[lo];
    };
    let raf = 0;
    const tick = () => {
      const v = videoRef.current;
      if (v) {
        const ms = v.currentTime * 1000;
        if (Math.abs(ms - lastTs.current) > 80) {
          lastTs.current = ms;
          const nf = nearest(frames, hTimes, ms);
          setFrame(nf);
          if (nf?.left_hand_landmarks) lastLeft.current = nf.left_hand_landmarks;
          if (nf?.right_hand_landmarks) lastRight.current = nf.right_hand_landmarks;
          setHead(nearest(headFrames, kTimes, ms));
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [videoRef, frames, headFrames]);

  const H = (operatorHeightCm || 170) / 100;
  const headEuler = head?.tracked ? quatToEuler(head.quaternion) : null;
  // Effective hands: live this frame, else the last-seen pose (kept stagnant).
  const effLeft = frame?.left_hand_landmarks ?? lastLeft.current;
  const effRight = frame?.right_hand_landmarks ?? lastRight.current;
  const handData = [
    ...handTraces(effLeft, "#22c55e"),
    ...handTraces(effRight, "#ef4444"),
  ];
  const hasHands = handData.length > 0;

  return (
    <div className="space-y-3">
      <Panel3D title="3D hand pose · both hands"
        controls={camControls(handLive, handLock, setHandLock, "pose_handcam")}>
        {hasHands ? (
          <Plot
            data={handData as any}
            layout={{ ...LAYOUT_BASE, scene: { ...SCENE, camera: handLock ?? HAND_CAM, ...(handLock ? { dragmode: false } : {}) } } as any}
            config={{ displayModeBar: false, responsive: true } as any}
            style={{ width: "100%", height: "250px" }}
            onRelayout={(e: any) => onRelay(e, setHandLive)}
            useResizeHandler
          />
        ) : (
          <Empty>No hand detected at this moment</Empty>
        )}
      </Panel3D>
      {/* Body panel hidden: the body is INFERRED (a standing template), so it
          misrepresents real posture (e.g. shows standing while seated). Misleading
          for buyers. Code kept (bodyTraces / BODY_CAM / bodyLock). Lead with the
          MEASURED hands only. */}
    </div>
  );
}

function Panel3D({ title, controls, children }: { title: string; controls?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="overflow-hidden rounded-lg border border-slate-200 bg-white">
      <div className="flex items-center justify-between border-b border-slate-200 px-3 py-1.5 text-xs">
        <span className="font-medium text-slate-700">{title}</span>
        {controls}
      </div>
      {children}
    </div>
  );
}
function Empty({ children }: { children: React.ReactNode }) {
  return <div className="flex h-[240px] items-center justify-center bg-white text-xs text-slate-400">{children}</div>;
}
