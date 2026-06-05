// Shared body-pose kinematics: turn measured head pose + hands into named 3D
// joint positions (meters; +X right, +Y up, +Z forward; feet near y=0). Used by
// both the Plotly stick-figure view and the Three.js humanoid model so they
// always agree.
import type { Landmark } from "@/lib/types";

export type V3 = [number, number, number];

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: V3) => Math.hypot(a[0], a[1], a[2]);
const norm = (a: V3): V3 => {
  const l = len(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

export function quatToEuler(q: [number, number, number, number]) {
  const [w, x, y, z] = q;
  const pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (w * y - z * x))));
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
  return { yaw, pitch, roll };
}

// 2-bone IK: shoulder S -> target T with bone lengths Lu, Lf. Returns elbow+wrist.
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
  let ref: V3 = [0, -1, -0.3];
  let perp = sub(ref, mul(n, dot(ref, n)));
  perp = norm(perp);
  const elbow = add(add(S, mul(n, a)), mul(perp, h));
  return { elbow, wrist };
}

function wristTarget(w: Landmark | undefined, H: number, side: number, sh: V3): V3 {
  if (!w) return add(sh, [side * 0.05, -0.38 * H, 0.12 * H]);
  const x = (w[0] - 0.5) * 0.9 * H;
  const y = 0.85 * H - (0.12 + w[1] * 0.5) * H;
  const z = (0.28 + (1 - w[1]) * 0.12) * H;
  return [x, y, z];
}

const rotX = (p: V3, c: V3, a: number): V3 => {
  const y = p[1] - c[1], z = p[2] - c[2], ca = Math.cos(a), sa = Math.sin(a);
  return [p[0], c[1] + y * ca - z * sa, c[2] + y * sa + z * ca];
};
const rotY = (p: V3, c: V3, a: number): V3 => {
  const x = p[0] - c[0], z = p[2] - c[2], ca = Math.cos(a), sa = Math.sin(a);
  return [c[0] + x * ca + z * sa, p[1], c[2] - x * sa + z * ca];
};

export interface BodyJoints {
  pelvis: V3; chest: V3; neck: V3; head: V3;
  lSh: V3; rSh: V3; lElbow: V3; rElbow: V3; lWrist: V3; rWrist: V3;
  lHip: V3; rHip: V3; lKnee: V3; rKnee: V3; lAnk: V3; rAnk: V3;
}

export function computeBodyJoints(
  H: number,
  leftHand: Landmark[] | null,
  rightHand: Landmark[] | null,
  head: { yaw: number; pitch: number; roll: number } | null
): BodyJoints {
  const j: Record<string, V3> = {
    headTop: [0, H, 0],
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
  // Drive the upper body with the MEASURED head pose: lean by pitch, turn by yaw.
  if (head) {
    const pelvis = j.pelvis;
    const lean = head.pitch * 0.6;
    const turn = head.yaw * 0.7;
    for (const k of ["neck", "chest", "lSh", "rSh", "headTop"]) {
      j[k] = rotX(rotY(j[k], pelvis, turn), pelvis, lean);
    }
    const neck = j.neck;
    j.headTop = rotX(rotY(j.headTop, neck, head.yaw * 0.3), neck, head.pitch * 0.4);
  }
  const Lu = 0.17 * H, Lf = 0.15 * H;
  const lArm = ik(j.lSh, wristTarget(leftHand?.[0], H, -1, j.lSh), Lu, Lf);
  const rArm = ik(j.rSh, wristTarget(rightHand?.[0], H, 1, j.rSh), Lu, Lf);
  const head3: V3 = [
    (j.neck[0] + j.headTop[0]) / 2,
    (j.neck[1] + j.headTop[1]) / 2,
    (j.neck[2] + j.headTop[2]) / 2,
  ];
  return {
    pelvis: j.pelvis, chest: j.chest, neck: j.neck, head: head3,
    lSh: j.lSh, rSh: j.rSh,
    lElbow: lArm.elbow, rElbow: rArm.elbow, lWrist: lArm.wrist, rWrist: rArm.wrist,
    lHip: j.lHip, rHip: j.rHip, lKnee: j.lKnee, rKnee: j.rKnee, lAnk: j.lAnk, rAnk: j.rAnk,
  };
}

/** Nearest-by-timestamp lookup for syncing pose frames to the video clock. */
export function nearestByTime<T extends { timestamp_ms: number }>(
  arr: T[], times: number[], ms: number
): T | null {
  if (!arr.length) return null;
  let lo = 0, hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] < ms) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(times[lo - 1] - ms) < Math.abs(times[lo] - ms)) lo -= 1;
  return arr[lo];
}
