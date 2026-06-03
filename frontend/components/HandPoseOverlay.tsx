"use client";

import { useEffect, useRef } from "react";
import type { HandPoseFrame, Landmark } from "@/lib/types";

// MediaPipe Hands 21-point skeleton connections.
const CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

function drawHand(
  ctx: CanvasRenderingContext2D,
  lms: Landmark[],
  w: number,
  h: number,
  bone: string,
  joint: string
) {
  ctx.lineWidth = 2;
  ctx.strokeStyle = bone;
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(lms[a][0] * w, lms[a][1] * h);
    ctx.lineTo(lms[b][0] * w, lms[b][1] * h);
    ctx.stroke();
  }
  ctx.fillStyle = joint;
  for (const p of lms) {
    ctx.beginPath();
    ctx.arc(p[0] * w, p[1] * h, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

export default function HandPoseOverlay({
  videoRef,
  frames,
  enabled,
}: {
  videoRef: React.RefObject<HTMLVideoElement>;
  frames: HandPoseFrame[];
  enabled: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    // frames are sorted by timestamp; binary-search the nearest to currentTime.
    const times = frames.map((f) => f.timestamp_ms);
    const nearest = (ms: number): HandPoseFrame | null => {
      if (frames.length === 0) return null;
      let lo = 0,
        hi = times.length - 1;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (times[mid] < ms) lo = mid + 1;
        else hi = mid;
      }
      // pick the closer of lo and lo-1
      if (lo > 0 && Math.abs(times[lo - 1] - ms) <= Math.abs(times[lo] - ms)) lo -= 1;
      // only show if within ~150ms (otherwise the hand wasn't sampled here)
      return Math.abs(times[lo] - ms) <= 150 ? frames[lo] : null;
    };

    let raf = 0;
    const render = () => {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        const w = video.clientWidth;
        const h = video.clientHeight;
        if (canvas.width !== w) canvas.width = w;
        if (canvas.height !== h) canvas.height = h;
        ctx.clearRect(0, 0, w, h);
        if (enabled) {
          const f = nearest(video.currentTime * 1000);
          if (f) {
            if (f.left_hand_landmarks) drawHand(ctx, f.left_hand_landmarks, w, h, "#22c55e", "#ef4444");
            if (f.right_hand_landmarks) drawHand(ctx, f.right_hand_landmarks, w, h, "#38bdf8", "#ef4444");
          }
        }
      }
      raf = requestAnimationFrame(render);
    };
    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, [videoRef, frames, enabled]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 h-full w-full"
    />
  );
}
