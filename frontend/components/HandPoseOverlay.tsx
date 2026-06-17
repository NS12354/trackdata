"use client";

import { useEffect, useRef } from "react";
import { interpolateHandsAt } from "@/lib/pose";
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

    // Pose is sampled at ~10fps; the video plays at 30/60. Interpolating
    // between the surrounding samples at the exact playhead time keeps the
    // skeleton glued to the hand instead of stepping/lagging at 10Hz.
    const times = frames.map((f) => f.timestamp_ms);

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
          const hands = interpolateHandsAt(frames, times, video.currentTime * 1000);
          if (hands.left) drawHand(ctx, hands.left, w, h, "#22c55e", "#ef4444");
          if (hands.right) drawHand(ctx, hands.right, w, h, "#38bdf8", "#ef4444");
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
