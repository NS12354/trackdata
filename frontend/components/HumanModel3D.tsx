"use client";

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Grid } from "@react-three/drei";
import * as THREE from "three";
import type { HandPoseFrame, HeadPoseFrameT, Landmark } from "@/lib/types";
import { computeBodyJoints, quatToEuler, nearestByTime, type BodyJoints, type V3 } from "@/lib/pose";

const UP = new THREE.Vector3(0, 1, 0);
const SKIN = "#b9bec7";   // mannequin gray
const JOINTC = "#8a9099"; // ball-joint gray

const vec = (p: V3) => new THREE.Vector3(p[0], p[1], p[2]);

function Humanoid({
  videoRef, frames, headFrames, H,
}: {
  videoRef: React.RefObject<HTMLVideoElement>;
  frames: HandPoseFrame[];
  headFrames: HeadPoseFrameT[];
  H: number;
}) {
  const hTimes = useMemo(() => frames.map((f) => f.timestamp_ms), [frames]);
  const kTimes = useMemo(() => headFrames.map((f) => f.timestamp_ms), [headFrames]);
  const lastL = useRef<Landmark[] | null>(null);
  const lastR = useRef<Landmark[] | null>(null);

  const limbRefs = useRef<Record<string, THREE.Mesh | null>>({});
  const jointRefs = useRef<Record<string, THREE.Mesh | null>>({});
  const headRef = useRef<THREE.Mesh>(null);

  // [from, to, radius] — torso/limbs as tapered cylinders, joints as spheres.
  const BONES = useMemo<[keyof BodyJoints, keyof BodyJoints, number][]>(() => [
    ["pelvis", "chest", 0.10 * H],
    ["chest", "neck", 0.07 * H],
    ["neck", "head", 0.045 * H],
    ["lSh", "rSh", 0.05 * H],
    ["lHip", "rHip", 0.06 * H],
    ["lSh", "lElbow", 0.05 * H], ["lElbow", "lWrist", 0.042 * H],
    ["rSh", "rElbow", 0.05 * H], ["rElbow", "rWrist", 0.042 * H],
    ["lHip", "lKnee", 0.06 * H], ["lKnee", "lAnk", 0.05 * H],
    ["rHip", "rKnee", 0.06 * H], ["rKnee", "rAnk", 0.05 * H],
  ], [H]);
  const JOINTS = useMemo<(keyof BodyJoints)[]>(() => [
    "pelvis", "chest", "neck", "lSh", "rSh", "lElbow", "rElbow",
    "lWrist", "rWrist", "lHip", "rHip", "lKnee", "rKnee", "lAnk", "rAnk",
  ], []);

  useFrame(() => {
    let left = lastL.current, right = lastR.current;
    let headE: { yaw: number; pitch: number; roll: number } | null = null;
    const v = videoRef.current;
    if (v && frames.length) {
      const ms = v.currentTime * 1000;
      const hf = nearestByTime(frames, hTimes, ms);
      if (hf?.left_hand_landmarks) { left = hf.left_hand_landmarks; lastL.current = left; }
      if (hf?.right_hand_landmarks) { right = hf.right_hand_landmarks; lastR.current = right; }
      const kf = nearestByTime(headFrames, kTimes, ms);
      if (kf?.tracked) headE = quatToEuler(kf.quaternion);
    }
    const j = computeBodyJoints(H, left, right, headE);

    for (const [a, b] of BONES) {
      const m = limbRefs.current[`${a}_${b}`];
      if (!m) continue;
      const pa = vec(j[a]), pb = vec(j[b]);
      const dir = pb.clone().sub(pa);
      const L = dir.length() || 1e-3;
      m.position.copy(pa.add(pb).multiplyScalar(0.5));
      m.scale.set(1, L, 1);
      m.quaternion.setFromUnitVectors(UP, dir.normalize());
    }
    for (const key of JOINTS) {
      const m = jointRefs.current[key];
      if (m) m.position.copy(vec(j[key]));
    }
    if (headRef.current) headRef.current.position.copy(vec(j.head));
  });

  return (
    <group>
      {BONES.map(([a, b, r]) => (
        <mesh key={`${a}_${b}`} ref={(el) => { limbRefs.current[`${a}_${b}`] = el; }} castShadow>
          <cylinderGeometry args={[r, r, 1, 14]} />
          <meshStandardMaterial color={SKIN} metalness={0.15} roughness={0.55} />
        </mesh>
      ))}
      {JOINTS.map((key) => (
        <mesh key={key} ref={(el) => { jointRefs.current[key] = el; }} castShadow>
          <sphereGeometry args={[0.052 * H, 16, 16]} />
          <meshStandardMaterial color={JOINTC} metalness={0.25} roughness={0.5} />
        </mesh>
      ))}
      <mesh ref={headRef} castShadow>
        <sphereGeometry args={[0.12 * H, 24, 24]} />
        <meshStandardMaterial color={SKIN} metalness={0.15} roughness={0.55} />
      </mesh>
    </group>
  );
}

export default function HumanModel3D({
  videoRef, frames, headFrames, operatorHeightCm,
}: {
  videoRef: React.RefObject<HTMLVideoElement>;
  frames: HandPoseFrame[];
  headFrames: HeadPoseFrameT[];
  operatorHeightCm: number;
}) {
  const H = (operatorHeightCm || 170) / 100;
  return (
    <Canvas shadows camera={{ position: [0.2 * H, 0.95 * H, 2.3 * H], fov: 42 }}
      style={{ width: "100%", height: "100%" }}>
      <color attach="background" args={["#0b0e13"]} />
      <ambientLight intensity={0.65} />
      <directionalLight position={[3, 6, 4]} intensity={1.15} castShadow
        shadow-mapSize-width={1024} shadow-mapSize-height={1024} />
      <directionalLight position={[-3, 2, -2]} intensity={0.4} />
      <Grid
        args={[24, 24]}
        cellSize={0.25} cellColor="#27313f"
        sectionSize={1} sectionColor="#3b4757"
        infiniteGrid fadeDistance={20} fadeStrength={1.5}
        position={[0, 0, 0]}
      />
      <Humanoid videoRef={videoRef} frames={frames} headFrames={headFrames} H={H} />
      <OrbitControls target={[0, 0.55 * H, 0]} enablePan={false}
        minDistance={H} maxDistance={H * 4.5} />
    </Canvas>
  );
}
