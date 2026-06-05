"use client";

import { Suspense, useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Grid, useFBX, Html } from "@react-three/drei";
import * as THREE from "three";
import { clone as skeletonClone } from "three/examples/jsm/utils/SkeletonUtils.js";
import { computeBodyJoints, quatToEuler, nearestByTime, type BodyJoints } from "@/lib/pose";
import type { HandPoseFrame, HeadPoseFrameT, Landmark } from "@/lib/types";

const MODEL_URL = "/models/ybot.fbx";

// Normalize bone names so we match regardless of how the loader mangles them
// ("mixamorig:LeftArm" / "mixamorigLeftArm" / "LeftArm" all -> "leftarm").
const norm = (s: string) =>
  s.replace(/^mixamorig:?/i, "").replace(/[^a-z0-9]/gi, "").toLowerCase();

// Each driven bone: aim from joint `from` toward joint `to`. `childBone` is the
// rig's next bone, used to read the bind-pose aim direction.
const DRIVEN: { bone: string; childBone: string; from: keyof BodyJoints; to: keyof BodyJoints }[] = [
  { bone: "Spine", childBone: "Spine1", from: "pelvis", to: "chest" },
  { bone: "Neck", childBone: "Head", from: "neck", to: "head" },
  { bone: "LeftArm", childBone: "LeftForeArm", from: "lSh", to: "lElbow" },
  { bone: "LeftForeArm", childBone: "LeftHand", from: "lElbow", to: "lWrist" },
  { bone: "RightArm", childBone: "RightForeArm", from: "rSh", to: "rElbow" },
  { bone: "RightForeArm", childBone: "RightHand", from: "rElbow", to: "rWrist" },
  { bone: "LeftUpLeg", childBone: "LeftLeg", from: "lHip", to: "lKnee" },
  { bone: "LeftLeg", childBone: "LeftFoot", from: "lKnee", to: "lAnk" },
  { bone: "RightUpLeg", childBone: "RightLeg", from: "rHip", to: "rKnee" },
  { bone: "RightLeg", childBone: "RightFoot", from: "rKnee", to: "rAnk" },
];

function Rig({
  videoRef, frames, headFrames, H,
}: {
  videoRef: React.RefObject<HTMLVideoElement>;
  frames: HandPoseFrame[];
  headFrames: HeadPoseFrameT[];
  H: number;
}) {
  const fbx = useFBX(MODEL_URL);
  const model = useMemo(() => skeletonClone(fbx) as THREE.Group, [fbx]);

  const bonesRef = useRef<Record<string, THREE.Bone>>({});
  const restRef = useRef<Record<string, { q: THREE.Quaternion; dir: THREE.Vector3 }>>({});
  const hudRef = useRef<HTMLDivElement>(null);
  const matchedRef = useRef(0);
  const hTimes = useMemo(() => frames.map((f) => f.timestamp_ms), [frames]);
  const kTimes = useMemo(() => headFrames.map((f) => f.timestamp_ms), [headFrames]);
  const lastL = useRef<Landmark[] | null>(null);
  const lastR = useRef<Landmark[] | null>(null);

  const tmpQ = useMemo(() => new THREE.Quaternion(), []);
  const tmpQ2 = useMemo(() => new THREE.Quaternion(), []);
  const tmpV = useMemo(() => new THREE.Vector3(), []);

  useEffect(() => {
    const bones: Record<string, THREE.Bone> = {};
    model.traverse((o: any) => { if (o.isBone) bones[norm(o.name)] = o as THREE.Bone; });
    bonesRef.current = bones;

    model.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(model);
    const size = new THREE.Vector3(); box.getSize(size);
    model.scale.setScalar(H / (size.y || 1));
    model.updateMatrixWorld(true);
    const box2 = new THREE.Box3().setFromObject(model);
    model.position.x -= (box2.min.x + box2.max.x) / 2;
    model.position.z -= (box2.min.z + box2.max.z) / 2;
    model.position.y -= box2.min.y;
    model.updateMatrixWorld(true);

    const rest: Record<string, { q: THREE.Quaternion; dir: THREE.Vector3 }> = {};
    let matched = 0;
    for (const d of DRIVEN) {
      const b = bones[norm(d.bone)], c = bones[norm(d.childBone)];
      if (!b || !c) continue;
      matched++;
      const q = new THREE.Quaternion(); b.getWorldQuaternion(q);
      const pb = new THREE.Vector3(); b.getWorldPosition(pb);
      const pc = new THREE.Vector3(); c.getWorldPosition(pc);
      rest[norm(d.bone)] = { q, dir: pc.sub(pb).normalize() };
    }
    restRef.current = rest;
    matchedRef.current = matched;
    // eslint-disable-next-line no-console
    console.log("[HumanModelFBX] bones:", Object.keys(bones).length,
      "| driven matched:", matched, "/", DRIVEN.length,
      "| sample names:", Object.keys(bones).slice(0, 6));
  }, [model, H]);

  useFrame(() => {
    const bones = bonesRef.current, rest = restRef.current;
    let left = lastL.current, right = lastR.current;
    let headE: { yaw: number; pitch: number; roll: number } | null = null;
    let ms = -1;
    const v = videoRef.current;
    if (v && frames.length) {
      ms = v.currentTime * 1000;
      const hf = nearestByTime(frames, hTimes, ms);
      if (hf?.left_hand_landmarks) { left = hf.left_hand_landmarks; lastL.current = left; }
      if (hf?.right_hand_landmarks) { right = hf.right_hand_landmarks; lastR.current = right; }
      const kf = nearestByTime(headFrames, kTimes, ms);
      if (kf?.tracked) headE = quatToEuler(kf.quaternion);
    }
    const j = computeBodyJoints(H, left, right, headE);

    for (const d of DRIVEN) {
      const b = bones[norm(d.bone)], r = rest[norm(d.bone)];
      if (!b || !r) continue;
      const from = j[d.from], to = j[d.to];
      // Mirror X: our pose space has +X = subject's right, but the Mixamo rig's
      // bind arms point the opposite way. Without this the upper-arm aim is ~180°
      // off, which folds the joint and flings the forearm off.
      tmpV.set(-(to[0] - from[0]), to[1] - from[1], to[2] - from[2]);
      if (tmpV.lengthSq() < 1e-9) continue;
      tmpV.normalize();
      tmpQ.setFromUnitVectors(r.dir, tmpV).multiply(r.q);
      if (b.parent) {
        b.parent.updateWorldMatrix(true, false);
        b.parent.getWorldQuaternion(tmpQ2).invert();
        b.quaternion.copy(tmpQ2.multiply(tmpQ));
      } else {
        b.quaternion.copy(tmpQ);
      }
      b.updateWorldMatrix(false, false);
    }

    if (hudRef.current) {
      hudRef.current.textContent =
        `bones matched ${matchedRef.current}/${DRIVEN.length} · frames ${frames.length} · t=${ms < 0 ? "—" : (ms / 1000).toFixed(1) + "s"}`;
    }
  });

  return (
    <>
      <primitive object={model} />
      <Html position={[0, H * 1.15, 0]} center distanceFactor={H * 6}>
        <div ref={hudRef} style={{
          color: "#9fb0c3", font: "11px monospace", whiteSpace: "nowrap",
          background: "rgba(11,14,19,0.7)", padding: "2px 6px", borderRadius: 4,
        }} />
      </Html>
    </>
  );
}

export default function HumanModelFBX({
  videoRef, frames, headFrames, operatorHeightCm,
}: {
  videoRef: React.RefObject<HTMLVideoElement>;
  frames: HandPoseFrame[];
  headFrames: HeadPoseFrameT[];
  operatorHeightCm: number;
}) {
  const H = (operatorHeightCm || 170) / 100;
  return (
    <Canvas shadows camera={{ position: [0.3 * H, 0.95 * H, 2.4 * H], fov: 42 }}
      style={{ width: "100%", height: "100%" }}>
      <color attach="background" args={["#0b0e13"]} />
      <ambientLight intensity={0.75} />
      <directionalLight position={[3, 6, 4]} intensity={1.1} castShadow
        shadow-mapSize-width={1024} shadow-mapSize-height={1024} />
      <directionalLight position={[-3, 2, -2]} intensity={0.4} />
      <Grid args={[24, 24]} cellSize={0.25} cellColor="#27313f"
        sectionSize={1} sectionColor="#3b4757" infiniteGrid fadeDistance={20} />
      <Suspense fallback={null}>
        <Rig videoRef={videoRef} frames={frames} headFrames={headFrames} H={H} />
      </Suspense>
      <OrbitControls target={[0, 0.55 * H, 0]} enablePan={false}
        minDistance={H} maxDistance={H * 4.5} />
    </Canvas>
  );
}
