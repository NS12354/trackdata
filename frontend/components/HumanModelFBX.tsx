"use client";

import { Suspense, useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Grid, useFBX } from "@react-three/drei";
import * as THREE from "three";
import { clone as skeletonClone } from "three/examples/jsm/utils/SkeletonUtils.js";
import { computeBodyJoints, quatToEuler, nearestByTime, type BodyJoints } from "@/lib/pose";
import type { HandPoseFrame, HeadPoseFrameT, Landmark } from "@/lib/types";

const MODEL_URL = "/models/ybot.fbx";

// Each driven bone: aim it from joint `from` toward joint `to`. `childBone` is the
// rig's own next bone, used to read the bind-pose aim direction.
const DRIVEN: { bone: string; childBone: string; from: keyof BodyJoints; to: keyof BodyJoints }[] = [
  { bone: "mixamorig:Spine", childBone: "mixamorig:Spine1", from: "pelvis", to: "chest" },
  { bone: "mixamorig:Neck", childBone: "mixamorig:Head", from: "neck", to: "head" },
  { bone: "mixamorig:LeftArm", childBone: "mixamorig:LeftForeArm", from: "lSh", to: "lElbow" },
  { bone: "mixamorig:LeftForeArm", childBone: "mixamorig:LeftHand", from: "lElbow", to: "lWrist" },
  { bone: "mixamorig:RightArm", childBone: "mixamorig:RightForeArm", from: "rSh", to: "rElbow" },
  { bone: "mixamorig:RightForeArm", childBone: "mixamorig:RightHand", from: "rElbow", to: "rWrist" },
  { bone: "mixamorig:LeftUpLeg", childBone: "mixamorig:LeftLeg", from: "lHip", to: "lKnee" },
  { bone: "mixamorig:LeftLeg", childBone: "mixamorig:LeftFoot", from: "lKnee", to: "lAnk" },
  { bone: "mixamorig:RightUpLeg", childBone: "mixamorig:RightLeg", from: "rHip", to: "rKnee" },
  { bone: "mixamorig:RightLeg", childBone: "mixamorig:RightFoot", from: "rKnee", to: "rAnk" },
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
  const hTimes = useMemo(() => frames.map((f) => f.timestamp_ms), [frames]);
  const kTimes = useMemo(() => headFrames.map((f) => f.timestamp_ms), [headFrames]);
  const lastL = useRef<Landmark[] | null>(null);
  const lastR = useRef<Landmark[] | null>(null);

  const tmpQ = useMemo(() => new THREE.Quaternion(), []);
  const tmpQ2 = useMemo(() => new THREE.Quaternion(), []);
  const tmpV = useMemo(() => new THREE.Vector3(), []);

  // One-time setup: index bones, scale to operator height, stand on the floor,
  // and capture each driven bone's bind-pose world orientation + aim direction.
  useEffect(() => {
    const bones: Record<string, THREE.Bone> = {};
    model.traverse((o: any) => { if (o.isBone) bones[o.name] = o as THREE.Bone; });
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
    for (const d of DRIVEN) {
      const b = bones[d.bone], c = bones[d.childBone];
      if (!b || !c) continue;
      const q = new THREE.Quaternion(); b.getWorldQuaternion(q);
      const pb = new THREE.Vector3(); b.getWorldPosition(pb);
      const pc = new THREE.Vector3(); c.getWorldPosition(pc);
      rest[d.bone] = { q, dir: pc.sub(pb).normalize() };
    }
    restRef.current = rest;
  }, [model, H]);

  useFrame(() => {
    const bones = bonesRef.current, rest = restRef.current;
    if (!bones["mixamorig:Hips"]) return;

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

    for (const d of DRIVEN) {
      const b = bones[d.bone], r = rest[d.bone];
      if (!b || !r) continue;
      const from = j[d.from], to = j[d.to];
      tmpV.set(to[0] - from[0], to[1] - from[1], to[2] - from[2]);
      if (tmpV.lengthSq() < 1e-9) continue;
      tmpV.normalize();
      // world delta (bind aim -> target aim), composed onto the bind world quat
      tmpQ.setFromUnitVectors(r.dir, tmpV).multiply(r.q);
      // express as local rotation under the (current) parent
      if (b.parent) {
        b.parent.updateWorldMatrix(true, false);
        b.parent.getWorldQuaternion(tmpQ2).invert();
        b.quaternion.copy(tmpQ2.multiply(tmpQ));
      } else {
        b.quaternion.copy(tmpQ);
      }
      b.updateWorldMatrix(false, false);
    }
  });

  return <primitive object={model} />;
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
