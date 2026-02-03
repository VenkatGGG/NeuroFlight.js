/**
 * Scene.tsx - React Three Fiber visualization
 */
import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Sky, Environment } from '@react-three/drei';
import * as THREE from 'three';
import { useTrainingStore, Obstacle } from '../store/trainingStore';

function Drone() {
  const meshRef = useRef<THREE.Group>(null);
  const readSharedState = useTrainingStore((state) => state.readSharedState);
  const sharedArray = useTrainingStore((state) => state.sharedArray);
  const droneState = useTrainingStore((state) => state.droneState);

  useFrame(() => {
    if (!meshRef.current) return;

    // Read from shared buffer if available, otherwise use store state
    const state = sharedArray ? readSharedState() : droneState;

    meshRef.current.position.set(...state.position);
    meshRef.current.quaternion.set(...state.rotation);
  });

  return (
    <group ref={meshRef}>
      {/* Main body */}
      <mesh castShadow>
        <boxGeometry args={[1, 0.15, 1]} />
        <meshStandardMaterial color="#1a1a2e" metalness={0.8} roughness={0.2} />
      </mesh>

      {/* Arms */}
      {[
        [0.4, 0, 0.4],
        [-0.4, 0, 0.4],
        [-0.4, 0, -0.4],
        [0.4, 0, -0.4],
      ].map((pos, i) => (
        <group key={i} position={pos as [number, number, number]}>
          {/* Arm */}
          <mesh castShadow>
            <cylinderGeometry args={[0.03, 0.03, 0.3, 8]} />
            <meshStandardMaterial color="#333" metalness={0.5} roughness={0.5} />
          </mesh>
          {/* Motor */}
          <mesh position={[0, 0.15, 0]} castShadow>
            <cylinderGeometry args={[0.08, 0.08, 0.08, 16]} />
            <meshStandardMaterial color="#0f4c75" metalness={0.7} roughness={0.3} />
          </mesh>
          {/* Propeller */}
          <mesh position={[0, 0.2, 0]} rotation={[0, i * 0.5, 0]}>
            <boxGeometry args={[0.35, 0.01, 0.04]} />
            <meshStandardMaterial
              color={i % 2 === 0 ? '#3282b8' : '#bbe1fa'}
              transparent
              opacity={0.7}
            />
          </mesh>
        </group>
      ))}

      {/* LED indicators */}
      <pointLight position={[0.4, -0.1, 0.4]} color="#00ff00" intensity={0.5} distance={2} />
      <pointLight position={[-0.4, -0.1, 0.4]} color="#00ff00" intensity={0.5} distance={2} />
      <pointLight position={[-0.4, -0.1, -0.4]} color="#ff0000" intensity={0.5} distance={2} />
      <pointLight position={[0.4, -0.1, -0.4]} color="#ff0000" intensity={0.5} distance={2} />
    </group>
  );
}

function Target() {
  const meshRef = useRef<THREE.Mesh>(null);
  const readSharedState = useTrainingStore((state) => state.readSharedState);
  const sharedArray = useTrainingStore((state) => state.sharedArray);
  const droneState = useTrainingStore((state) => state.droneState);

  useFrame((_, delta) => {
    if (!meshRef.current) return;

    const state = sharedArray ? readSharedState() : droneState;
    meshRef.current.position.set(...state.targetPosition);
    meshRef.current.rotation.y += delta * 2;
  });

  return (
    <mesh ref={meshRef}>
      <octahedronGeometry args={[0.5, 0]} />
      <meshStandardMaterial
        color="#ffd700"
        emissive="#ffa500"
        emissiveIntensity={0.5}
        metalness={0.8}
        roughness={0.2}
      />
    </mesh>
  );
}

function Tree({ position, radius, height }: Obstacle) {
  const trunkHeight = height * 0.3;
  const foliageHeight = height * 0.7;

  return (
    <group position={position}>
      {/* Trunk */}
      <mesh position={[0, -height / 2 + trunkHeight / 2, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[radius * 0.3, radius * 0.4, trunkHeight, 8]} />
        <meshStandardMaterial color="#4a3728" roughness={0.9} />
      </mesh>

      {/* Foliage layers */}
      {[0, 0.3, 0.6].map((offset, i) => (
        <mesh
          key={i}
          position={[0, -height / 2 + trunkHeight + foliageHeight * offset, 0]}
          castShadow
        >
          <coneGeometry
            args={[radius * (1 - offset * 0.3), foliageHeight * (0.5 - offset * 0.1), 8]}
          />
          <meshStandardMaterial
            color={`hsl(${120 + i * 10}, ${60 - i * 5}%, ${25 + i * 5}%)`}
            roughness={0.8}
          />
        </mesh>
      ))}
    </group>
  );
}

function Forest() {
  const obstacles = useTrainingStore((state) => state.obstacles);

  return (
    <group>
      {obstacles.map((obstacle, i) => (
        <Tree key={i} {...obstacle} />
      ))}
    </group>
  );
}

function Ground() {
  return (
    <>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.05, 0]} receiveShadow>
        <planeGeometry args={[100, 100]} />
        <meshStandardMaterial color="#2d5016" roughness={1} />
      </mesh>
      <Grid
        position={[0, 0.01, 0]}
        args={[100, 100]}
        cellSize={1}
        cellThickness={0.5}
        cellColor="#3a6b1e"
        sectionSize={10}
        sectionThickness={1}
        sectionColor="#4a8b2e"
        fadeDistance={50}
        fadeStrength={1}
        followCamera={false}
      />
    </>
  );
}

function Raycasts() {
  const readSharedState = useTrainingStore((state) => state.readSharedState);
  const sharedArray = useTrainingStore((state) => state.sharedArray);
  const droneState = useTrainingStore((state) => state.droneState);
  const isRunning = useTrainingStore((state) => state.isRunning);

  const groupRef = useRef<THREE.Group>(null);
  const linesRef = useRef<THREE.Line[]>([]);
  const angles = useMemo(() => [0, 30, -30, 60, -60, 90, -90], []);

  // Create line geometries
  const lineGeometries = useMemo(() => {
    return angles.map(() => {
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(6); // 2 points * 3 coords
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      return geometry;
    });
  }, [angles]);

  useFrame(() => {
    if (!isRunning) return;

    const state = sharedArray ? readSharedState() : droneState;
    const [px, py, pz] = state.position;

    linesRef.current.forEach((line, i) => {
      if (!line) return;

      const angle = (angles[i] * Math.PI) / 180;
      const range = 10;
      const endX = px + Math.sin(angle) * range;
      const endZ = pz + Math.cos(angle) * range;

      const positions = line.geometry.attributes.position as THREE.BufferAttribute;
      positions.setXYZ(0, px, py, pz);
      positions.setXYZ(1, endX, py, endZ);
      positions.needsUpdate = true;
    });
  });

  return (
    <group ref={groupRef}>
      {lineGeometries.map((geometry, i) => (
        <primitive
          key={i}
          object={(() => {
            const material = new THREE.LineBasicMaterial({
              color: '#ff6b6b',
              opacity: 0.5,
              transparent: true,
            });
            const line = new THREE.Line(geometry, material);
            linesRef.current[i] = line;
            return line;
          })()}
        />
      ))}
    </group>
  );
}

export function Scene() {
  return (
    <Canvas
      shadows
      camera={{ position: [15, 15, 15], fov: 60 }}
      style={{ background: 'linear-gradient(to bottom, #1a1a2e, #16213e)' }}
    >
      <Sky sunPosition={[100, 50, 100]} turbidity={0.3} rayleigh={0.5} />
      <Environment preset="forest" />

      <ambientLight intensity={0.4} />
      <directionalLight
        position={[50, 50, 25]}
        intensity={1}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-far={100}
        shadow-camera-left={-50}
        shadow-camera-right={50}
        shadow-camera-top={50}
        shadow-camera-bottom={-50}
      />

      <Ground />
      <Forest />
      <Drone />
      <Target />
      <Raycasts />

      <OrbitControls
        enablePan
        enableZoom
        enableRotate
        maxPolarAngle={Math.PI / 2.1}
        minDistance={5}
        maxDistance={100}
      />
    </Canvas>
  );
}
