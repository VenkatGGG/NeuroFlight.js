/**
 * PhysicsWorld.ts - Headless Rapier physics simulation
 * NO Three.js imports - runs in Web Worker
 */
import RAPIER from '@dimforge/rapier3d-compat';

export interface DroneState {
  position: [number, number, number];
  rotation: [number, number, number, number]; // quaternion
  velocity: [number, number, number];
  angularVelocity: [number, number, number];
  raycastDistances: number[]; // 7 normalized distances [0-1]
}

export interface Obstacle {
  position: [number, number, number];
  radius: number;
  height: number;
}

export interface PhysicsConfig {
  gravity: number;
  droneSize: number;
  motorOffset: number;
  maxMotorForce: number;
  raycastRange: number;
  numRaycasts: number;
  worldSize: number;
  numObstacles: number;
}

const DEFAULT_CONFIG: PhysicsConfig = {
  gravity: -9.81,
  droneSize: 0.5,
  motorOffset: 0.3,
  maxMotorForce: 6.0, // Increased - hover at ~50% thrust
  raycastRange: 10,
  numRaycasts: 7,
  worldSize: 50,
  numObstacles: 20,
};

const TERMINATION_HEIGHT = 30; // Very high ceiling - lots of room to learn
const IDEAL_FLIGHT_HEIGHT = 5; // Mid-level through trees
const TARGET_HEIGHT_RANGE = [4, 7]; // Targets around ideal height

export class PhysicsWorld {
  private world!: RAPIER.World;
  private drone!: RAPIER.RigidBody;
  private droneCollider!: RAPIER.Collider;
  private obstacles: Obstacle[] = [];
  private obstacleColliders: RAPIER.Collider[] = [];
  private config: PhysicsConfig;
  private targetPosition: [number, number, number] = [10, 2, 10];
  private hasCollided = false;
  private stepCount = 0;
  private prevDistanceToTarget = 0;

  constructor(config: Partial<PhysicsConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  async init(): Promise<void> {
    await RAPIER.init();
    this.createWorld();
  }

  private createWorld(): void {
    this.world = new RAPIER.World({ x: 0, y: this.config.gravity, z: 0 });

    this.createGround();
    this.createDrone();
    this.generateObstacles();
    this.randomizeTarget();

    this.prevDistanceToTarget = this.getDistanceToTarget();
  }

  private createGround(): void {
    const groundDesc = RAPIER.RigidBodyDesc.fixed();
    const ground = this.world.createRigidBody(groundDesc);

    const groundColliderDesc = RAPIER.ColliderDesc.cuboid(
      this.config.worldSize,
      0.1,
      this.config.worldSize
    );
    this.world.createCollider(groundColliderDesc, ground);
  }

  private createDrone(): void {
    const droneDesc = RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(0, 2, 0)
      .setLinearDamping(1.5) // Moderate drag
      .setAngularDamping(4.0); // Balanced - some tilt allowed but no crazy flipping

    this.drone = this.world.createRigidBody(droneDesc);

    // Main body collider
    const colliderDesc = RAPIER.ColliderDesc.cuboid(
      this.config.droneSize,
      this.config.droneSize * 0.3,
      this.config.droneSize
    ).setDensity(1.0);

    this.droneCollider = this.world.createCollider(colliderDesc, this.drone);
  }

  private generateObstacles(): void {
    this.obstacles = [];
    this.obstacleColliders = [];

    for (let i = 0; i < this.config.numObstacles; i++) {
      const x = (Math.random() - 0.5) * this.config.worldSize * 0.8;
      const z = (Math.random() - 0.5) * this.config.worldSize * 0.8;
      const radius = 0.5 + Math.random() * 1.5;
      const height = 3 + Math.random() * 8;

      // Don't place obstacles too close to spawn
      if (Math.abs(x) < 3 && Math.abs(z) < 3) continue;

      const obstacleDesc = RAPIER.RigidBodyDesc.fixed().setTranslation(x, height / 2, z);
      const obstacle = this.world.createRigidBody(obstacleDesc);

      const colliderDesc = RAPIER.ColliderDesc.cylinder(height / 2, radius);
      const collider = this.world.createCollider(colliderDesc, obstacle);

      this.obstacles.push({ position: [x, height / 2, z], radius, height });
      this.obstacleColliders.push(collider);
    }
  }

  private randomizeTarget(): void {
    const angle = Math.random() * Math.PI * 2;
    // Closer targets for easier learning (8-15m instead of 15-35m)
    const distance = 8 + Math.random() * 7;
    // Target height within tree canopy level
    const targetHeight =
      TARGET_HEIGHT_RANGE[0] + Math.random() * (TARGET_HEIGHT_RANGE[1] - TARGET_HEIGHT_RANGE[0]);
    this.targetPosition = [Math.cos(angle) * distance, targetHeight, Math.sin(angle) * distance];
  }

  applyMotorForces(thrusts: number[]): void {
    const rotation = this.drone.rotation();

    // Calculate total thrust and tilt from motor differences
    const t0 = Math.max(0, Math.min(1, thrusts[0])); // Front-Right
    const t1 = Math.max(0, Math.min(1, thrusts[1])); // Front-Left
    const t2 = Math.max(0, Math.min(1, thrusts[2])); // Back-Left
    const t3 = Math.max(0, Math.min(1, thrusts[3])); // Back-Right

    // Total thrust magnitude
    const totalThrust = ((t0 + t1 + t2 + t3) / 4) * this.config.maxMotorForce;

    // Pitch control: front vs back (positive = nose down)
    const pitchDiff = (t2 + t3 - (t0 + t1)) / 4;
    // Roll control: right vs left (positive = roll right)
    const rollDiff = (t1 + t2 - (t0 + t3)) / 4;
    // Yaw control: diagonal pairs
    const yawDiff = (t0 + t2 - (t1 + t3)) / 4;

    // CRITICAL: Apply thrust in drone's LOCAL up direction
    // When drone tilts, thrust vector tilts with it → horizontal movement!
    const localUp: [number, number, number] = [0, 1, 0];
    const worldThrustDir = this.rotateVector(localUp, rotation);

    const thrustImpulse = {
      x: worldThrustDir[0] * totalThrust * 0.016,
      y: worldThrustDir[1] * totalThrust * 0.016,
      z: worldThrustDir[2] * totalThrust * 0.016,
    };
    this.drone.applyImpulse(thrustImpulse, true);

    // Apply torques for attitude control - balanced
    const torqueScale = 0.08;
    this.drone.applyTorqueImpulse(
      {
        x: pitchDiff * torqueScale, // Pitch
        y: yawDiff * torqueScale * 0.3, // Yaw (less)
        z: rollDiff * torqueScale, // Roll
      },
      true
    );

    // Auto-stabilization - helps prevent flipping while allowing controlled tilt
    const angVel = this.drone.angvel();
    const stabilization = 0.01;
    this.drone.applyTorqueImpulse(
      {
        x: -angVel.x * stabilization,
        y: -angVel.y * stabilization * 0.5,
        z: -angVel.z * stabilization,
      },
      true
    );
  }

  private rotateVector(
    v: [number, number, number],
    q: { x: number; y: number; z: number; w: number }
  ): [number, number, number] {
    // Quaternion rotation of vector
    const qx = q.x,
      qy = q.y,
      qz = q.z,
      qw = q.w;
    const vx = v[0],
      vy = v[1],
      vz = v[2];

    const ix = qw * vx + qy * vz - qz * vy;
    const iy = qw * vy + qz * vx - qx * vz;
    const iz = qw * vz + qx * vy - qy * vx;
    const iw = -qx * vx - qy * vy - qz * vz;

    return [
      ix * qw + iw * -qx + iy * -qz - iz * -qy,
      iy * qw + iw * -qy + iz * -qx - ix * -qz,
      iz * qw + iw * -qz + ix * -qy - iy * -qx,
    ];
  }

  performRaycasts(): number[] {
    const distances: number[] = [];
    const dronePos = this.drone.translation();
    const rotation = this.drone.rotation();

    // 7 rays: forward, ±30°, ±60°, ±90°
    const angles = [0, 30, -30, 60, -60, 90, -90];

    for (const angleDeg of angles) {
      const angle = (angleDeg * Math.PI) / 180;

      // Local direction
      const localDir: [number, number, number] = [Math.sin(angle), 0, Math.cos(angle)];

      // Transform to world space
      const worldDir = this.rotateVector(localDir, rotation);

      const ray = new RAPIER.Ray(
        { x: dronePos.x, y: dronePos.y, z: dronePos.z },
        { x: worldDir[0], y: worldDir[1], z: worldDir[2] }
      );

      const hit = this.world.castRay(
        ray,
        this.config.raycastRange,
        true,
        undefined,
        undefined,
        this.droneCollider // Exclude drone
      );

      // Normalize distance [0-1], 1 = no obstacle
      const normalizedDist = hit ? hit.timeOfImpact / this.config.raycastRange : 1.0;

      distances.push(normalizedDist);
    }

    return distances;
  }

  step(): void {
    this.world.step();
    this.stepCount++;
    this.checkCollisions();
  }

  private checkCollisions(): void {
    this.world.contactPairsWith(this.droneCollider, (otherCollider: RAPIER.Collider) => {
      // Check if we hit an obstacle (not ground)
      if (this.obstacleColliders.includes(otherCollider)) {
        this.hasCollided = true;
      }
    });

    // Check ground collision (drone too low)
    const pos = this.drone.translation();
    if (pos.y < 0.3) {
      this.hasCollided = true;
    }
  }

  getState(): DroneState {
    const pos = this.drone.translation();
    const rot = this.drone.rotation();
    const vel = this.drone.linvel();
    const angVel = this.drone.angvel();
    const raycastDistances = this.performRaycasts();

    return {
      position: [pos.x, pos.y, pos.z],
      rotation: [rot.x, rot.y, rot.z, rot.w],
      velocity: [vel.x, vel.y, vel.z],
      angularVelocity: [angVel.x, angVel.y, angVel.z],
      raycastDistances,
    };
  }

  getObservation(): number[] {
    const state = this.getState();
    const dronePos = state.position;

    // Calculate angle and distance to target
    const dx = this.targetPosition[0] - dronePos[0];
    const dz = this.targetPosition[2] - dronePos[2];
    const horizontalDist = Math.sqrt(dx * dx + dz * dz);
    const angleToTarget = Math.atan2(dx, dz);

    // Get drone's forward angle from quaternion
    const rot = state.rotation;
    const forwardAngle = Math.atan2(
      2 * (rot[3] * rot[1] + rot[0] * rot[2]),
      1 - 2 * (rot[1] * rot[1] + rot[2] * rot[2])
    );

    // Relative angle to target (normalized to [-1, 1])
    let relativeAngle = angleToTarget - forwardAngle;
    while (relativeAngle > Math.PI) relativeAngle -= 2 * Math.PI;
    while (relativeAngle < -Math.PI) relativeAngle += 2 * Math.PI;

    // Tilt angles (pitch and roll) - normalized
    const pitch = Math.asin(2 * (rot[3] * rot[0] - rot[2] * rot[1]));
    const roll = Math.atan2(
      2 * (rot[3] * rot[2] + rot[1] * rot[0]),
      1 - 2 * (rot[0] * rot[0] + rot[2] * rot[2])
    );

    // Height error from ideal (negative = too low, positive = too high)
    const heightError = (dronePos[1] - IDEAL_FLIGHT_HEIGHT) / 5.0;

    return [
      // 7 raycast distances (already normalized 0-1)
      ...state.raycastDistances,
      // Velocity (normalized, assume max ~10 m/s)
      state.velocity[0] / 10,
      state.velocity[1] / 10, // Vertical velocity - important signal!
      state.velocity[2] / 10,
      // Angular velocity (normalized)
      state.angularVelocity[0] / 5,
      state.angularVelocity[1] / 5,
      state.angularVelocity[2] / 5,
      // Target info
      relativeAngle / Math.PI, // -1 to 1
      Math.min(horizontalDist / 20, 1), // normalized distance (closer targets now)
      // Height info (critical for learning altitude control)
      heightError, // -1 to 1 ish, 0 = ideal height
      // Orientation
      pitch / (Math.PI / 2),
      roll / (Math.PI / 2),
    ];
  }

  private getDistanceToTarget(): number {
    // Return full 3D distance to target
    const pos = this.drone.translation();
    const dx = this.targetPosition[0] - pos.x;
    const dy = this.targetPosition[1] - pos.y;
    const dz = this.targetPosition[2] - pos.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  calculateReward(): { reward: number; done: boolean; info: Record<string, number> } {
    const state = this.getState();
    const pos = state.position;

    // === TERMINATION CONDITIONS (only hard failures) ===
    if (this.hasCollided) {
      return { reward: -20.0, done: true, info: { collision: 1 } };
    }

    // Out of bounds (horizontal)
    if (Math.abs(pos[0]) > this.config.worldSize || Math.abs(pos[2]) > this.config.worldSize) {
      return { reward: -10.0, done: true, info: { outOfBounds: 1 } };
    }

    // Too high - terminate at 30m ceiling
    if (pos[1] > TERMINATION_HEIGHT) {
      return { reward: -10.0, done: true, info: { tooHigh: 1 } };
    }

    // Too low - crashed into ground
    if (pos[1] < 0.3) {
      return { reward: -10.0, done: true, info: { crashed: 1 } };
    }

    // Calculate 3D distance to target
    const dx = this.targetPosition[0] - pos[0];
    const dy = this.targetPosition[1] - pos[1];
    const dz = this.targetPosition[2] - pos[2];
    const distToTarget = Math.sqrt(dx * dx + dy * dy + dz * dz);

    // Reached target
    if (distToTarget < 3.0) {
      return { reward: 10.0, done: true, info: { reachedTarget: 1 } };
    }

    // === SCALED REWARDS - PPO-friendly range ===
    let reward = 0;

    // 1. DISTANCE TO TARGET - scaled down for stable learning
    const fullDist = distToTarget;
    const distImprovement = this.prevDistanceToTarget - fullDist;
    reward += distImprovement * 2.0; // Scaled down from 20x
    this.prevDistanceToTarget = fullDist;

    // 2. PROXIMITY BONUS - bigger reward when close to target
    if (fullDist < 10) {
      reward += (10 - fullDist) * 0.1; // Increased: up to +1.0 when very close
    }

    // 3. SURVIVAL BONUS - Slightly increased
    reward += 0.01;

    // 4. ANGULAR VELOCITY PENALTY (Stability)
    // Penalize spinning too fast
    const angVel = state.angularVelocity;
    const spinMagnitude = Math.sqrt(angVel[0] ** 2 + angVel[1] ** 2 + angVel[2] ** 2);
    reward -= spinMagnitude * 0.005;

    // 5. ORIENTATION PENALTY (Uprightness)
    // Penalize being upside down or tilted too much
    // dot product of local up (0,1,0) rotated to world vs world up (0,1,0)
    // We can use the pitch/roll from getObservation logic or just check Y component of rotated up vector
    // Simple version: penalize large pitch/roll
    const pitch = Math.abs(state.rotation[0]); // Approximation from quaternion if small angles
    const roll = Math.abs(state.rotation[2]);
    if (pitch > 0.5 || roll > 0.5) {
       reward -= 0.05;
    }

    // Clip reward to stable range
    reward = Math.max(-10, Math.min(10, reward));

    // Max steps check
    const done = this.stepCount >= 2000;

    return {
      reward,
      done,
      info: {
        distToTarget: fullDist,
        height: pos[1],
      },
    };
  }

  reset(): DroneState {
    // Reset drone position and velocity
    this.drone.setTranslation({ x: 0, y: 2, z: 0 }, true);
    this.drone.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    this.drone.setLinvel({ x: 0, y: 0, z: 0 }, true);
    this.drone.setAngvel({ x: 0, y: 0, z: 0 }, true);

    this.hasCollided = false;
    this.stepCount = 0;
    this.randomizeTarget();
    this.prevDistanceToTarget = this.getDistanceToTarget();

    return this.getState();
  }

  getTargetPosition(): [number, number, number] {
    return [...this.targetPosition];
  }

  getObstacles(): Obstacle[] {
    return [...this.obstacles];
  }

  getStepCount(): number {
    return this.stepCount;
  }

  destroy(): void {
    this.world.free();
  }
}
