/**
 * trainingStore.ts - Zustand store for training state and drone visualization
 * Uses SharedArrayBuffer for high-performance worker communication
 */
import { create } from 'zustand';

export interface TrainingStats {
  episode: number;
  totalEpisodes: number;
  averageReward: number;
  bestReward: number;
  policyLoss: number;
  valueLoss: number;
  recentRewards: number[];
}

export interface DroneVisualState {
  position: [number, number, number];
  rotation: [number, number, number, number];
  targetPosition: [number, number, number];
}

export interface Obstacle {
  position: [number, number, number];
  radius: number;
  height: number;
}

interface TrainingStore {
  // Training state
  isTraining: boolean;
  isPaused: boolean;
  trainingProgress: number;
  stats: TrainingStats;

  // Visualization state
  isRunning: boolean;
  droneState: DroneVisualState;
  obstacles: Obstacle[];

  // Model weights
  modelWeights: ArrayBuffer | null;

  // SharedArrayBuffer for high-frequency updates
  sharedBuffer: SharedArrayBuffer | null;
  sharedArray: Float32Array | null;

  // Actions
  setTraining: (isTraining: boolean) => void;
  setPaused: (isPaused: boolean) => void;
  setProgress: (progress: number) => void;
  setStats: (stats: Partial<TrainingStats>) => void;
  setRunning: (isRunning: boolean) => void;
  setDroneState: (state: DroneVisualState) => void;
  setObstacles: (obstacles: Obstacle[]) => void;
  setModelWeights: (weights: ArrayBuffer | null) => void;
  initSharedBuffer: () => void;
  readSharedState: () => DroneVisualState;
  reset: () => void;
}

const initialStats: TrainingStats = {
  episode: 0,
  totalEpisodes: 1000,
  averageReward: 0,
  bestReward: -Infinity,
  policyLoss: 0,
  valueLoss: 0,
  recentRewards: [],
};

const initialDroneState: DroneVisualState = {
  position: [0, 2, 0],
  rotation: [0, 0, 0, 1],
  targetPosition: [10, 2, 10],
};

// SharedArrayBuffer layout:
// [0-2]: position (x, y, z)
// [3-6]: rotation (x, y, z, w)
// [7-9]: target position (x, y, z)
// [10]: update flag
const SHARED_BUFFER_SIZE = 11 * Float32Array.BYTES_PER_ELEMENT;

export const useTrainingStore = create<TrainingStore>((set, get) => ({
  isTraining: false,
  isPaused: false,
  trainingProgress: 0,
  stats: initialStats,
  isRunning: false,
  droneState: initialDroneState,
  obstacles: [],
  modelWeights: null,
  sharedBuffer: null,
  sharedArray: null,

  setTraining: (isTraining) => set({ isTraining }),
  setPaused: (isPaused) => set({ isPaused }),
  setProgress: (progress) => set({ trainingProgress: progress }),

  setStats: (newStats) => set((state) => ({
    stats: { ...state.stats, ...newStats },
  })),

  setRunning: (isRunning) => set({ isRunning }),
  setDroneState: (droneState) => set({ droneState }),
  setObstacles: (obstacles) => set({ obstacles }),
  setModelWeights: (modelWeights) => set({ modelWeights }),

  initSharedBuffer: () => {
    if (typeof SharedArrayBuffer === 'undefined') {
      console.warn('SharedArrayBuffer not available. Falling back to postMessage.');
      return;
    }

    const sharedBuffer = new SharedArrayBuffer(SHARED_BUFFER_SIZE);
    const sharedArray = new Float32Array(sharedBuffer);

    // Initialize with default drone state
    sharedArray[0] = 0;  // x
    sharedArray[1] = 2;  // y
    sharedArray[2] = 0;  // z
    sharedArray[3] = 0;  // qx
    sharedArray[4] = 0;  // qy
    sharedArray[5] = 0;  // qz
    sharedArray[6] = 1;  // qw
    sharedArray[7] = 10; // target x
    sharedArray[8] = 2;  // target y
    sharedArray[9] = 10; // target z
    sharedArray[10] = 0; // update flag

    set({ sharedBuffer, sharedArray });
  },

  readSharedState: () => {
    const { sharedArray } = get();
    if (!sharedArray) {
      return get().droneState;
    }

    return {
      position: [sharedArray[0], sharedArray[1], sharedArray[2]] as [number, number, number],
      rotation: [sharedArray[3], sharedArray[4], sharedArray[5], sharedArray[6]] as [number, number, number, number],
      targetPosition: [sharedArray[7], sharedArray[8], sharedArray[9]] as [number, number, number],
    };
  },

  reset: () => set({
    isTraining: false,
    isPaused: false,
    trainingProgress: 0,
    stats: initialStats,
    isRunning: false,
    droneState: initialDroneState,
  }),
}));

// Helper to write drone state to SharedArrayBuffer (used by worker)
export function writeToSharedBuffer(
  sharedArray: Float32Array,
  position: [number, number, number],
  rotation: [number, number, number, number],
  targetPosition: [number, number, number]
): void {
  sharedArray[0] = position[0];
  sharedArray[1] = position[1];
  sharedArray[2] = position[2];
  sharedArray[3] = rotation[0];
  sharedArray[4] = rotation[1];
  sharedArray[5] = rotation[2];
  sharedArray[6] = rotation[3];
  sharedArray[7] = targetPosition[0];
  sharedArray[8] = targetPosition[1];
  sharedArray[9] = targetPosition[2];
  sharedArray[10] = 1; // Set update flag
}
