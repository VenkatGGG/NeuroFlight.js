/**
 * types.ts - Shared types for PPO Agent
 */

export interface PPOConfig {
  observationSize: number;
  actionSize: number;
  hiddenLayers: number[];
  learningRate: number;
  gamma: number;
  epsilon: number; // PPO clip range
  valueCoef: number;
  entropyCoef: number;
  batchSize: number;
  epochs: number;
  maxGradNorm: number; // Gradient clipping
  valueClipRange: number; // Value function clipping
}

export const DEFAULT_CONFIG: PPOConfig = {
  observationSize: 18,
  actionSize: 4,
  hiddenLayers: [256, 128, 64],
  learningRate: 0.0003,
  gamma: 0.99,
  epsilon: 0.2,
  valueCoef: 0.5,
  entropyCoef: 0.01,
  batchSize: 64,
  epochs: 4,
  maxGradNorm: 0.5,
  valueClipRange: 10.0,
};

export interface Trajectory {
  observations: number[][];
  actions: number[][];
  rewards: number[];
  values: number[];
  logProbs: number[];
  dones: boolean[];
}
