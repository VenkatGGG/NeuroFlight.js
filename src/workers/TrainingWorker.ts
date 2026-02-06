/**
 * TrainingWorker.ts - Web Worker for headless RL training
 * Runs physics + AI training at max speed without blocking UI
 */
import { PhysicsWorld } from '../core/PhysicsWorld';
import { PPOAgent } from '../core/ppo/Agent';
import { Trajectory } from '../core/ppo/types';
import { writeToSharedBuffer } from '../store/trainingStore';

interface TrainingConfig {
  totalEpisodes: number;
  maxStepsPerEpisode: number;
  stepsPerUpdate: number;
  visualizationMode: boolean;
}

interface WorkerMessage {
  type: 'start' | 'stop' | 'pause' | 'resume' | 'run' | 'loadWeights';
  config?: TrainingConfig;
  sharedBuffer?: SharedArrayBuffer;
  weights?: ArrayBuffer;
}

interface WorkerResponse {
  type: 'progress' | 'complete' | 'error' | 'obstacles' | 'state';
  data?: {
    episode?: number;
    totalEpisodes?: number;
    averageReward?: number;
    bestReward?: number;
    policyLoss?: number;
    valueLoss?: number;
    recentRewards?: number[];
    weights?: ArrayBuffer;
    obstacles?: Array<{ position: [number, number, number]; radius: number; height: number }>;
    droneState?: {
      position: [number, number, number];
      rotation: [number, number, number, number];
      targetPosition: [number, number, number];
    };
  };
  error?: string;
}

let physics: PhysicsWorld | null = null;
let agent: PPOAgent | null = null;
let isTraining = false;
let isPaused = false;
let isRunning = false;
let sharedArray: Float32Array | null = null;

const defaultConfig: TrainingConfig = {
  totalEpisodes: 1000,
  maxStepsPerEpisode: 2000,
  stepsPerUpdate: 2048,
  visualizationMode: false,
};

async function initializeEnvironment(): Promise<void> {
  console.log('Initializing environment...');

  console.log('Creating PhysicsWorld...');
  physics = new PhysicsWorld();
  await physics.init();
  console.log('PhysicsWorld initialized');

  console.log('Creating PPOAgent...');
  agent = new PPOAgent({
    observationSize: 18,
    actionSize: 4,
    hiddenLayers: [256, 128, 64],
    batchSize: 64,
    entropyCoef: 0.01,
  });
  await agent.init();
  console.log('PPOAgent initialized');

  // Send initial obstacles to main thread
  const obstacles = physics.getObstacles();
  console.log('Sending obstacles:', obstacles.length);
  postResponse({
    type: 'obstacles',
    data: { obstacles },
  });
}

async function trainLoop(config: TrainingConfig): Promise<void> {
  console.log('trainLoop started');
  if (!physics || !agent) {
    console.log('Need to initialize environment');
    await initializeEnvironment();
  }
  console.log('Environment ready, starting training loop');

  const recentRewards: number[] = [];
  let bestReward = -Infinity;

  // Batch training: accumulate steps before training
  const BATCH_SIZE = 2048; // Train every N steps
  let batchTrajectory: Trajectory = {
    observations: [],
    actions: [],
    rewards: [],
    values: [],
    logProbs: [],
    dones: [],
  };
  let episodeRewards: number[] = [];
  let lastLosses = { policyLoss: 0, valueLoss: 0 };

  for (let episode = 0; episode < config.totalEpisodes && isTraining; episode++) {
    while (isPaused && isTraining) {
      await sleep(100);
    }

    if (!isTraining) break;

    let state = physics!.reset();
    let episodeReward = 0;
    let done = false;

    for (let step = 0; step < config.maxStepsPerEpisode && !done; step++) {
      const observation = physics!.getObservation();

      // Select action using policy
      const { action, logProb, value } = agent!.selectAction(observation);

      // Apply action to physics
      physics!.applyMotorForces(action);
      physics!.step();

      // Get reward and check termination
      const result = physics!.calculateReward();

      // Store transition in batch
      batchTrajectory.observations.push(observation);
      batchTrajectory.actions.push(action);
      batchTrajectory.rewards.push(result.reward);
      batchTrajectory.values.push(value);
      batchTrajectory.logProbs.push(logProb);
      batchTrajectory.dones.push(result.done);

      episodeReward += result.reward;
      done = result.done;
      state = physics!.getState();

      // Update shared buffer for visualization (every 10 steps to reduce overhead)
      if (sharedArray && step % 10 === 0) {
        writeToSharedBuffer(
          sharedArray,
          state.position,
          state.rotation,
          physics!.getTargetPosition()
        );
      }
    }

    episodeRewards.push(episodeReward);
    recentRewards.push(episodeReward);
    if (recentRewards.length > 100) {
      recentRewards.shift();
    }
    if (episodeReward > bestReward) {
      bestReward = episodeReward;
    }

    // Train when we have enough steps (batch training)
    if (batchTrajectory.observations.length >= BATCH_SIZE) {
      lastLosses = await agent!.train(batchTrajectory);

      // Reset batch
      batchTrajectory = {
        observations: [],
        actions: [],
        rewards: [],
        values: [],
        logProbs: [],
        dones: [],
      };
      episodeRewards = [];
    }

    // Report progress every 5 episodes
    if (episode % 5 === 0) {
      const averageReward = recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length;
      postResponse({
        type: 'progress',
        data: {
          episode: episode + 1,
          totalEpisodes: config.totalEpisodes,
          averageReward,
          bestReward,
          policyLoss: lastLosses.policyLoss,
          valueLoss: lastLosses.valueLoss,
          recentRewards: [...recentRewards],
        },
      });
    }

    // Allow other operations to process
    if (episode % 5 === 0) {
      await sleep(0);
    }
  }

  // Train on any remaining data
  if (batchTrajectory.observations.length > 0) {
    await agent!.train(batchTrajectory);
  }

  // Training complete - send final weights
  if (agent && isTraining) {
    const weights = await agent.saveWeights();
    postResponse({
      type: 'complete',
      data: {
        weights,
        episode: config.totalEpisodes,
        totalEpisodes: config.totalEpisodes,
        averageReward: recentRewards.reduce((a, b) => a + b, 0) / recentRewards.length,
        bestReward,
        recentRewards: [...recentRewards],
      },
    });
  }

  isTraining = false;
}

async function runLoop(): Promise<void> {
  if (!physics || !agent) {
    postResponse({ type: 'error', error: 'Environment not initialized' });
    return;
  }

  let state = physics.reset();

  // Send initial state
  postResponse({
    type: 'obstacles',
    data: { obstacles: physics.getObstacles() },
  });

  const targetFPS = 60;
  const frameTime = 1000 / targetFPS;
  let lastTime = performance.now();

  while (isRunning) {
    const now = performance.now();
    const delta = now - lastTime;

    if (delta >= frameTime) {
      lastTime = now;

      const observation = physics.getObservation();
      const { action } = agent.selectAction(observation, true); // Deterministic for inference

      physics.applyMotorForces(action);
      physics.step();

      const result = physics.calculateReward();
      state = physics.getState();

      // Update shared buffer
      if (sharedArray) {
        writeToSharedBuffer(
          sharedArray,
          state.position,
          state.rotation,
          physics.getTargetPosition()
        );
      } else {
        // Fallback to postMessage
        postResponse({
          type: 'state',
          data: {
            droneState: {
              position: state.position,
              rotation: state.rotation,
              targetPosition: physics.getTargetPosition(),
            },
          },
        });
      }

      // Reset if episode ended
      if (result.done) {
        state = physics.reset();
        postResponse({
          type: 'obstacles',
          data: { obstacles: physics.getObstacles() },
        });
      }
    }

    await sleep(0);
  }
}

function postResponse(response: WorkerResponse): void {
  self.postMessage(response);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Message handler
self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  console.log('Worker received message:', event.data);
  const { type, config, sharedBuffer, weights } = event.data;

  switch (type) {
    case 'start':
      console.log('Starting training...');
      if (isTraining) {
        console.log('Already training, ignoring');
        return;
      }
      isTraining = true;
      isPaused = false;

      if (sharedBuffer) {
        sharedArray = new Float32Array(sharedBuffer);
      }

      try {
        console.log('Calling trainLoop with config:', config || defaultConfig);
        await trainLoop(config || defaultConfig);
      } catch (error) {
        console.error('Training error:', error);
        postResponse({
          type: 'error',
          error: error instanceof Error ? error.message : 'Training failed',
        });
        isTraining = false;
      }
      break;

    case 'stop':
      isTraining = false;
      isRunning = false;
      break;

    case 'pause':
      isPaused = true;
      break;

    case 'resume':
      isPaused = false;
      break;

    case 'run':
      if (isRunning) return;
      isRunning = true;
      isTraining = false;

      if (sharedBuffer) {
        sharedArray = new Float32Array(sharedBuffer);
      }

      try {
        if (!physics || !agent) {
          await initializeEnvironment();
        }
        await runLoop();
      } catch (error) {
        postResponse({
          type: 'error',
          error: error instanceof Error ? error.message : 'Run failed',
        });
        isRunning = false;
      }
      break;

    case 'loadWeights':
      try {
        if (!agent) {
          await initializeEnvironment();
        }
        if (weights && agent) {
          await agent.loadWeights(weights);
          postResponse({ type: 'complete', data: {} });
        }
      } catch (error) {
        postResponse({
          type: 'error',
          error: error instanceof Error ? error.message : 'Failed to load weights',
        });
      }
      break;
  }
};

export {};
