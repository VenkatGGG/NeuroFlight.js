# NeuroFlight.js

A browser-based reinforcement learning environment for training autonomous drone navigation. The drone learns to fly through a procedurally generated forest using Proximal Policy Optimization (PPO).

## Overview

NeuroFlight.js implements a "digital twin" architecture where physics simulation runs separately from visualization. This allows training to happen in a Web Worker without blocking the UI, while a 3D view shows the drone's behavior in real-time.

The drone has four motors and must learn to:
- Maintain stable flight
- Navigate toward target positions
- Avoid collisions with trees
- Control altitude appropriately

## Tech Stack

- **React + Vite** - Frontend framework and build tool
- **React Three Fiber** - 3D visualization using Three.js
- **Rapier** - Physics engine (WASM-based, runs in browser)
- **TensorFlow.js** - Neural network training and inference
- **Zustand** - State management
- **SharedArrayBuffer** - High-performance data sharing between worker and main thread

## Architecture

```
src/
  core/
    PhysicsWorld.ts   - Headless Rapier physics simulation
    PPOAgent.ts       - PPO reinforcement learning agent
  workers/
    TrainingWorker.ts - Web Worker for background training
  components/
    Scene.tsx         - React Three Fiber 3D visualization
    Dashboard.tsx     - Training controls and statistics
  store/
    trainingStore.ts  - Zustand store with SharedArrayBuffer sync
```

## How It Works

1. **Physics Simulation**: The drone is simulated using Rapier physics. Motor forces are applied based on the agent's actions, and the environment tracks position, velocity, orientation, and collisions.

2. **Observations**: The agent receives 18 inputs including raycast distances to obstacles, velocity, angular velocity, relative angle to target, and orientation.

3. **Actions**: The agent outputs 4 continuous values representing thrust for each motor. Differential thrust creates torque for turning and tilting.

4. **Rewards**: The agent is rewarded for moving toward the target and penalized for collisions, going out of bounds, or flying too high/low.

5. **Training**: PPO collects trajectories over multiple episodes, computes advantages using Generalized Advantage Estimation (GAE), and updates the policy and value networks.

## Running Locally

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open http://localhost:5173 in your browser.

## Training

1. Click "Start Training" to begin
2. The drone will initially behave randomly
3. Over hundreds of episodes, it should learn to navigate toward targets
4. Use "Pause" to stop training and observe the current policy
5. Training statistics are displayed in the dashboard

## Configuration

Key parameters can be adjusted in the source files:

**PhysicsWorld.ts**
- `TERMINATION_HEIGHT` - Maximum flight altitude
- `TARGET_HEIGHT_RANGE` - Height range for target placement
- Reward scaling factors

**PPOAgent.ts**
- `learningRate` - Neural network learning rate
- `gamma` - Discount factor for future rewards
- `epsilon` - PPO clipping range
- `hiddenLayers` - Network architecture

**TrainingWorker.ts**
- `BATCH_SIZE` - Steps to collect before training
- `maxStepsPerEpisode` - Maximum episode length

## Browser Requirements

This project uses SharedArrayBuffer for efficient data sharing between the training worker and main thread. This requires specific HTTP headers (COOP/COEP) which are configured in vite.config.ts.

Modern browsers (Chrome, Firefox, Edge) support this feature. Safari has limited support.

## License

MIT
