/**
 * Agent.ts - Proximal Policy Optimization agent
 */
import * as tf from '@tensorflow/tfjs';
import { PPOConfig, DEFAULT_CONFIG, Trajectory } from './types';
import { NetworkFactory } from './Network';

export class PPOAgent {
  private config: PPOConfig;
  private policyNetwork!: tf.Sequential;
  private valueNetwork!: tf.Sequential;
  private policyOptimizer!: tf.Optimizer;
  private valueOptimizer!: tf.Optimizer;
  private logStd: number[];

  constructor(config: Partial<PPOConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.logStd = new Array(this.config.actionSize).fill(-0.5);
  }

  async init(): Promise<void> {
    this.policyNetwork = NetworkFactory.createPolicyNetwork(
      this.config.observationSize,
      this.config.actionSize,
      this.config.hiddenLayers
    );

    this.valueNetwork = NetworkFactory.createValueNetwork(
      this.config.observationSize,
      this.config.hiddenLayers
    );

    this.policyOptimizer = tf.train.adam(this.config.learningRate);
    this.valueOptimizer = tf.train.adam(this.config.learningRate);
  }

  private gaussianLogProb(
    actions: tf.Tensor,
    means: tf.Tensor,
    logStd: number[]
  ): tf.Tensor {
    const std = logStd.map(Math.exp);
    const variance = std.map((s) => s * s);

    return tf.tidy(() => {
      const diff = actions.sub(means);
      const diffSquared = diff.square();
      const varianceTensor = tf.tensor1d(variance);
      const logStdTensor = tf.tensor1d(logStd);

      // -0.5 * (log(2*pi) + 2*logStd + (x-mean)^2/var)
      const logProb = tf.scalar(-0.5).mul(
        tf.scalar(Math.log(2 * Math.PI))
          .add(logStdTensor.mul(2))
          .add(diffSquared.div(varianceTensor))
      );

      return logProb.sum(-1);
    });
  }

  selectAction(observation: number[], deterministic = false): {
    action: number[];
    logProb: number;
    value: number;
  } {
    return tf.tidy(() => {
      const obsTensor = tf.tensor2d([observation]);
      const actionMean = this.policyNetwork.predict(obsTensor) as tf.Tensor;

      let action: tf.Tensor;
      let logProb: number;

      if (deterministic) {
        action = actionMean;
        logProb = 0;
      } else {
        const std = this.logStd.map(Math.exp);
        const noise = tf.randomNormal([1, this.config.actionSize]);
        const stdTensor = tf.tensor2d([std]);
        action = actionMean.add(noise.mul(stdTensor));

        const logProbTensor = this.gaussianLogProb(action, actionMean, this.logStd);
        logProb = logProbTensor.dataSync()[0];
      }

      const scaledAction = action.add(1).div(2).clipByValue(0, 1);
      const value = this.valueNetwork.predict(obsTensor) as tf.Tensor;

      return {
        action: Array.from(scaledAction.dataSync()),
        logProb,
        value: value.dataSync()[0],
      };
    });
  }

  async train(trajectory: Trajectory): Promise<{ policyLoss: number; valueLoss: number }> {
    const { observations, actions, rewards, values, logProbs, dones } = trajectory;

    if (observations.length === 0) {
      return { policyLoss: 0, valueLoss: 0 };
    }

    const advantages = this.computeGAE(rewards, values, dones);
    const returns = advantages.map((adv, i) => adv + values[i]);

    // Normalize advantages
    const advMean = advantages.reduce((a, b) => a + b, 0) / advantages.length;
    const advVariance = advantages.reduce((sum, a) => sum + (a - advMean) ** 2, 0) / advantages.length;
    const advStd = Math.sqrt(advVariance) + 1e-8;
    const normalizedAdvantages = advantages.map((a) => (a - advMean) / advStd);

    let totalPolicyLoss = 0;
    let totalValueLoss = 0;
    let updateCount = 0;

    for (let epoch = 0; epoch < this.config.epochs; epoch++) {
      const indices = Array.from({ length: observations.length }, (_, i) => i);
      // Simple shuffle
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      for (let i = 0; i < observations.length; i += this.config.batchSize) {
        const batchIndices = indices.slice(i, Math.min(i + this.config.batchSize, observations.length));
        if (batchIndices.length === 0) continue;

        const batchObs = batchIndices.map((idx) => observations[idx]);
        const batchActions = batchIndices.map((idx) => actions[idx]);
        const batchAdvantages = batchIndices.map((idx) => normalizedAdvantages[idx]);
        const batchReturns = batchIndices.map((idx) => returns[idx]);
        const batchOldLogProbs = batchIndices.map((idx) => logProbs[idx]);
        const batchOldValues = batchIndices.map((idx) => values[idx]);

        const losses = await this.trainStep(
          batchObs,
          batchActions,
          batchAdvantages,
          batchReturns,
          batchOldLogProbs,
          batchOldValues
        );

        totalPolicyLoss += losses.policyLoss;
        totalValueLoss += losses.valueLoss;
        updateCount++;
      }
    }

    // Decay exploration
    this.logStd = this.logStd.map((s) => Math.max(s * 0.995, -1.5));

    return {
      policyLoss: updateCount > 0 ? totalPolicyLoss / updateCount : 0,
      valueLoss: updateCount > 0 ? totalValueLoss / updateCount : 0,
    };
  }

  private clipGradients(
    grads: { [varName: string]: tf.Tensor },
    maxNorm: number
  ): { [varName: string]: tf.Tensor } {
    const gradValues = Object.values(grads);
    const gradNames = Object.keys(grads);

    let sumSquares = tf.scalar(0);
    for (const grad of gradValues) {
      sumSquares = sumSquares.add(grad.square().sum());
    }
    const globalNorm = sumSquares.sqrt();
    const clipCoef = tf.minimum(tf.scalar(1), tf.scalar(maxNorm).div(globalNorm.add(1e-6)));

    const clippedGrads: { [varName: string]: tf.Tensor } = {};
    for (let i = 0; i < gradNames.length; i++) {
      clippedGrads[gradNames[i]] = gradValues[i].mul(clipCoef);
    }
    
    sumSquares.dispose();
    globalNorm.dispose();
    clipCoef.dispose();
    return clippedGrads;
  }

  private async trainStep(
    observations: number[][],
    actions: number[][],
    advantages: number[],
    returns: number[],
    oldLogProbs: number[],
    oldValues: number[]
  ): Promise<{ policyLoss: number; valueLoss: number }> {
    let policyLossValue = 0;
    let valueLossValue = 0;

    // Policy update
    const policyGrads = tf.variableGrads(() => {
      const obsTensor = tf.tensor2d(observations);
      const actionTensor = tf.tensor2d(actions).mul(2).sub(1);
      const advantageTensor = tf.tensor1d(advantages);
      const oldLogProbTensor = tf.tensor1d(oldLogProbs);

      const actionMean = this.policyNetwork.predict(obsTensor) as tf.Tensor;
      const newLogProb = this.gaussianLogProb(actionTensor, actionMean, this.logStd);

      const ratio = tf.exp(newLogProb.sub(oldLogProbTensor));
      const clippedRatio = ratio.clipByValue(1 - this.config.epsilon, 1 + this.config.epsilon);

      const surrogate1 = ratio.mul(advantageTensor);
      const surrogate2 = clippedRatio.mul(advantageTensor);
      const policyLoss = tf.minimum(surrogate1, surrogate2).mean().neg();

      const entropy = tf.scalar(0.5 * this.config.actionSize * (1 + Math.log(2 * Math.PI)))
        .add(tf.tensor1d(this.logStd).sum());

      policyLossValue = policyLoss.dataSync()[0];
      return policyLoss.sub(entropy.mul(this.config.entropyCoef)) as tf.Scalar;
    });

    const clippedPolicyGrads = this.clipGradients(policyGrads.grads, this.config.maxGradNorm);
    this.policyOptimizer.applyGradients(clippedPolicyGrads);
    
    Object.values(clippedPolicyGrads).forEach(g => g.dispose());
    tf.dispose(policyGrads);

    // Value update
    const valueGrads = tf.variableGrads(() => {
      const obsTensor = tf.tensor2d(observations);
      const returnTensor = tf.tensor1d(returns);
      const oldValueTensor = tf.tensor1d(oldValues);

      const valuePred = (this.valueNetwork.predict(obsTensor) as tf.Tensor).squeeze();
      const valueClipped = oldValueTensor.add(
        valuePred.sub(oldValueTensor).clipByValue(
          -this.config.valueClipRange,
          this.config.valueClipRange
        )
      );

      const valueLoss1 = valuePred.sub(returnTensor).square();
      const valueLoss2 = valueClipped.sub(returnTensor).square();
      const valueLoss = tf.maximum(valueLoss1, valueLoss2).mean();

      valueLossValue = valueLoss.dataSync()[0];
      return valueLoss.mul(this.config.valueCoef) as tf.Scalar;
    });

    const clippedValueGrads = this.clipGradients(valueGrads.grads, this.config.maxGradNorm);
    this.valueOptimizer.applyGradients(clippedValueGrads);
    
    Object.values(clippedValueGrads).forEach(g => g.dispose());
    tf.dispose(valueGrads);

    return { policyLoss: policyLossValue, valueLoss: valueLossValue };
  }

  private computeGAE(
    rewards: number[],
    values: number[],
    dones: boolean[],
    lambda = 0.95
  ): number[] {
    const advantages: number[] = new Array(rewards.length).fill(0);
    let lastAdvantage = 0;

    for (let t = rewards.length - 1; t >= 0; t--) {
      const nextValue = t === rewards.length - 1 ? 0 : values[t + 1];
      const delta = rewards[t] + this.config.gamma * nextValue * (dones[t] ? 0 : 1) - values[t];
      advantages[t] = lastAdvantage = delta + this.config.gamma * lambda * (dones[t] ? 0 : 1) * lastAdvantage;
    }
    return advantages;
  }

  async saveWeights(): Promise<ArrayBuffer> {
    const policyWeights = this.policyNetwork.getWeights();
    const valueWeights = this.valueNetwork.getWeights();
    const data = {
      policy: policyWeights.map((w) => ({ shape: w.shape, data: Array.from(w.dataSync()) })),
      value: valueWeights.map((w) => ({ shape: w.shape, data: Array.from(w.dataSync()) })),
      logStd: this.logStd,
    };
    return new TextEncoder().encode(JSON.stringify(data)).buffer;
  }

  async loadWeights(buffer: ArrayBuffer): Promise<void> {
    const data = JSON.parse(new TextDecoder().decode(buffer));
    const policyWeights = data.policy.map((w: any) => tf.tensor(w.data, w.shape));
    this.policyNetwork.setWeights(policyWeights);
    const valueWeights = data.value.map((w: any) => tf.tensor(w.data, w.shape));
    this.valueNetwork.setWeights(valueWeights);
    if (data.logStd) this.logStd = data.logStd;
  }

  dispose(): void {
    this.policyNetwork.dispose();
    this.valueNetwork.dispose();
  }
}
