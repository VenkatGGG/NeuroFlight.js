/**
 * Network.ts - Neural network definitions for PPO
 */
import * as tf from '@tensorflow/tfjs';

export class NetworkFactory {
  static createPolicyNetwork(
    observationSize: number,
    actionSize: number,
    hiddenLayers: number[]
  ): tf.Sequential {
    const model = tf.sequential();
    
    // Input layer
    model.add(tf.layers.dense({
      inputShape: [observationSize],
      units: hiddenLayers[0],
      activation: 'relu',
      kernelInitializer: 'heNormal',
    }));

    // Hidden layers
    for (let i = 1; i < hiddenLayers.length; i++) {
      model.add(tf.layers.dense({
        units: hiddenLayers[i],
        activation: 'relu',
        kernelInitializer: 'heNormal',
      }));
    }

    // Output layer (continuous actions, tanh activation for range [-1, 1])
    model.add(tf.layers.dense({
      units: actionSize,
      activation: 'tanh',
      kernelInitializer: tf.initializers.glorotNormal({}),
    }));

    return model;
  }

  static createValueNetwork(
    observationSize: number,
    hiddenLayers: number[]
  ): tf.Sequential {
    const model = tf.sequential();

    // Input layer
    model.add(tf.layers.dense({
      inputShape: [observationSize],
      units: hiddenLayers[0],
      activation: 'relu',
      kernelInitializer: 'heNormal',
    }));

    // Hidden layers
    for (let i = 1; i < hiddenLayers.length; i++) {
      model.add(tf.layers.dense({
        units: hiddenLayers[i],
        activation: 'relu',
        kernelInitializer: 'heNormal',
      }));
    }

    // Output layer (single value estimate)
    model.add(tf.layers.dense({
      units: 1,
      kernelInitializer: tf.initializers.glorotNormal({}),
    }));

    return model;
  }
}
