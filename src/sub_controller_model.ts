import tf from './tf';
import type * as TFT from '@tensorflow/tfjs';

const BOARD_SIZE = Number(process.env.BOARD_SIZE || 15);
const RESIDUAL_BLOCKS = 5;
const CONV_FILTERS = 64;
const L2_REGULARIZATION = 0.0001;
const NUM_SUBGOALS = 2; // create_open_four, block_open_four

function createSqueezeExciteBlock(
  inputTensor: TFT.SymbolicTensor,
  ratio = 16
): TFT.SymbolicTensor {
  const init = inputTensor;
  const filters = init.shape[init.shape.length - 1] as number;
  const se = tf.layers.globalAveragePooling2d({}).apply(init) as TFT.SymbolicTensor;
  const seReshaped = tf.layers
    .reshape({ targetShape: [1, 1, filters] })
    .apply(se) as TFT.SymbolicTensor;
  const dense1 = tf.layers
    .dense({ units: Math.max(1, Math.floor(filters / ratio)), activation: 'relu' })
    .apply(seReshaped) as TFT.SymbolicTensor;
  const dense2 = tf.layers
    .dense({ units: filters, activation: 'sigmoid' })
    .apply(dense1) as TFT.SymbolicTensor;
  const scaled = tf.layers.multiply().apply([init, dense2]) as TFT.SymbolicTensor;
  return scaled;
}

function createResidualBlock(inputTensor: TFT.SymbolicTensor): TFT.SymbolicTensor {
  const l2_regularizer = tf.regularizers.l2({ l2: L2_REGULARIZATION });

  const conv1 = tf.layers
    .conv2d({ filters: CONV_FILTERS, kernelSize: 3, padding: 'same', kernelRegularizer: l2_regularizer })
    .apply(inputTensor) as TFT.SymbolicTensor;
  const bn1 = tf.layers.batchNormalization().apply(conv1) as TFT.SymbolicTensor;
  const relu1 = tf.layers.reLU().apply(bn1) as TFT.SymbolicTensor;

  const conv2 = tf.layers
    .conv2d({ filters: CONV_FILTERS, kernelSize: 3, padding: 'same', kernelRegularizer: l2_regularizer })
    .apply(relu1) as TFT.SymbolicTensor;
  const bn2 = tf.layers.batchNormalization().apply(conv2) as TFT.SymbolicTensor;

  const se = createSqueezeExciteBlock(bn2);
  const add = tf.layers.add().apply([inputTensor, se]) as TFT.SymbolicTensor;
  const out = tf.layers.reLU().apply(add) as TFT.SymbolicTensor;
  return out;
}

export function createSubControllerModel(): TFT.LayersModel {
  const l2_regularizer = tf.regularizers.l2({ l2: L2_REGULARIZATION });

  // Inputs
  const boardInput = tf.input({ shape: [BOARD_SIZE, BOARD_SIZE, 3], name: 'board_input' });
  const subgoalInput = tf.input({ shape: [NUM_SUBGOALS], name: 'subgoal_input' });

  // Body
  const initialConv = tf.layers
    .conv2d({ filters: CONV_FILTERS, kernelSize: 3, padding: 'same', kernelRegularizer: l2_regularizer })
    .apply(boardInput) as TFT.SymbolicTensor;
  const bn = tf.layers.batchNormalization().apply(initialConv) as TFT.SymbolicTensor;
  let body = tf.layers.reLU().apply(bn) as TFT.SymbolicTensor;

  for (let i = 0; i < RESIDUAL_BLOCKS; i++) {
    body = createResidualBlock(body);
  }

  // Policy head (board features)
  const policyConv = tf.layers
    .conv2d({ filters: 2, kernelSize: 1, kernelRegularizer: l2_regularizer })
    .apply(body) as TFT.SymbolicTensor;
  const policyBn = tf.layers.batchNormalization().apply(policyConv) as TFT.SymbolicTensor;
  const policyRelu = tf.layers.reLU().apply(policyBn) as TFT.SymbolicTensor;
  const policyFlatten = tf.layers.flatten().apply(policyRelu) as TFT.SymbolicTensor;

  // Concatenate board features with subgoal embedding
  const concat = tf.layers.concatenate().apply([policyFlatten, subgoalInput]) as TFT.SymbolicTensor;
  const dense = tf.layers.dense({ units: 256, activation: 'relu' }).apply(concat) as TFT.SymbolicTensor;
  const policyOutput = tf.layers
    .dense({ units: BOARD_SIZE * BOARD_SIZE, activation: 'softmax', name: 'policy_head' })
    .apply(dense) as TFT.SymbolicTensor;

  const model = tf.model({ inputs: [boardInput, subgoalInput], outputs: policyOutput });
  model.compile({ optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

  console.log('Sub-Controller model created and compiled successfully.');
  return model;
}
