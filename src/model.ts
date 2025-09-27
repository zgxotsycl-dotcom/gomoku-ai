import tf from './tf';
import type * as TFT from '@tensorflow/tfjs';

const BOARD_SIZE = Number(process.env.BOARD_SIZE || 15);
const RESIDUAL_BLOCKS = Number(process.env.RESIDUAL_BLOCKS || 5);
const CONV_FILTERS = Number(process.env.CONV_FILTERS || 64);
const L2_REGULARIZATION = Number(process.env.L2_REGULARIZATION || 0.0001);
const USE_SE_BLOCK = (process.env.USE_SE_BLOCK || 'true').toLowerCase() === 'true';
const USE_BOTTLENECK = (process.env.USE_BOTTLENECK || 'false').toLowerCase() === 'true';
const BOTTLENECK_FACTOR = Math.max(1, Number(process.env.BOTTLENECK_FACTOR || 4));
const RESIDUAL_DROPOUT = Math.min(Math.max(Number(process.env.RESIDUAL_DROPOUT || 0), 0), 0.9);
const HEAD_DROPOUT = Math.min(Math.max(Number(process.env.HEAD_DROPOUT || 0), 0), 0.9);
const POLICY_HEAD_FILTERS = Math.max(1, Number(process.env.POLICY_HEAD_FILTERS || 2));
const VALUE_HEAD_FILTERS = Math.max(1, Number(process.env.VALUE_HEAD_FILTERS || 1));
const VALUE_HEAD_UNITS = Math.max(8, Number(process.env.VALUE_HEAD_UNITS || 64));
const MODEL_LEARNING_RATE = Number(process.env.MODEL_LEARNING_RATE || 0.001);

function createSqueezeExciteBlock(inputTensor: TFT.SymbolicTensor, ratio = 16): TFT.SymbolicTensor {
  if (!USE_SE_BLOCK) return inputTensor;
  const filters = inputTensor.shape[inputTensor.shape.length - 1] as number;
  const squeezed = tf.layers.globalAveragePooling2d({}).apply(inputTensor) as TFT.SymbolicTensor;
  const reshaped = tf.layers.reshape({ targetShape: [1, 1, filters] }).apply(squeezed) as TFT.SymbolicTensor;

  const reducedUnits = Math.max(1, Math.floor(filters / ratio));
  const dense1 = tf.layers.dense({ units: reducedUnits, activation: 'relu' }).apply(reshaped) as TFT.SymbolicTensor;
  const dense2 = tf.layers.dense({ units: filters, activation: 'sigmoid' }).apply(dense1) as TFT.SymbolicTensor;

  return tf.layers.multiply().apply([inputTensor, dense2]) as TFT.SymbolicTensor;
}

function applyDropout(tensor: TFT.SymbolicTensor, rate: number): TFT.SymbolicTensor {
  if (rate <= 0) return tensor;
  return tf.layers.dropout({ rate }).apply(tensor) as TFT.SymbolicTensor;
}

function createResidualBlock(inputTensor: TFT.SymbolicTensor): TFT.SymbolicTensor {
  const l2_regularizer = tf.regularizers.l2({ l2: L2_REGULARIZATION });
  let blockOutput: TFT.SymbolicTensor;

  if (USE_BOTTLENECK) {
    const reducedFilters = Math.max(1, Math.floor(CONV_FILTERS / BOTTLENECK_FACTOR));
    const conv1 = tf.layers.conv2d({
      filters: reducedFilters,
      kernelSize: 1,
      padding: 'same',
      kernelRegularizer: l2_regularizer
    }).apply(inputTensor) as TFT.SymbolicTensor;
    const bn1 = tf.layers.batchNormalization().apply(conv1) as TFT.SymbolicTensor;
    const relu1 = tf.layers.reLU().apply(bn1) as TFT.SymbolicTensor;

    const conv2 = tf.layers.conv2d({
      filters: reducedFilters,
      kernelSize: 3,
      padding: 'same',
      kernelRegularizer: l2_regularizer
    }).apply(relu1) as TFT.SymbolicTensor;
    const bn2 = tf.layers.batchNormalization().apply(conv2) as TFT.SymbolicTensor;
    const relu2 = tf.layers.reLU().apply(bn2) as TFT.SymbolicTensor;

    const conv3 = tf.layers.conv2d({
      filters: CONV_FILTERS,
      kernelSize: 1,
      padding: 'same',
      kernelRegularizer: l2_regularizer
    }).apply(relu2) as TFT.SymbolicTensor;
    blockOutput = tf.layers.batchNormalization().apply(conv3) as TFT.SymbolicTensor;
  } else {
    const convA = tf.layers.conv2d({
      filters: CONV_FILTERS,
      kernelSize: 3,
      padding: 'same',
      kernelRegularizer: l2_regularizer
    }).apply(inputTensor) as TFT.SymbolicTensor;
    const bnA = tf.layers.batchNormalization().apply(convA) as TFT.SymbolicTensor;
    const reluA = tf.layers.reLU().apply(bnA) as TFT.SymbolicTensor;

    const convB = tf.layers.conv2d({
      filters: CONV_FILTERS,
      kernelSize: 3,
      padding: 'same',
      kernelRegularizer: l2_regularizer
    }).apply(reluA) as TFT.SymbolicTensor;
    blockOutput = tf.layers.batchNormalization().apply(convB) as TFT.SymbolicTensor;
  }

  blockOutput = applyDropout(blockOutput, RESIDUAL_DROPOUT);
  const seApplied = createSqueezeExciteBlock(blockOutput);
  const add = tf.layers.add().apply([inputTensor, seApplied]) as TFT.SymbolicTensor;
  return tf.layers.reLU().apply(add) as TFT.SymbolicTensor;
}

export function createDualResNetModel(): TFT.LayersModel {
  const l2_regularizer = tf.regularizers.l2({ l2: L2_REGULARIZATION });
  const inputShape = [BOARD_SIZE, BOARD_SIZE, 3];
  const input = tf.input({ shape: inputShape });

  const initialConv = tf.layers.conv2d({
    filters: CONV_FILTERS,
    kernelSize: 3,
    padding: 'same',
    kernelRegularizer: l2_regularizer
  }).apply(input) as TFT.SymbolicTensor;
  const bn = tf.layers.batchNormalization().apply(initialConv) as TFT.SymbolicTensor;
  let body = tf.layers.reLU().apply(bn) as TFT.SymbolicTensor;

  for (let i = 0; i < RESIDUAL_BLOCKS; i++) {
    body = createResidualBlock(body);
  }

  // Policy Head
  const policyConv = tf.layers.conv2d({
    filters: POLICY_HEAD_FILTERS,
    kernelSize: 1,
    padding: 'same',
    kernelRegularizer: l2_regularizer
  }).apply(body) as TFT.SymbolicTensor;
  const policyBn = tf.layers.batchNormalization().apply(policyConv) as TFT.SymbolicTensor;
  const policyRelu = tf.layers.reLU().apply(policyBn) as TFT.SymbolicTensor;
  let policyFlatten = tf.layers.flatten().apply(policyRelu) as TFT.SymbolicTensor;
  policyFlatten = applyDropout(policyFlatten, HEAD_DROPOUT);
  const policyOutput = tf.layers.dense({
    units: BOARD_SIZE * BOARD_SIZE,
    activation: 'softmax',
    name: 'policy_head'
  }).apply(policyFlatten) as TFT.SymbolicTensor;

  // Value Head
  const valueConv = tf.layers.conv2d({
    filters: VALUE_HEAD_FILTERS,
    kernelSize: 1,
    padding: 'same',
    kernelRegularizer: l2_regularizer
  }).apply(body) as TFT.SymbolicTensor;
  const valueBn = tf.layers.batchNormalization().apply(valueConv) as TFT.SymbolicTensor;
  const valueRelu = tf.layers.reLU().apply(valueBn) as TFT.SymbolicTensor;
  let valueFlatten = tf.layers.flatten().apply(valueRelu) as TFT.SymbolicTensor;
  valueFlatten = applyDropout(valueFlatten, HEAD_DROPOUT);
  const valueDense = tf.layers.dense({ units: VALUE_HEAD_UNITS, activation: 'relu' }).apply(valueFlatten) as TFT.SymbolicTensor;
  const valueOutput = tf.layers.dense({ units: 1, activation: 'tanh', name: 'value_head' }).apply(valueDense) as TFT.SymbolicTensor;

  const model = tf.model({ inputs: input, outputs: [policyOutput, valueOutput] });

  model.compile({
    optimizer: tf.train.adam(MODEL_LEARNING_RATE),
    loss: { policy_head: 'categoricalCrossentropy', value_head: 'meanSquaredError' },
    metrics: { policy_head: 'accuracy', value_head: 'mae' }
  });

  console.log(`Neural network created (filters=${CONV_FILTERS}, blocks=${RESIDUAL_BLOCKS}, bottleneck=${USE_BOTTLENECK}).`);
  return model;
}
