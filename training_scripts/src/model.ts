import * as tf from '@tensorflow/tfjs-node-gpu';

// This function is not strictly needed for node, but good practice.
export function initializeBackend() {
    console.log(`TensorFlow.js backend: ${tf.getBackend()}`);
}

const BOARD_SIZE = 19;
const RESIDUAL_BLOCKS = 5;
const CONV_FILTERS = 64;
const L2_REGULARIZATION = 0.0001; // L2 regularization factor

function createResidualBlock(inputTensor: tf.SymbolicTensor): tf.SymbolicTensor {
    const l2_regularizer = tf.regularizers.l2({ l2: L2_REGULARIZATION });

    const initialConv = tf.layers.conv2d({
        filters: CONV_FILTERS,
        kernelSize: 3,
        padding: 'same',
        kernelRegularizer: l2_regularizer
    }).apply(inputTensor) as tf.SymbolicTensor;
    const bn1 = tf.layers.batchNormalization().apply(initialConv) as tf.SymbolicTensor;
    const relu1 = tf.layers.reLU().apply(bn1) as tf.SymbolicTensor;

    const nextConv = tf.layers.conv2d({
        filters: CONV_FILTERS,
        kernelSize: 3,
        padding: 'same',
        kernelRegularizer: l2_regularizer
    }).apply(relu1) as tf.SymbolicTensor;
    const bn2 = tf.layers.batchNormalization().apply(nextConv) as tf.SymbolicTensor;

    const add = tf.layers.add().apply([inputTensor, bn2]) as tf.SymbolicTensor;
    const output = tf.layers.reLU().apply(add) as tf.SymbolicTensor;
    return output;
}

export function createDualResNetModel(): tf.LayersModel {
    const l2_regularizer = tf.regularizers.l2({ l2: L2_REGULARIZATION });
    const inputShape = [BOARD_SIZE, BOARD_SIZE, 3];
    const input = tf.input({ shape: inputShape });

    const initialConv = tf.layers.conv2d({
        filters: CONV_FILTERS,
        kernelSize: 3,
        padding: 'same',
        kernelRegularizer: l2_regularizer
    }).apply(input) as tf.SymbolicTensor;
    const bn = tf.layers.batchNormalization().apply(initialConv) as tf.SymbolicTensor;
    let body = tf.layers.reLU().apply(bn) as tf.SymbolicTensor;

    for (let i = 0; i < RESIDUAL_BLOCKS; i++) {
        body = createResidualBlock(body);
    }

    // Policy Head
    const policyConv = tf.layers.conv2d({
        filters: 2, kernelSize: 1, kernelRegularizer: l2_regularizer
    }).apply(body) as tf.SymbolicTensor;
    const policyBn = tf.layers.batchNormalization().apply(policyConv) as tf.SymbolicTensor;
    const policyRelu = tf.layers.reLU().apply(policyBn) as tf.SymbolicTensor;
    const policyFlatten = tf.layers.flatten().apply(policyRelu) as tf.SymbolicTensor;
    const policyOutput = tf.layers.dense({ units: BOARD_SIZE * BOARD_SIZE, activation: 'softmax', name: 'policy' }).apply(policyFlatten) as tf.SymbolicTensor;

    // Value Head
    const valueConv = tf.layers.conv2d({
        filters: 1, kernelSize: 1, kernelRegularizer: l2_regularizer
    }).apply(body) as tf.SymbolicTensor;
    const valueBn = tf.layers.batchNormalization().apply(valueConv) as tf.SymbolicTensor;
    const valueRelu = tf.layers.reLU().apply(valueBn) as tf.SymbolicTensor;
    const valueFlatten = tf.layers.flatten().apply(valueRelu) as tf.SymbolicTensor;
    const valueDense = tf.layers.dense({ units: 64, activation: 'relu' }).apply(valueFlatten) as tf.SymbolicTensor;
    const valueOutput = tf.layers.dense({ units: 1, activation: 'tanh', name: 'value' }).apply(valueDense) as tf.SymbolicTensor;

    const model = tf.model({ inputs: input, outputs: [policyOutput, valueOutput] });

    model.compile({
        optimizer: tf.train.adam(),
        loss: { 'policy': 'categoricalCrossentropy', 'value': 'meanSquaredError' },
        metrics: { 'policy': 'accuracy', 'value': tf.metrics.meanAbsoluteError }
    });

    console.log('Neural network model with L2 Regularization created and compiled successfully.');
    return model;
}