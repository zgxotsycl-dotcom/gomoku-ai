import tf from './tf';
import type * as TFT from '@tensorflow/tfjs';

const BOARD_SIZE = Number(process.env.BOARD_SIZE || 15);
const RESIDUAL_BLOCKS = Number(process.env.RESIDUAL_BLOCKS || 5);
const CONV_FILTERS = Number(process.env.CONV_FILTERS || 64);
const L2_REGULARIZATION = Number(process.env.L2_REGULARIZATION || 0.0001);

function createSqueezeExciteBlock(inputTensor: TFT.SymbolicTensor, ratio = 16): TFT.SymbolicTensor {
    const init = inputTensor;
    const filters = init.shape[init.shape.length - 1] as number;

    const se = tf.layers.globalAveragePooling2d({}).apply(init) as TFT.SymbolicTensor;
    const seReshaped = tf.layers.reshape({ targetShape: [1, 1, filters] }).apply(se) as TFT.SymbolicTensor;
    
    const dense1 = tf.layers.dense({ units: filters / ratio, activation: 'relu' }).apply(seReshaped) as TFT.SymbolicTensor;
    const dense2 = tf.layers.dense({ units: filters, activation: 'sigmoid' }).apply(dense1) as TFT.SymbolicTensor;

    const scale = tf.layers.multiply().apply([init, dense2]) as TFT.SymbolicTensor;
    return scale;
}

function createResidualBlock(inputTensor: TFT.SymbolicTensor): TFT.SymbolicTensor {
    const l2_regularizer = tf.regularizers.l2({ l2: L2_REGULARIZATION });

    const initialConv = tf.layers.conv2d({
        filters: CONV_FILTERS,
        kernelSize: 3,
        padding: 'same',
        kernelRegularizer: l2_regularizer
    }).apply(inputTensor) as TFT.SymbolicTensor;
    const bn1 = tf.layers.batchNormalization().apply(initialConv) as TFT.SymbolicTensor;
    const relu1 = tf.layers.reLU().apply(bn1) as TFT.SymbolicTensor;

    const nextConv = tf.layers.conv2d({
        filters: CONV_FILTERS,
        kernelSize: 3,
        padding: 'same',
        kernelRegularizer: l2_regularizer
    }).apply(relu1) as TFT.SymbolicTensor;
    const bn2 = tf.layers.batchNormalization().apply(nextConv) as TFT.SymbolicTensor;

    const seBlock = createSqueezeExciteBlock(bn2);

    const add = tf.layers.add().apply([inputTensor, seBlock]) as TFT.SymbolicTensor;
    const output = tf.layers.reLU().apply(add) as TFT.SymbolicTensor;
    return output;
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
        filters: 2, kernelSize: 1, kernelRegularizer: l2_regularizer
    }).apply(body) as TFT.SymbolicTensor;
    const policyBn = tf.layers.batchNormalization().apply(policyConv) as TFT.SymbolicTensor;
    const policyRelu = tf.layers.reLU().apply(policyBn) as TFT.SymbolicTensor;
    const policyFlatten = tf.layers.flatten().apply(policyRelu) as TFT.SymbolicTensor;
    const policyOutput = tf.layers.dense({ units: BOARD_SIZE * BOARD_SIZE, activation: 'softmax', name: 'policy_head' }).apply(policyFlatten) as TFT.SymbolicTensor;

    // Value Head
    const valueConv = tf.layers.conv2d({
        filters: 1, kernelSize: 1, kernelRegularizer: l2_regularizer
    }).apply(body) as TFT.SymbolicTensor;
    const valueBn = tf.layers.batchNormalization().apply(valueConv) as TFT.SymbolicTensor;
    const valueRelu = tf.layers.reLU().apply(valueBn) as TFT.SymbolicTensor;
    const valueFlatten = tf.layers.flatten().apply(valueRelu) as TFT.SymbolicTensor;
    const valueDense = tf.layers.dense({ units: 64, activation: 'relu' }).apply(valueFlatten) as TFT.SymbolicTensor;
    const valueOutput = tf.layers.dense({ units: 1, activation: 'tanh', name: 'value_head' }).apply(valueDense) as TFT.SymbolicTensor;

    const model = tf.model({ inputs: input, outputs: [policyOutput, valueOutput] });

    model.compile({
        optimizer: tf.train.adam(),
        loss: { 'policy_head': 'categoricalCrossentropy', 'value_head': 'meanSquaredError' },
        metrics: { 'policy_head': 'accuracy', 'value_head': 'mae' }
    });

    console.log('Neural network model with SE-ResNet architecture created and compiled successfully.');
    return model;
}
