"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.initializeBackend = initializeBackend;
exports.createDualResNetModel = createDualResNetModel;
const tf = __importStar(require("@tensorflow/tfjs-node-gpu"));
// This function is not strictly needed for node, but good practice.
function initializeBackend() {
    console.log(`TensorFlow.js backend: ${tf.getBackend()}`);
}
const BOARD_SIZE = 19;
const RESIDUAL_BLOCKS = 5;
const CONV_FILTERS = 64;
const L2_REGULARIZATION = 0.0001; // L2 regularization factor
function createResidualBlock(inputTensor) {
    const l2_regularizer = tf.regularizers.l2({ l2: L2_REGULARIZATION });
    const initialConv = tf.layers.conv2d({
        filters: CONV_FILTERS,
        kernelSize: 3,
        padding: 'same',
        kernelRegularizer: l2_regularizer
    }).apply(inputTensor);
    const bn1 = tf.layers.batchNormalization().apply(initialConv);
    const relu1 = tf.layers.reLU().apply(bn1);
    const nextConv = tf.layers.conv2d({
        filters: CONV_FILTERS,
        kernelSize: 3,
        padding: 'same',
        kernelRegularizer: l2_regularizer
    }).apply(relu1);
    const bn2 = tf.layers.batchNormalization().apply(nextConv);
    const add = tf.layers.add().apply([inputTensor, bn2]);
    const output = tf.layers.reLU().apply(add);
    return output;
}
function createDualResNetModel() {
    const l2_regularizer = tf.regularizers.l2({ l2: L2_REGULARIZATION });
    const inputShape = [BOARD_SIZE, BOARD_SIZE, 3];
    const input = tf.input({ shape: inputShape });
    const initialConv = tf.layers.conv2d({
        filters: CONV_FILTERS,
        kernelSize: 3,
        padding: 'same',
        kernelRegularizer: l2_regularizer
    }).apply(input);
    const bn = tf.layers.batchNormalization().apply(initialConv);
    let body = tf.layers.reLU().apply(bn);
    for (let i = 0; i < RESIDUAL_BLOCKS; i++) {
        body = createResidualBlock(body);
    }
    // Policy Head
    const policyConv = tf.layers.conv2d({
        filters: 2, kernelSize: 1, kernelRegularizer: l2_regularizer
    }).apply(body);
    const policyBn = tf.layers.batchNormalization().apply(policyConv);
    const policyRelu = tf.layers.reLU().apply(policyBn);
    const policyFlatten = tf.layers.flatten().apply(policyRelu);
    const policyOutput = tf.layers.dense({ units: BOARD_SIZE * BOARD_SIZE, activation: 'softmax', name: 'policy' }).apply(policyFlatten);
    // Value Head
    const valueConv = tf.layers.conv2d({
        filters: 1, kernelSize: 1, kernelRegularizer: l2_regularizer
    }).apply(body);
    const valueBn = tf.layers.batchNormalization().apply(valueConv);
    const valueRelu = tf.layers.reLU().apply(valueBn);
    const valueFlatten = tf.layers.flatten().apply(valueRelu);
    const valueDense = tf.layers.dense({ units: 64, activation: 'relu' }).apply(valueFlatten);
    const valueOutput = tf.layers.dense({ units: 1, activation: 'tanh', name: 'value' }).apply(valueDense);
    const model = tf.model({ inputs: input, outputs: [policyOutput, valueOutput] });
    model.compile({
        optimizer: tf.train.adam(),
        loss: { 'policy': 'categoricalCrossentropy', 'value': 'meanSquaredError' },
        metrics: { 'policy': 'accuracy', 'value': tf.metrics.meanAbsoluteError }
    });
    console.log('Neural network model with L2 Regularization created and compiled successfully.');
    return model;
}
