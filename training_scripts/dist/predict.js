"use strict";
/**
 * @file Model Prediction Script
 * This script loads a pre-trained model and uses it to make a prediction on a sample board state.
 * It serves as a crucial test to ensure the model was saved correctly and is ready for integration.
 */
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
const tf = __importStar(require("@tensorflow/tfjs-node-gpu"));
const fs = __importStar(require("fs"));
// --- Configuration ---
const MODEL_PATH = 'file://./gomoku_model/model.json';
const BOARD_SIZE = 19;
/**
 * Converts a board state into a 3-channel tensor for model input.
 * @param board The 19x19 board state.
 * @param player The current player.
 * @returns A 4D tensor of shape [1, 19, 19, 3].
 */
function boardToInputTensor(board, player) {
    const opponent = player === 'black' ? 'white' : 'black';
    const playerChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));
    const opponentChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            if (board[r][c] === player) {
                playerChannel[r][c] = 1;
            }
            else if (board[r][c] === opponent) {
                opponentChannel[r][c] = 1;
            }
        }
    }
    const colorChannelValue = player === 'black' ? 1 : 0;
    const colorChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(colorChannelValue));
    const tensor = tf.tensor4d([playerChannel, opponentChannel, colorChannel], [1, BOARD_SIZE, BOARD_SIZE, 3]);
    // The default channel order is channel-last, but tfjs-node might need explicit permutation.
    // Let's permute to be safe: [batch, channels, height, width] -> [batch, height, width, channels]
    return tensor.transpose([0, 2, 3, 1]);
}
/**
 * Main function to load the model and run a prediction.
 */
async function runPrediction() {
    console.log('--- Gomoku AI Model Prediction Test ---');
    if (!fs.existsSync('./gomoku_model/model.json')) {
        console.error(`
Error: Model file not found at ${MODEL_PATH}`);
        console.error('Please run the training script (train_nn.ts) first to generate the model.');
        return;
    }
    console.log(`Loading model from ${MODEL_PATH}...`);
    const model = await tf.loadLayersModel(MODEL_PATH);
    console.log('Model loaded successfully.');
    model.summary();
    // Create a sample board state (e.g., an empty board for the first move)
    const sampleBoard = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(null));
    const currentPlayer = 'black';
    console.log(`
Predicting for a sample board state (Player: ${currentPlayer})...
`);
    const inputTensor = boardToInputTensor(sampleBoard, currentPlayer);
    const prediction = model.predict(inputTensor);
    const [policyTensor, valueTensor] = prediction;
    const policy = await policyTensor.data();
    const value = await valueTensor.data();
    // Find the best move from the policy output
    const bestMoveIndex = tf.argMax(policy).dataSync()[0];
    const bestMoveRow = Math.floor(bestMoveIndex / BOARD_SIZE);
    const bestMoveCol = bestMoveIndex % BOARD_SIZE;
    const confidence = policy[bestMoveIndex];
    console.log('\n--- Prediction Results ---');
    console.log(`Predicted Value (Win/Loss estimate): ${value[0].toFixed(4)}`);
    console.log(`Policy Head's Best Move: [${bestMoveRow}, ${bestMoveCol}] with confidence ${confidence.toFixed(4)}`);
    console.log('------------------------\n');
    // Dispose tensors to free up memory
    inputTensor.dispose();
    policyTensor.dispose();
    valueTensor.dispose();
}
// --- Run Script ---
runPrediction().catch(err => {
    console.error('\nAn unexpected error occurred during prediction:', err);
});
