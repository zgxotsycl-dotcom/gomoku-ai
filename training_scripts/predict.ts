/**
 * @file Model Prediction Script
 * This script loads a pre-trained model and uses it to make a prediction on a sample board state.
 * It serves as a crucial test to ensure the model was saved correctly and is ready for integration.
 */

import * as tf from '@tensorflow/tfjs-node-gpu';
import * as fs from 'fs';
import { Player } from './src/ai';

// --- Configuration ---
const MODEL_PATH = 'file://./gomoku_model/model.json';
const BOARD_SIZE = 19;

/**
 * Converts a board state into a 3-channel tensor for model input.
 * @param board The 19x19 board state.
 * @param player The current player.
 * @returns A 4D tensor of shape [1, 19, 19, 3].
 */
function boardToInputTensor(board: (Player | null)[][], player: Player): tf.Tensor4D {
    const opponent = player === 'black' ? 'white' : 'black';
    
    const playerChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));
    const opponentChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));

    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            if (board[r][c] === player) {
                playerChannel[r][c] = 1;
            } else if (board[r][c] === opponent) {
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
        console.error(
`
Error: Model file not found at ${MODEL_PATH}`
);
        console.error('Please run the training script (train_nn.ts) first to generate the model.');
        return;
    }

    console.log(`Loading model from ${MODEL_PATH}...`);
    const model = await tf.loadLayersModel(MODEL_PATH);
    console.log('Model loaded successfully.');
    model.summary();

    // Create a sample board state (e.g., an empty board for the first move)
    const sampleBoard: (Player | null)[][] = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(null));
    const currentPlayer: Player = 'black';
    console.log(
`
Predicting for a sample board state (Player: ${currentPlayer})...
`
);

    const inputTensor = boardToInputTensor(sampleBoard, currentPlayer);

    const prediction = model.predict(inputTensor) as tf.Tensor[];
    const [policyTensor, valueTensor] = prediction;

    const policy = await policyTensor.data() as Float32Array;
    const value = await valueTensor.data() as Float32Array;

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