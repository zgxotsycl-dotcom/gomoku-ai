"use strict";
/**
 * @file Standalone Training Worker
 * This script runs in an infinite loop, continuously checking for new training data
 * in the replay buffer, training the model on it, and saving new checkpoints.
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
const fs = __importStar(require("fs/promises"));
const path = __importStar(require("path"));
// --- Configuration ---
const MAIN_MODEL_PATH = './model_main';
const REPLAY_BUFFER_PATH = './replay_buffer';
const ARCHIVE_PATH = './replay_buffer_archive';
const CHECKPOINT_PATH = './training_checkpoints';
const MIN_GAMES_TO_TRAIN = 100;
const TRAIN_INTERVAL_MS = 60 * 1000;
const BOARD_SIZE = 19;
// --- Training Hyperparameters ---
const EPOCHS = 5;
const CHUNK_SIZE = 8192;
const BATCH_SIZE = 128;
// --- Helper Functions ---
function getSymmetries(state, policy) {
    const symmetries = [];
    let currentBoard = state.map(row => [...row]);
    let currentPolicy = [...policy];
    for (let i = 0; i < 4; i++) {
        symmetries.push({ state: currentBoard, policy: currentPolicy });
        symmetries.push({ state: flipBoard(currentBoard), policy: flipPolicy(currentPolicy) });
        currentBoard = rotateBoard(currentBoard);
        currentPolicy = rotatePolicy(currentPolicy);
    }
    return symmetries;
}
function rotateBoard(board) {
    const newBoard = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(null));
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            newBoard[c][BOARD_SIZE - 1 - r] = board[r][c];
        }
    }
    return newBoard;
}
function flipBoard(board) {
    return board.map(row => row.slice().reverse());
}
function rotatePolicy(policy) {
    const newPolicy = Array(BOARD_SIZE * BOARD_SIZE).fill(0);
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            newPolicy[c * BOARD_SIZE + (BOARD_SIZE - 1 - r)] = policy[r * BOARD_SIZE + c];
        }
    }
    return newPolicy;
}
function flipPolicy(policy) {
    const newPolicy = Array(BOARD_SIZE * BOARD_SIZE).fill(0);
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            newPolicy[r * BOARD_SIZE + (BOARD_SIZE - 1 - c)] = policy[r * BOARD_SIZE + c];
        }
    }
    return newPolicy;
}
function augmentAndConvertToTensors(samples) {
    return tf.tidy(() => {
        const augmentedStates = [];
        const augmentedPolicies = [];
        const augmentedValues = [];
        for (const sample of samples) {
            const symmetries = getSymmetries(sample.state, sample.policy);
            for (const sym of symmetries) {
                const player = sample.player || 'black';
                const opponent = player === 'black' ? 'white' : 'black';
                const playerChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));
                const opponentChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));
                for (let r = 0; r < BOARD_SIZE; r++) {
                    for (let c = 0; c < BOARD_SIZE; c++) {
                        if (sym.state[r][c] === player)
                            playerChannel[r][c] = 1;
                        else if (sym.state[r][c] === opponent)
                            opponentChannel[r][c] = 1;
                    }
                }
                const colorChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(player === 'black' ? 1 : 0));
                const stacked = tf.stack([tf.tensor2d(playerChannel), tf.tensor2d(opponentChannel), tf.tensor2d(colorChannel)], 2);
                augmentedStates.push(stacked.expandDims(0));
                augmentedPolicies.push(tf.tensor2d([sym.policy]));
                augmentedValues.push([sample.value]);
            }
        }
        return {
            xs: tf.concat(augmentedStates),
            ys: {
                policy: tf.concat(augmentedPolicies),
                value: tf.tensor2d(augmentedValues)
            }
        };
    });
}
async function train() {
    console.log('--- Training Worker Started ---');
    while (true) {
        try {
            const files = await fs.readdir(REPLAY_BUFFER_PATH);
            const gameFiles = files.filter(f => f.endsWith('.json'));
            if (gameFiles.length < MIN_GAMES_TO_TRAIN) {
                console.log(`[Trainer] Not enough games to train (${gameFiles.length}/${MIN_GAMES_TO_TRAIN}). Waiting...`);
                await new Promise(resolve => setTimeout(resolve, TRAIN_INTERVAL_MS));
                continue;
            }
            console.log(`[Trainer] Found ${gameFiles.length} new games. Processing and training...`);
            let allSamples = [];
            for (const file of gameFiles) {
                const filePath = path.join(REPLAY_BUFFER_PATH, file);
                try {
                    const fileContent = await fs.readFile(filePath, 'utf-8');
                    allSamples.push(...JSON.parse(fileContent));
                    await fs.rename(filePath, path.join(ARCHIVE_PATH, file));
                }
                catch (e) {
                    console.error(`[Trainer] Error processing file ${file}:`, e);
                }
            }
            console.log(`[Trainer] Loaded a total of ${allSamples.length} samples.`);
            if (allSamples.length === 0)
                continue;
            const model = await tf.loadLayersModel(`file://${path.resolve(MAIN_MODEL_PATH)}/model.json`);
            model.compile({
                optimizer: tf.train.adam(),
                loss: { 'policy': 'categoricalCrossentropy', 'value': 'meanSquaredError' },
                metrics: { 'policy': 'accuracy', 'value': tf.metrics.meanAbsoluteError }
            });
            tf.util.shuffle(allSamples);
            const NUM_CHUNKS = Math.ceil(allSamples.length / CHUNK_SIZE);
            for (let epoch = 0; epoch < EPOCHS; epoch++) {
                console.log(`[Trainer] --- Epoch ${epoch + 1} / ${EPOCHS} ---`);
                for (let i = 0; i < NUM_CHUNKS; i++) {
                    const chunk = allSamples.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE);
                    if (chunk.length === 0)
                        continue;
                    const { xs, ys } = augmentAndConvertToTensors(chunk);
                    await model.fit(xs, ys, { batchSize: BATCH_SIZE, epochs: 1, shuffle: true });
                    xs.dispose();
                    ys.policy.dispose();
                    ys.value.dispose();
                    console.log(`[Trainer] Finished training chunk ${i + 1}/${NUM_CHUNKS}. Num Tensors: ${tf.memory().numTensors}`);
                }
            }
            const checkpointName = `model_checkpoint_${Date.now()}`;
            const checkpointDir = path.resolve(CHECKPOINT_PATH, checkpointName);
            await fs.mkdir(checkpointDir, { recursive: true });
            await model.save(`file://${checkpointDir}`);
            console.log(`[Trainer] New checkpoint saved: ${checkpointName}`);
        }
        catch (e) {
            console.error('[Trainer] An error occurred in the main loop:', e);
        }
        await new Promise(resolve => setTimeout(resolve, TRAIN_INTERVAL_MS));
    }
}
train();
