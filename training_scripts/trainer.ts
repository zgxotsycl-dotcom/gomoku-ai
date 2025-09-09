/**
 * @file Standalone Training Worker
 * This script runs in an infinite loop, continuously checking for new training data
 * in the replay buffer, training the model on it, and saving new checkpoints.
 */

import * as tf from '@tensorflow/tfjs-node-gpu';
import * as fs from 'fs/promises';
import * as path from 'path';
import { createDualResNetModel } from './src/model';
import type { Player } from './src/ai';

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

interface TrainingSample {
    state: (Player | null)[][];
    policy: number[];
    value: number;
    player: Player;
}

// --- Helper Functions ---

function getSymmetries(state: (Player | null)[][], policy: number[]): { state: (Player | null)[][], policy: number[] }[] {
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

function rotateBoard(board: (Player | null)[][]): (Player | null)[][] {
    const newBoard = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(null));
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) { newBoard[c][BOARD_SIZE - 1 - r] = board[r][c]; }
    }
    return newBoard;
}

function flipBoard(board: (Player | null)[][]): (Player | null)[][] {
    return board.map(row => row.slice().reverse());
}

function rotatePolicy(policy: number[]): number[] {
    const newPolicy = Array(BOARD_SIZE * BOARD_SIZE).fill(0);
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) { newPolicy[c * BOARD_SIZE + (BOARD_SIZE - 1 - r)] = policy[r * BOARD_SIZE + c]; }
    }
    return newPolicy;
}

function flipPolicy(policy: number[]): number[] {
    const newPolicy = Array(BOARD_SIZE * BOARD_SIZE).fill(0);
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) { newPolicy[r * BOARD_SIZE + (BOARD_SIZE - 1 - c)] = policy[r * BOARD_SIZE + c]; }
    }
    return newPolicy;
}

function augmentAndConvertToTensors(samples: TrainingSample[]): { xs: tf.Tensor4D, ys: { policy: tf.Tensor2D, value: tf.Tensor2D } } {
    return tf.tidy(() => {
        const augmentedStates: tf.Tensor4D[] = [];
        const augmentedPolicies: tf.Tensor2D[] = [];
        const augmentedValues: number[][] = [];

        for (const sample of samples) {
            const symmetries = getSymmetries(sample.state, sample.policy);
            for (const sym of symmetries) {
                const player = sample.player || 'black';
                const opponent: Player = player === 'black' ? 'white' : 'black';

                const playerChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));
                const opponentChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));
                for (let r = 0; r < BOARD_SIZE; r++) {
                    for (let c = 0; c < BOARD_SIZE; c++) {
                        if (sym.state[r][c] === player) playerChannel[r][c] = 1;
                        else if (sym.state[r][c] === opponent) opponentChannel[r][c] = 1;
                    }
                }
                const colorChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(player === 'black' ? 1 : 0));

                const stacked = tf.stack([tf.tensor2d(playerChannel), tf.tensor2d(opponentChannel), tf.tensor2d(colorChannel)], 2);
                augmentedStates.push(stacked.expandDims(0) as tf.Tensor4D);
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
            let allSamples: TrainingSample[] = [];
            for (const file of gameFiles) {
                const filePath = path.join(REPLAY_BUFFER_PATH, file);
                try {
                    const fileContent = await fs.readFile(filePath, 'utf-8');
                    allSamples.push(...JSON.parse(fileContent));
                    await fs.rename(filePath, path.join(ARCHIVE_PATH, file));
                } catch (e) { console.error(`[Trainer] Error processing file ${file}:`, e); }
            }
            console.log(`[Trainer] Loaded a total of ${allSamples.length} samples.`);

            if (allSamples.length === 0) continue;

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
                    if (chunk.length === 0) continue;

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

        } catch (e) {
            console.error('[Trainer] An error occurred in the main loop:', e);
        }
        await new Promise(resolve => setTimeout(resolve, TRAIN_INTERVAL_MS));
    }
}

train();