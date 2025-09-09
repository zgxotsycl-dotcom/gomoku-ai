/**
 * @file Standalone Self-Play Worker
 * This script runs in an infinite loop to continuously play games against itself
 * using the current best model, and saves the generated training data to a replay buffer.
 */

import * as tf from '@tensorflow/tfjs-node-gpu';
import * as fs from 'fs/promises';
import * as path from 'path';
import { findBestMoveNN, checkWin, getOpponent } from './src/ai';
import type { Player } from './src/ai';

// --- Configuration ---
const MAIN_MODEL_PATH = './model_main';
const REPLAY_BUFFER_PATH = './replay_buffer';
const BOARD_SIZE = 19;
const MCTS_THINK_TIME = 2000; // 2 seconds per move
const EXPLORATION_MOVES = 15; // Number of moves to use temperature sampling for exploration

// --- Worker Setup ---
const workerId = process.argv[2] || '0'; // Get worker ID from command-line argument
let model: tf.LayersModel | null = null;

async function loadModel(): Promise<tf.LayersModel> {
    console.log(`[Worker ${workerId}] Loading model from ${MAIN_MODEL_PATH}...`);
    try {
        return await tf.loadLayersModel(`file://${path.resolve(MAIN_MODEL_PATH)}/model.json`);
    } catch (e) {
        console.error(`[Worker ${workerId}] Could not load model. Error: ${e}`);
        console.log(`[Worker ${workerId}] Waiting for model to become available...`);
        await new Promise(resolve => setTimeout(resolve, 10000)); 
        return loadModel(); // Retry recursively
    }
}

async function runSingleGame() {
    if (!model) throw new Error('Model is not loaded.');

    let board: (Player | null)[][] = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(null));
    let player: Player = 'black';
    const history: { state: (Player | null)[][], policy: number[], player: Player }[] = [];

    for (let moveCount = 0; moveCount < (BOARD_SIZE * BOARD_SIZE); moveCount++) {
        const { bestMove, policy: mctsPolicy } = await findBestMoveNN(model, board, player, MCTS_THINK_TIME);
        if (!bestMove || bestMove[0] === -1) break;

        const policyTarget = new Array(BOARD_SIZE * BOARD_SIZE).fill(0);
        let totalVisits = 0;
        mctsPolicy.forEach(p => totalVisits += p.visits);
        if (totalVisits > 0) {
            mctsPolicy.forEach(p => {
                const moveIndex = p.move[0] * BOARD_SIZE + p.move[1];
                policyTarget[moveIndex] = p.visits / totalVisits;
            });
        }
        history.push({ state: JSON.parse(JSON.stringify(board)), player, policy: policyTarget });

        let chosenMove: [number, number];
        // During exploration phase, if there are multiple moves, sample one.
        if (moveCount < EXPLORATION_MOVES && mctsPolicy.length > 1) {
            chosenMove = tf.tidy(() => {
                const moves = mctsPolicy.map(p => p.move);
                const probabilities = mctsPolicy.map(p => p.visits / totalVisits);
                const logits = tf.tensor1d(probabilities).log();
                const moveIndex = tf.multinomial(logits, 1).dataSync()[0];
                return moves[moveIndex];
            });
        } else {
            // If there's only one move, or we are past the exploration phase, take the best move.
            chosenMove = bestMove;
        }

        if (!chosenMove) break; // Should not happen, but as a safeguard.

        board[chosenMove[0]][chosenMove[1]] = player;

        if (checkWin(board, player, chosenMove)) {
            return history.map(h => ({ ...h, value: h.player === player ? 1 : -1 }));
        }
        player = getOpponent(player);
    }
    return history.map(h => ({ ...h, value: 0 })); // Draw
}

async function main() {
    console.log(`[Worker ${workerId}] Starting...`);
    model = await loadModel(); // Initial model load

    let gameCounter = 0;
    while (true) {
        console.log(`[Worker ${workerId}] Starting game #${++gameCounter}`);
        try {
            if (gameCounter % 5 === 0) { 
                model = await loadModel();
            }

            const gameData = await runSingleGame();
            const fileName = `game_${workerId}_${Date.now()}.json`;
            const filePath = path.join(REPLAY_BUFFER_PATH, fileName);
            await fs.writeFile(filePath, JSON.stringify(gameData));
            console.log(`[Worker ${workerId}] Game finished. Saved ${gameData.length} states to ${fileName}`);

        } catch (e) {
            console.error(`[Worker ${workerId}] An error occurred during game loop:`, e);
            await new Promise(resolve => setTimeout(resolve, 5000));
        }
    }
}

main();