/**
 * @file Model Evaluation Script
 * This script pits two models (a stable 'production' model and a new 'challenger' model)
 * against each other for a specified number of games to objectively measure if the
 * challenger is stronger.
 */

import * as tf from '@tensorflow/tfjs-node-gpu';
import { findBestMoveNN, checkWin, getOpponent } from './src/ai';
import type { Player } from './src/ai';

// --- Configuration ---
const PROD_MODEL_PATH = './gomoku_model_prod';
const CHALLENGER_MODEL_PATH = './gomoku_model';
const NUM_GAMES = 50; // Must be an even number
const MCTS_THINK_TIME = 1000; // 1 second per move for fast evaluation
const WIN_THRESHOLD = 0.55; // Challenger must win >55% of games to be promoted
const BOARD_SIZE = 19;

async function loadModel(modelPath: string): Promise<tf.LayersModel> {
    console.log(`Loading model from ${modelPath}...`);
    try {
        const model = await tf.loadLayersModel(`file://${modelPath}/model.json`);
        console.log(`Model loaded successfully from ${modelPath}.`);
        return model;
    } catch (e) {
        console.error(`Could not load model from ${modelPath}. Error: ${e}`);
        process.exit(1);
    }
}

async function runEvaluation() {
    console.log('--- Starting Model Evaluation ---');
    const prodModel = await loadModel(PROD_MODEL_PATH);
    const challengerModel = await loadModel(CHALLENGER_MODEL_PATH);

    let challengerWins = 0;
    let draws = 0;

    for (let i = 0; i < NUM_GAMES; i++) {
        console.log(`
--- Starting Game ${i + 1} / ${NUM_GAMES} ---`);
        let board: (Player | null)[][] = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(null));
        let player: Player = 'black';

        // Alternate who goes first
        const challengerIsBlack = i % 2 === 0;
        const players = {
            black: challengerIsBlack ? challengerModel : prodModel,
            white: challengerIsBlack ? prodModel : challengerModel,
        };
        console.log(`Challenger plays as: ${challengerIsBlack ? 'Black' : 'White'}`);

        let winner: Player | null = null;
        for (let moveCount = 0; moveCount < BOARD_SIZE * BOARD_SIZE; moveCount++) {
            const currentModel = players[player];
            const { bestMove } = await findBestMoveNN(currentModel, board, player, MCTS_THINK_TIME);

            if (!bestMove || bestMove[0] === -1) {
                console.log('Game ends in a draw (no more moves).');
                draws++;
                winner = null;
                break;
            }

            board[bestMove[0]][bestMove[1]] = player;

            if (checkWin(board, player, bestMove)) {
                winner = player;
                console.log(`Game Over. Winner: ${winner}`);
                break;
            }
            player = getOpponent(player);
        }

        if (winner) {
            if ((challengerIsBlack && winner === 'black') || (!challengerIsBlack && winner === 'white')) {
                challengerWins++;
                console.log('Challenger WINS');
            } else {
                console.log('Challenger LOSES');
            }
        }
    }

    console.log('\n--- Evaluation Finished ---');
    const winRate = challengerWins / NUM_GAMES;
    console.log(`Total Games: ${NUM_GAMES}`);
    console.log(`Challenger Wins: ${challengerWins}`);
    console.log(`Draws: ${draws}`);
    console.log(`Challenger Win Rate: ${(winRate * 100).toFixed(2)}%`);

    if (winRate > WIN_THRESHOLD) {
        console.log(`
PASSED! New model is stronger. Win rate exceeds threshold of ${(WIN_THRESHOLD * 100).toFixed(2)}%.`);
    } else {
        console.log(`
FAILED. New model is not significantly stronger. Win rate is below threshold.`);
    }
}

runEvaluation().catch(console.error);
