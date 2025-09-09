"use strict";
/**
 * @file Standalone Self-Play Worker
 * This script runs in an infinite loop to continuously play games against itself
 * using the current best model, and saves the generated training data to a replay buffer.
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
const ai_1 = require("./src/ai");
// --- Configuration ---
const MAIN_MODEL_PATH = './model_main';
const REPLAY_BUFFER_PATH = './replay_buffer';
const BOARD_SIZE = 19;
const MCTS_THINK_TIME = 2000; // 2 seconds per move
const EXPLORATION_MOVES = 15; // Number of moves to use temperature sampling for exploration
// --- Worker Setup ---
const workerId = process.argv[2] || '0'; // Get worker ID from command-line argument
let model = null;
async function loadModel() {
    console.log(`[Worker ${workerId}] Loading model from ${MAIN_MODEL_PATH}...`);
    try {
        return await tf.loadLayersModel(`file://${path.resolve(MAIN_MODEL_PATH)}/model.json`);
    }
    catch (e) {
        console.error(`[Worker ${workerId}] Could not load model. Error: ${e}`);
        console.log(`[Worker ${workerId}] Waiting for model to become available...`);
        await new Promise(resolve => setTimeout(resolve, 10000));
        return loadModel(); // Retry recursively
    }
}
async function runSingleGame() {
    if (!model)
        throw new Error('Model is not loaded.');
    let board = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(null));
    let player = 'black';
    const history = [];
    for (let moveCount = 0; moveCount < (BOARD_SIZE * BOARD_SIZE); moveCount++) {
        const { bestMove, policy: mctsPolicy } = await (0, ai_1.findBestMoveNN)(model, board, player, MCTS_THINK_TIME);
        if (!bestMove || bestMove[0] === -1)
            break;
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
        let chosenMove;
        // During exploration phase, if there are multiple moves, sample one.
        if (moveCount < EXPLORATION_MOVES && mctsPolicy.length > 1) {
            chosenMove = tf.tidy(() => {
                const moves = mctsPolicy.map(p => p.move);
                const probabilities = mctsPolicy.map(p => p.visits / totalVisits);
                const logits = tf.tensor1d(probabilities).log();
                const moveIndex = tf.multinomial(logits, 1).dataSync()[0];
                return moves[moveIndex];
            });
        }
        else {
            // If there's only one move, or we are past the exploration phase, take the best move.
            chosenMove = bestMove;
        }
        if (!chosenMove)
            break; // Should not happen, but as a safeguard.
        board[chosenMove[0]][chosenMove[1]] = player;
        if ((0, ai_1.checkWin)(board, player, chosenMove)) {
            return history.map(h => ({ ...h, value: h.player === player ? 1 : -1 }));
        }
        player = (0, ai_1.getOpponent)(player);
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
        }
        catch (e) {
            console.error(`[Worker ${workerId}] An error occurred during game loop:`, e);
            await new Promise(resolve => setTimeout(resolve, 5000));
        }
    }
}
main();
