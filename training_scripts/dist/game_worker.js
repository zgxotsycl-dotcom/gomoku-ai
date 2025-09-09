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
const node_worker_threads_1 = require("node:worker_threads");
const tf = __importStar(require("@tensorflow/tfjs-node-gpu"));
const ai_1 = require("./src/ai");
const BOARD_SIZE = 19;
const MODEL_PATH = './gomoku_model';
const MCTS_THINK_TIME = 2000; // 셀프플레이에서는 한 수당 2초씩 생각
let model = null;
async function loadModel() {
    if (model)
        return model;
    console.log(`[Worker ${node_worker_threads_1.workerData.workerId}] Loading model...`);
    model = await tf.loadLayersModel(`file://${MODEL_PATH}/model.json`);
    console.log(`[Worker ${node_worker_threads_1.workerData.workerId}] Model loaded.`);
    return model;
}
function checkWin(board, player, move) {
    if (!move || move[0] === -1)
        return false;
    const [r, c] = move;
    const directions = [[[0, 1], [0, -1]], [[1, 0], [-1, 0]], [[1, 1], [-1, -1]], [[-1, 1], [1, -1]]];
    for (const dir of directions) {
        let count = 1;
        for (const [dr, dc] of dir) {
            for (let i = 1; i < 5; i++) {
                const newR = r + dr * i;
                const newC = c + dc * i;
                if (newR >= 0 && newR < BOARD_SIZE && newC >= 0 && newC < BOARD_SIZE && board[newR][newC] === player) {
                    count++;
                }
                else {
                    break;
                }
            }
        }
        if (count >= 5)
            return true;
    }
    return false;
}
function getOpponent(player) {
    return player === 'black' ? 'white' : 'black';
}
// --- Main Self-Play Game Logic ---
async function runSelfPlayGame() {
    if (!model) {
        console.error(`[Worker ${node_worker_threads_1.workerData.workerId}] Model not loaded! Exiting.`);
        return;
    }
    let board = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(null));
    let player = 'black';
    const history = [];
    for (let i = 0; i < (BOARD_SIZE * BOARD_SIZE); i++) {
        const { bestMove, policy: mctsPolicy } = await (0, ai_1.findBestMoveNN)(model, board, player, MCTS_THINK_TIME);
        if (!bestMove || bestMove[0] === -1)
            break; // No more moves
        // Create the policy target for training
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
        board[bestMove[0]][bestMove[1]] = player;
        if (checkWin(board, player, bestMove)) {
            // Game ended, assign results and send back
            const winner = player;
            const trainingSamples = history.map(h => ({
                ...h,
                value: h.player === winner ? 1 : -1, // Value from the perspective of the player in that state
            }));
            node_worker_threads_1.parentPort?.postMessage({ trainingSamples });
            return;
        }
        player = getOpponent(player);
    }
    // Draw
    const trainingSamples = history.map(h => ({ ...h, value: 0 }));
    node_worker_threads_1.parentPort?.postMessage({ trainingSamples });
}
// --- Worker Initialization ---
node_worker_threads_1.parentPort?.on('message', async (msg) => {
    if (msg === 'start_new_game') {
        try {
            await runSelfPlayGame();
        }
        catch (e) {
            console.error(`[Worker ${node_worker_threads_1.workerData.workerId}] Error during self-play:`, e);
        }
    }
});
// Load the model once when the worker starts, then start the first game.
loadModel().then(() => {
    runSelfPlayGame();
}).catch(e => {
    console.error(`[Worker ${node_worker_threads_1.workerData.workerId}] Failed to initialize model:`, e);
});
