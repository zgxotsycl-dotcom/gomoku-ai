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
const tf = __importStar(require("@tensorflow/tfjs-node")); // Use tfjs-node
const worker_threads_1 = require("worker_threads");
const path = __importStar(require("path"));
const ai_1 = require("./ai");
if (!worker_threads_1.parentPort) {
    throw new Error("This script must be run as a worker thread.");
}
let model = null;
const BOARD_SIZE = 19;
async function initialize() {
    if (model)
        return;
    console.log('Worker: Initializing TensorFlow backend...');
    // tfjs-node uses the native C++ backend by default. No need to set it.
    await tf.ready();
    console.log(`Worker backend: ${tf.getBackend()}`);
    const modelJsonPath = path.resolve(__dirname, '../../gomoku_model/model.json');
    console.log(`Worker: Loading model from ${modelJsonPath}`);
    // Use the fileSystem IO handler from tfjs-node
    model = await tf.loadLayersModel(tf.io.fileSystem(modelJsonPath));
    console.log("Worker model loaded successfully.");
}
function createPolicyVector(policy, boardSize) {
    const vector = Array(boardSize * boardSize).fill(0);
    if (!policy || policy.length === 0)
        return vector;
    let totalVisits = 0;
    for (const p of policy)
        totalVisits += p.visits;
    if (totalVisits > 0) {
        for (const p of policy) {
            const [r, c] = p.move;
            vector[r * boardSize + c] = p.visits / totalVisits;
        }
    }
    return vector;
}
async function playGame() {
    if (!model) {
        console.error("Worker model is not loaded. Cannot start game.");
        return;
    }
    const THINK_TIME_MS = 250;
    let board = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(null));
    let currentPlayer = 'black';
    let winner = null;
    const gameHistory = [];
    while (winner === null && gameHistory.length < BOARD_SIZE * BOARD_SIZE) {
        const { bestMove, policy } = await (0, ai_1.findBestMoveNN)(model, board, currentPlayer, THINK_TIME_MS);
        if (!bestMove || bestMove[0] === -1) {
            winner = 'Draw';
            break;
        }
        gameHistory.push({ board: board.map(row => [...row]), player: currentPlayer, policy });
        const [row, col] = bestMove;
        if (board[row][col] !== null) {
            winner = 'Draw';
            break;
        }
        board[row][col] = currentPlayer;
        if ((0, ai_1.checkWin)(board, currentPlayer, bestMove)) {
            winner = currentPlayer;
            break;
        }
        currentPlayer = (0, ai_1.getOpponent)(currentPlayer);
    }
    if (winner === null)
        winner = 'Draw';
    const trainingSamples = gameHistory.map(h => ({
        state: h.board,
        policy: createPolicyVector(h.policy, BOARD_SIZE),
        value: winner === 'Draw' ? 0 : (h.player === winner ? 1 : -1),
        player: h.player
    }));
    worker_threads_1.parentPort.postMessage({ trainingSamples });
}
// Initialize model on startup, then start the first game.
(async () => {
    try {
        await initialize();
        await playGame(); // Start the first game automatically
    }
    catch (error) {
        console.error("Critical error during worker initialization or first game:", error);
        process.exit(1); // Terminate the worker if it fails to initialize
    }
})();
// Listen for messages to start subsequent games
worker_threads_1.parentPort.on("message", async (e) => {
    if (e === 'start_new_game') {
        try {
            await playGame();
        }
        catch (error) {
            console.error("Error in worker during subsequent game:", error);
        }
    }
});
