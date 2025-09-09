import { parentPort, workerData } from 'node:worker_threads';
import * as tf from '@tensorflow/tfjs-node-gpu';
import { findBestMoveNN } from './src/ai';
import type { Player } from './src/ai';

const BOARD_SIZE = 19;
const MODEL_PATH = './gomoku_model';
const MCTS_THINK_TIME = 2000; // 2 seconds per move in self-play
const EXPLORATION_MOVES = 15; // Number of moves to use temperature sampling for exploration

let model: tf.LayersModel | null = null;

async function loadModel() {
    if (model) return model;
    console.log(`[Worker ${workerData.workerId}] Loading model...`);
    model = await tf.loadLayersModel(`file://${MODEL_PATH}/model.json`);
    console.log(`[Worker ${workerData.workerId}] Model loaded.`);
    return model;
}

function checkWin(board: (Player | null)[][], player: Player, move: [number, number]): boolean {
    if (!move || move[0] === -1) return false;
    const [r, c] = move;
    const directions = [[[0, 1], [0, -1]], [[1, 0], [-1, 0]], [[1, 1], [-1, -1]], [[-1, 1], [1, -1]]];
    for (const dir of directions) {
        let count = 1;
        for (const [dr, dc] of dir) {
            for (let i = 1; i < 5; i++) {
                const newR = r + dr * i, newC = c + dc * i;
                if (newR >= 0 && newR < BOARD_SIZE && newC >= 0 && newC < BOARD_SIZE && board[newR][newC] === player) {
                    count++;
                } else { break; }
            }
        }
        if (count >= 5) return true;
    }
    return false;
}

function getOpponent(player: Player): Player {
    return player === 'black' ? 'white' : 'black';
}

async function runSelfPlayGame() {
    if (!model) {
        console.error(`[Worker ${workerData.workerId}] Model not loaded! Exiting.`);
        return;
    }

    let board: (Player | null)[][] = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(null));
    let player: Player = 'black';
    const history: { state: (Player | null)[][], policy: number[], player: Player }[] = [];

    for (let moveCount = 0; moveCount < (BOARD_SIZE * BOARD_SIZE); moveCount++) {
        const { bestMove, policy: mctsPolicy } = await findBestMoveNN(model, board, player, MCTS_THINK_TIME);

        if (!bestMove || bestMove[0] === -1) break; // No more moves

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
        // Check if there is more than one move to choose from for exploration
        if (moveCount < EXPLORATION_MOVES && mctsPolicy.length > 1) {
            // --- Exploration: Use temperature sampling --- 
            const moves = mctsPolicy.map(p => p.move);
            const probabilities = mctsPolicy.map(p => p.visits / totalVisits);

            // Create a 2D tensor of log-probabilities directly to ensure type safety.
            const logits = tf.tidy(() => tf.tensor2d([probabilities]).log());

            const moveIndexTensor = tf.multinomial(logits, 1);
            const moveIndex = moveIndexTensor.dataSync()[0];
            chosenMove = moves[moveIndex];

            // Dispose intermediate tensors
            logits.dispose();
            moveIndexTensor.dispose();
        } else {
            // --- Exploitation: Use the best move ---
            chosenMove = bestMove;
        }

        board[chosenMove[0]][chosenMove[1]] = player;

        if (checkWin(board, player, chosenMove)) {
            const winner = player;
            const trainingSamples = history.map(h => ({
                ...h,
                value: h.player === winner ? 1 : -1,
            }));
            parentPort?.postMessage({ trainingSamples });
            return;
        }

        player = getOpponent(player);
    }
    
    // Draw
    const trainingSamples = history.map(h => ({ ...h, value: 0 }));
    parentPort?.postMessage({ trainingSamples });
}

parentPort?.on('message', async (msg) => {
    if (msg === 'start_new_game') {
        try {
            await runSelfPlayGame();
        } catch (e) {
            console.error(`[Worker ${workerData.workerId}] Error during self-play:`, e);
        }
    }
});

loadModel().then(() => {
    runSelfPlayGame();
}).catch(e => {
    console.error(`[Worker ${workerData.workerId}] Failed to initialize model:`, e);
});