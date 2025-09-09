"use strict";
/**
 * @file Gomoku AI Logic - MCTS (Monte Carlo Tree Search) (Memory Safe)
 * This file contains the core AI engine, now upgraded with a Neural Network-guided MCTS.
 * This version includes tf.tidy() to prevent GPU memory leaks during MCTS simulations.
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
exports.getOpponent = getOpponent;
exports.checkWin = checkWin;
exports.getPossibleMoves = getPossibleMoves;
exports.findBestMoveNN = findBestMoveNN;
const tf = __importStar(require("@tensorflow/tfjs-node-gpu"));
// --- Constants ---
const BOARD_SIZE = 19;
// --- Helper Functions ---
function getOpponent(player) {
    return player === 'black' ? 'white' : 'black';
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
function getPossibleMoves(board, radius = 1) {
    const moves = new Set();
    let hasStones = false;
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            if (board[r][c] !== null) {
                hasStones = true;
                for (let i = -radius; i <= radius; i++) {
                    for (let j = -radius; j <= radius; j++) {
                        const newR = r + i;
                        const newC = c + j;
                        if (newR >= 0 && newR < BOARD_SIZE && newC >= 0 && newC < BOARD_SIZE && board[newR][newC] === null) {
                            moves.add(`${newR},${newC}`);
                        }
                    }
                }
            }
        }
    }
    if (!hasStones)
        return [[Math.floor(BOARD_SIZE / 2), Math.floor(BOARD_SIZE / 2)]];
    return Array.from(moves).map(move => {
        const [r, c] = move.split(',').map(Number);
        return [r, c];
    });
}
function boardToInputTensor(board, player) {
    return tf.tidy(() => {
        const opponent = player === 'black' ? 'white' : 'black';
        const playerChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));
        const opponentChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));
        for (let r = 0; r < BOARD_SIZE; r++) {
            for (let c = 0; c < BOARD_SIZE; c++) {
                if (board[r][c] === player)
                    playerChannel[r][c] = 1;
                else if (board[r][c] === opponent)
                    opponentChannel[r][c] = 1;
            }
        }
        const colorChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(player === 'black' ? 1 : 0));
        const stackedChannels = tf.stack([
            tf.tensor2d(playerChannel),
            tf.tensor2d(opponentChannel),
            tf.tensor2d(colorChannel)
        ], 2);
        return stackedChannels.expandDims(0);
    });
}
// --- NN-Guided MCTS Implementation ---
class MCTSNodeNN {
    constructor(player, parent = null, move = null, prior = 0) {
        this.player = player;
        this.parent = parent;
        this.move = move;
        this.children = {};
        this.visits = 0;
        this.valueSum = 0;
        this.prior = prior;
    }
    get value() {
        return this.visits === 0 ? 0 : this.valueSum / this.visits;
    }
    selectChild() {
        const c_puct = 1.5;
        let bestScore = -Infinity;
        let bestChild = null;
        const sqrtTotalVisits = Math.sqrt(this.visits);
        for (const child of Object.values(this.children)) {
            const puctScore = child.value + c_puct * child.prior * (sqrtTotalVisits / (1 + child.visits));
            if (puctScore > bestScore) {
                bestScore = puctScore;
                bestChild = child;
            }
        }
        return bestChild;
    }
    expand(board, policy) {
        const possibleMoves = getPossibleMoves(board);
        for (const move of possibleMoves) {
            const [r, c] = move;
            const moveIndex = r * BOARD_SIZE + c;
            if (policy[moveIndex] > 0 && !(moveIndex in this.children)) {
                this.children[moveIndex] = new MCTSNodeNN(getOpponent(this.player), this, move, policy[moveIndex]);
            }
        }
    }
    backpropagate(value) {
        let node = this;
        while (node) {
            node.visits++;
            node.valueSum += value;
            value = -value;
            node = node.parent;
        }
    }
}
async function findBestMoveNN(model, board, player, timeLimit) {
    const startTime = Date.now();
    const root = new MCTSNodeNN(player);
    const rootPolicyData = tf.tidy(() => {
        const rootInputTensor = boardToInputTensor(board, player);
        const [rootPolicyTensor] = model.predict(rootInputTensor);
        return rootPolicyTensor.dataSync();
    });
    root.expand(board, rootPolicyData);
    while (Date.now() - startTime < timeLimit) {
        const currentBoard = board.map(row => [...row]);
        let node = root;
        while (Object.keys(node.children).length > 0) {
            node = node.selectChild();
            currentBoard[node.move[0]][node.move[1]] = node.parent.player;
        }
        const lastMove = node.move;
        const lastPlayer = node.parent?.player;
        let value;
        if (lastMove && lastPlayer && checkWin(currentBoard, lastPlayer, lastMove)) {
            value = -1;
        }
        else if (getPossibleMoves(currentBoard, 2).length === 0) {
            value = 0;
        }
        else {
            const { policy, value: predictedValue } = tf.tidy(() => {
                const inputTensor = boardToInputTensor(currentBoard, node.player);
                const [policyTensor, valueTensor] = model.predict(inputTensor);
                return {
                    policy: policyTensor.dataSync(),
                    value: valueTensor.dataSync()[0]
                };
            });
            value = predictedValue;
            node.expand(currentBoard, policy);
        }
        node.backpropagate(value);
    }
    if (Object.keys(root.children).length === 0) {
        const moves = getPossibleMoves(board);
        return { bestMove: moves[0] || [-1, -1], policy: [] };
    }
    let bestMove = null;
    let maxVisits = -1;
    for (const child of Object.values(root.children)) {
        if (child.visits > maxVisits) {
            maxVisits = child.visits;
            bestMove = child.move;
        }
    }
    return {
        bestMove: bestMove,
        policy: Object.values(root.children).map(child => ({ move: child.move, visits: child.visits }))
    };
}
