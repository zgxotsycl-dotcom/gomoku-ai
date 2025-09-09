/**
 * @file Gomoku AI Logic - MCTS (Monte Carlo Tree Search)
 * This file contains the core AI engine, now upgraded with a Neural Network-guided MCTS.
 * This library is intended to be used in a Deno environment.
 */

import * as tf from '@tensorflow/tfjs-node';

// --- Type Definitions ---
export type Player = 'black' | 'white';
export type PolicyData = { move: [number, number], visits: number };

// --- Constants ---
const BOARD_SIZE = 19;

// --- Helper Functions ---

export function getOpponent(player: Player): Player {
  return player === 'black' ? 'white' : 'black';
}

export function checkWin(board: (Player | null)[][], player: Player, move: [number, number]): boolean {
  if (!move || move[0] === -1) return false;
  const [r, c] = move;
  const directions = [[[0, 1], [0, -1]], [[1, 0], [-1, 0]], [[1, 1], [-1, -1]], [[-1, 1], [1, -1]]];
  for (const dir of directions) {
    let count = 1;
    for (const [dr, dc] of dir) {
      for (let i = 1; i < 5; i++) {
        const newR = r + dr * i; const newC = c + dc * i;
        if (newR >= 0 && newR < BOARD_SIZE && newC >= 0 && newC < BOARD_SIZE && board[newR][newC] === player) {
          count++;
        } else { break; }
      }
    }
    if (count >= 5) return true;
  }
  return false;
}

function boardToInputTensor(board: (Player | null)[][], player: Player): tf.Tensor4D {
    const opponent = player === 'black' ? 'white' : 'black';
    const playerChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));
    const opponentChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            if (board[r][c] === player) playerChannel[r][c] = 1;
            else if (board[r][c] === opponent) opponentChannel[r][c] = 1;
        }
    }
    const colorChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(player === 'black' ? 1 : 0));
    
    const stackedChannels = tf.stack([
        tf.tensor2d(playerChannel),
        tf.tensor2d(opponentChannel),
        tf.tensor2d(colorChannel)
    ], 2); // stack along the last axis to get shape [19, 19, 3]

    return stackedChannels.expandDims(0) as tf.Tensor4D; // add batch dimension -> [1, 19, 19, 3]
}

export function getPossibleMoves(board: (Player | null)[][], radius = 1): [number, number][] {
    const moves: Set<string> = new Set();
    let hasStones = false;
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            if (board[r][c] !== null) {
                hasStones = true;
                for (let i = -radius; i <= radius; i++) {
                    for (let j = -radius; j <= radius; j++) {
                        const newR = r + i; const newC = c + j;
                        if (newR >= 0 && newR < BOARD_SIZE && newC >= 0 && newC < BOARD_SIZE && board[newR][newC] === null) {
                            moves.add(`${newR},${newC}`);
                        }
                    }
                }
            }
        }
    }
    if (!hasStones) return [[Math.floor(BOARD_SIZE / 2), Math.floor(BOARD_SIZE / 2)]];
    return Array.from(moves).map(move => {
        const [r, c] = move.split(',').map(Number);
        return [r, c] as [number, number];
    });
}

// --- NN-Guided MCTS Implementation ---

class MCTSNodeNN {
    parent: MCTSNodeNN | null;
    children: { [moveIndex: number]: MCTSNodeNN };
    player: Player;
    move: [number, number] | null;
    prior: number;
    visits: number;
    valueSum: number;

    constructor(player: Player, parent: MCTSNodeNN | null = null, move: [number, number] | null = null, prior = 0) {
        this.player = player; this.parent = parent; this.move = move;
        this.children = {}; this.visits = 0; this.valueSum = 0; this.prior = prior;
    }

    get value(): number {
        return this.visits === 0 ? 0 : this.valueSum / this.visits;
    }

    selectChild(): MCTSNodeNN {
        const c_puct = 1.5;
        let bestScore = -Infinity;
        let bestChild: MCTSNodeNN | null = null;
        const sqrtTotalVisits = Math.sqrt(this.visits);
        for (const child of Object.values(this.children)) {
            const puctScore = child.value + c_puct * child.prior * (sqrtTotalVisits / (1 + child.visits));
            if (puctScore > bestScore) { bestScore = puctScore; bestChild = child; }
        }
        return bestChild!;
    }

    expand(board: (Player | null)[][], policy: Float32Array): void {
        const possibleMoves = getPossibleMoves(board);
        for (const move of possibleMoves) {
            const [r, c] = move;
            const moveIndex = r * BOARD_SIZE + c;
            if (policy[moveIndex] > 0 && !(moveIndex in this.children)) {
                this.children[moveIndex] = new MCTSNodeNN(getOpponent(this.player), this, move, policy[moveIndex]);
            }
        }
    }

    backpropagate(value: number): void {
        let node: MCTSNodeNN | null = this;
        while (node) {
            node.visits++; node.valueSum += value;
            value = -value;
            node = node.parent;
        }
    }
}

export async function findBestMoveNN(model: tf.LayersModel, board: (Player | null)[][], player: Player, timeLimit: number): Promise<{ bestMove: [number, number], policy: PolicyData[] }> {
    const startTime = Date.now();
    const root = new MCTSNodeNN(player);

    // Initial prediction for the root node
    const rootInputTensor = boardToInputTensor(board, player);
    const [rootPolicyTensor, rootValueTensor] = model.predict(rootInputTensor) as tf.Tensor[];
    const rootPolicy = await rootPolicyTensor.data() as Float32Array;
    tf.dispose([rootInputTensor, rootPolicyTensor, rootValueTensor]);
    root.expand(board, rootPolicy);

    let simulationCount = 0;
    while (Date.now() - startTime < timeLimit) {
        const currentBoard = board.map(row => [...row]);
        let node = root;

        while (Object.keys(node.children).length > 0) {
            node = node.selectChild();
            currentBoard[node.move![0]][node.move![1]] = node.parent!.player;
        }

        const lastMove = node.move;
        const lastPlayer = node.parent?.player;
        let value: number;

        if (lastMove && lastPlayer && checkWin(currentBoard, lastPlayer, lastMove)) {
            value = -1; // The parent player won, so this node is a loss for the current player.
        } else if (getPossibleMoves(currentBoard, 2).length === 0) {
            value = 0; // Draw
        } else {
            const inputTensor = boardToInputTensor(currentBoard, node.player);
            const [policyTensor, valueTensor] = model.predict(inputTensor) as tf.Tensor[];
            const policy = await policyTensor.data() as Float32Array;
            value = (await valueTensor.data() as Float32Array)[0];
            tf.dispose([inputTensor, policyTensor, valueTensor]);
            node.expand(currentBoard, policy);
        }

        node.backpropagate(value);
        simulationCount++;
    }

    if (Object.keys(root.children).length === 0) {
        const moves = getPossibleMoves(board);
        return { bestMove: moves.length > 0 ? moves[0] : [-1, -1], policy: [] };
    }

    let bestMove: [number, number] | null = null;
    let maxVisits = -1;
    for (const child of Object.values(root.children)) {
        if (child.visits > maxVisits) {
            maxVisits = child.visits;
            bestMove = child.move;
        }
    }

    return {
        bestMove: bestMove!,
        policy: Object.values(root.children).map(child => ({ move: child.move!, visits: child.visits }))
    };
}
