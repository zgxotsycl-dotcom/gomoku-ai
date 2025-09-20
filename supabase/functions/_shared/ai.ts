/**
 * @file Gomoku AI Logic - MCTS (Monte Carlo Tree Search)
 * This file contains the core AI engine, now upgraded with a Neural Network-guided MCTS.
 * This library is intended to be used in a Deno environment.
 */

import * as tf from '@tensorflow/tfjs-node';

// --- Type Definitions ---
export type Player = 'black' | 'white';
export type Move = [number, number];
export type PolicyData = { move: [number, number], visits: number };

// --- Constants ---
function getBoardSize(board: (Player | null)[][]): number { return board.length; }

// --- Helper Functions ---

export function getOpponent(player: Player): Player {
  return player === 'black' ? 'white' : 'black';
}

export function checkWin(board: (Player | null)[][], player: Player, move: [number, number]): boolean {
  if (!move || move[0] === -1) return false;
  const [r, c] = move;
  const BOARD_SIZE = getBoardSize(board);
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
    const BOARD_SIZE = getBoardSize(board);
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
    ], 2); // stack along the last axis to get shape [N, N, 3]

    return stackedChannels.expandDims(0) as tf.Tensor4D; // add batch dimension -> [1, N, N, 3]
}

export function getPossibleMoves(board: (Player|null)[][], radius = 1): Move[] {
  const BOARD_SIZE = getBoardSize(board);
  const moves = new Set<string>();
  let hasStones = false;
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (let c = 0; c < BOARD_SIZE; c++) {
      if (board[r][c] !== null) {
        hasStones = true;
        for (let i = -radius; i <= radius; i++) {
          for (let j = -radius; j <= radius; j++) {
            const nr = r + i, nc = c + j;
            if (nr >= 0 && nr < BOARD_SIZE && nc >= 0 && nc < BOARD_SIZE && board[nr][nc] === null) {
              moves.add(`${nr},${nc}`);
            }
          }
        }
      }
    }
  }
  if (!hasStones) return [[Math.floor(BOARD_SIZE/2), Math.floor(BOARD_SIZE/2)]];
  return [...moves].map(s => {
    const [r, c] = s.split(',').map(Number);
    return [r, c] as Move;
  });
}

// --- NN-Guided MCTS Implementation ---

function hasEmpty(board: (Player|null)[][]): boolean {
  const BOARD_SIZE = getBoardSize(board);
  for (let r = 0; r < BOARD_SIZE; r++)
    for (let c = 0; c < BOARD_SIZE; c++)
      if (board[r][c] === null) return true;
  return false;
}

class MCTSNodeNN {
    parent: MCTSNodeNN | null;
    children: { [moveIndex: number]: MCTSNodeNN };
    player: Player;
    move: Move | null;
    prior: number;
    visits: number;
    valueSum: number;
    depth: number;

    constructor(player: Player, parent: MCTSNodeNN | null = null, move: Move | null = null, prior = 0) {
        this.player = player; this.parent = parent; this.move = move;
        this.children = {}; this.visits = 0; this.valueSum = 0; this.prior = prior;
        this.depth = parent ? parent.depth + 1 : 0;
    }

    get value(): number {
        return this.visits === 0 ? 0 : this.valueSum / this.visits;
    }

    selectChild(): MCTSNodeNN | null {
      const c_puct = this.depth < 20 ? 2.0 : 1.5;
      const kids = Object.values(this.children);
      if (kids.length === 0) return null;
      const sqrtN = Math.sqrt(Math.max(1, this.visits));
      let best: MCTSNodeNN | null = null;
      let bestScore = -Infinity;
      for (const child of kids) {
        const qFromParent = -child.value; // 관점 변환
        const u = c_puct * child.prior * (sqrtN / (1 + child.visits));
        const score = qFromParent + u;
        if (score > bestScore) { bestScore = score; best = child; }
      }
      return best;
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

export async function findBestMoveNN(model: tf.LayersModel, board: (Player | null)[][], player: Player, timeLimit: number): Promise<{ bestMove: Move, policy: PolicyData[] }> {
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
        let node: MCTSNodeNN | null = root;

        while (Object.keys(node.children).length > 0) {
            node = node.selectChild();
            if (!node) break;
            currentBoard[node.move![0]][node.move![1]] = node.parent!.player;
        }
        if (!node) continue; // Should not happen if logic is correct

        const lastMove = node.move;
        const lastPlayer = node.parent?.player;
        let value: number;

        if (lastMove && lastPlayer && checkWin(currentBoard, lastPlayer, lastMove)) {
            value = -1; // The parent player won, so this node is a loss for the current player.
        } else if (!hasEmpty(currentBoard)) {
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

    let bestMove: Move | null = null;
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

// Zobrist hashing removed for dynamic board size (unused)

export function findThreats(board: (Player | null)[][], player: Player, threatType: 'open-four' | 'four'): Move[] {
    const BOARD_SIZE = getBoardSize(board);
    const threats: Move[] = [];
    const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];

    for (let r_start = 0; r_start < BOARD_SIZE; r_start++) {
        for (let c_start = 0; c_start < BOARD_SIZE; c_start++) {
            for (const [dr, dc] of directions) {
                const pos: Move[] = [];
                let playerCount = 0;
                let emptyCell: Move | null = null;

                // Create a line of 5
                for (let i = 0; i < 5; i++) {
                    const r = r_start + i * dr;
                    const c = c_start + i * dc;
                    if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) {
                        playerCount = -1;
                        break;
                    }
                    pos.push([r, c]);
                    const cell = board[r][c];
                    if (cell === player) {
                        playerCount++;
                    } else if (cell === null) {
                        if (emptyCell !== null) { // more than one empty cell
                            playerCount = -1;
                            break;
                        }
                        emptyCell = [r, c];
                    } else { // opponent
                        playerCount = -1;
                        break;
                    }
                }

                if (playerCount === 4 && emptyCell) {
                    if (threatType === 'four') {
                        threats.push(emptyCell);
                    } else if (threatType === 'open-four') {
                        const start = pos[0];
                        const end = pos[4];
                        const beforeR = start[0] - dr;
                        const beforeC = start[1] - dc;
                        const afterR = end[0] + dr;
                        const afterC = end[1] + dc;
                        
                        const openEnds = (board[beforeR]?.[beforeC] === null) && (board[afterR]?.[afterC] === null);
                        if (openEnds) {
                            threats.push(emptyCell);
                        }
                    }
                }
            }
        }
    }
    return threats;
}
