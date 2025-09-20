import tf from './tf';
import { getNumber, mixPolicies } from './tuning';
import type * as TFT from '@tensorflow/tfjs';

export type Player = 'black' | 'white';
export type Move = [number, number];
export type PolicyData = { move: Move; visits: number };

// Derive board size dynamically from input to support non-19 boards during experiments.
function getBoardSize(board: (Player | null)[][]): number {
  return board.length;
}

export function getOpponent(player: Player): Player {
  return player === 'black' ? 'white' : 'black';
}

export function checkWin(
  board: (Player | null)[][],
  player: Player,
  move: Move
): boolean {
  if (!move || move[0] === -1) return false;
  const size = getBoardSize(board);
  const [r, c] = move;
  const directions: [number, number][][] = [
    [
      [0, 1],
      [0, -1],
    ],
    [
      [1, 0],
      [-1, 0],
    ],
    [
      [1, 1],
      [-1, -1],
    ],
    [
      [-1, 1],
      [1, -1],
    ],
  ];
  for (const dir of directions) {
    let count = 1;
    for (const [dr, dc] of dir) {
      for (let i = 1; i < 5; i++) {
        const newR = r + dr * i;
        const newC = c + dc * i;
        if (
          newR >= 0 &&
          newR < size &&
          newC >= 0 &&
          newC < size &&
          board[newR][newC] === player
        ) {
          count++;
        } else {
          break;
        }
      }
    }
    if (count >= 5) return true;
  }
  return false;
}

function boardToInputTensor(
  board: (Player | null)[][],
  player: Player
): TFT.Tensor4D {
  const size = getBoardSize(board);
  const opponent = player === 'black' ? 'white' : 'black';
  const playerChannel = Array(size)
    .fill(0)
    .map(() => Array(size).fill(0));
  const opponentChannel = Array(size)
    .fill(0)
    .map(() => Array(size).fill(0));
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      if (board[r][c] === player) playerChannel[r][c] = 1;
      else if (board[r][c] === opponent) opponentChannel[r][c] = 1;
    }
  }
  const colorChannel = Array(size)
    .fill(0)
    .map(() => Array(size).fill(player === 'black' ? 1 : 0));

  const stackedChannels = tf.stack(
    [tf.tensor2d(playerChannel), tf.tensor2d(opponentChannel), tf.tensor2d(colorChannel)],
    2
  ); // [H, W, 3]
  return stackedChannels.expandDims(0) as TFT.Tensor4D; // [1, H, W, 3]
}

// --- Symmetry transforms for root augmentation ---
type TransformId = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7; // I, R90, R180, R270, FlipH, FlipV, Diag, AntiDiag

function transformRC(r: number, c: number, size: number, t: TransformId): [number, number] {
  const n = size;
  switch (t) {
    case 0: return [r, c];
    case 1: return [c, n - 1 - r]; // rot90
    case 2: return [n - 1 - r, n - 1 - c]; // rot180
    case 3: return [n - 1 - c, r]; // rot270
    case 4: return [r, n - 1 - c]; // flipH
    case 5: return [n - 1 - r, c]; // flipV
    case 6: return [c, r]; // transpose main diag
    case 7: return [n - 1 - c, n - 1 - r]; // anti diag
  }
}

function transformBoard(board: (Player | null)[][], t: TransformId): (Player | null)[][] {
  const n = getBoardSize(board);
  const out: (Player | null)[][] = Array.from({ length: n }, () => Array<Player | null>(n).fill(null));
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const [rr, cc] = transformRC(r, c, n, t);
      out[rr][cc] = board[r][c];
    }
  }
  return out;
}

async function predictPolicyValue(
  model: TFT.LayersModel,
  board: (Player | null)[][],
  player: Player
): Promise<{ policy: Float32Array; value: number }> {
  const input = boardToInputTensor(board, player);
  const [pT, vT] = model.predict(input) as TFT.Tensor[];
  const policy = (await pT.data()) as Float32Array;
  const value = (await vT.data())[0] as number;
  tf.dispose([input, pT, vT]);
  return { policy, value };
}

async function predictRootWithSymmetry(
  model: TFT.LayersModel,
  board: (Player | null)[][],
  player: Player,
  symCount: number
): Promise<{ policy: Float32Array; value: number }> {
  const size = getBoardSize(board);
  const trans: TransformId[] = (symCount >= 8 ? [0,1,2,3,4,5,6,7] : symCount >= 4 ? [0,1,2,3] : [0]);
  const aggPolicy = new Float64Array(size * size);
  let aggValue = 0;
  for (const t of trans) {
    const b = t === 0 ? board : transformBoard(board, t);
    const { policy, value } = await predictPolicyValue(model, b, player);
    aggValue += value;
    // inverse map policy to original orientation
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const [rr, cc] = transformRC(r, c, size, t);
        const idxT = rr * size + cc;
        const idx = r * size + c;
        aggPolicy[idx] += policy[idxT] || 0;
      }
    }
  }
  // average
  for (let i = 0; i < aggPolicy.length; i++) aggPolicy[i] /= trans.length;
  const outPolicy = new Float32Array(aggPolicy.length);
  for (let i = 0; i < outPolicy.length; i++) outPolicy[i] = aggPolicy[i];
  const outValue = aggValue / trans.length;
  return { policy: outPolicy, value: outValue };
}

export function getPossibleMoves(
  board: (Player | null)[][],
  radius = 1
): Move[] {
  const size = getBoardSize(board);
  const moves = new Set<string>();
  let hasStones = false;
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      if (board[r][c] !== null) {
        hasStones = true;
        for (let i = -radius; i <= radius; i++) {
          for (let j = -radius; j <= radius; j++) {
            const nr = r + i,
              nc = c + j;
            if (
              nr >= 0 &&
              nr < size &&
              nc >= 0 &&
              nc < size &&
              board[nr][nc] === null
            ) {
              moves.add(`${nr},${nc}`);
            }
          }
        }
      }
    }
  }
  if (!hasStones) {
    const mid = Math.floor(size / 2);
    return [[mid, mid]];
  }
  return [...moves].map((s) => {
    const [r, c] = s.split(',').map(Number);
    return [r, c] as Move;
  });
}

// Detect moves that create a connected 3 (contiguous) with at least one open end
function listConnectedThreeMakers(
  board: (Player | null)[][],
  player: Player,
  candidates: Move[]
): Move[] {
  const size = getBoardSize(board);
  const dirs: [number, number][] = [
    [0, 1],
    [1, 0],
    [1, 1],
    [1, -1],
  ];
  const out: Move[] = [];
  for (const [r0, c0] of candidates) {
    if (board[r0][c0] !== null) continue;
    board[r0][c0] = player;
    let ok = false;
    for (const [dr, dc] of dirs) {
      let cnt = 1;
      let openL = false, openR = false;
      let r = r0 + dr, c = c0 + dc;
      while (r >= 0 && r < size && c >= 0 && c < size && board[r][c] === player) { cnt++; r += dr; c += dc; }
      if (r >= 0 && r < size && c >= 0 && c < size && board[r][c] === null) openR = true;
      r = r0 - dr; c = c0 - dc;
      while (r >= 0 && r < size && c >= 0 && c < size && board[r][c] === player) { cnt++; r -= dr; c -= dc; }
      if (r >= 0 && r < size && c >= 0 && c < size && board[r][c] === null) openL = true;
      if (cnt >= 3 && (openL || openR)) { ok = true; break; }
    }
    board[r0][c0] = null;
    if (ok) out.push([r0, c0]);
  }
  return out;
}

// Long-range link heuristic: bridge two friendly stones along a line within small gaps
function listLongLinkMakers(
  board: (Player | null)[][],
  player: Player,
  candidates: Move[]
): Move[] {
  const size = getBoardSize(board);
  const dirs: [number, number][] = [
    [0, 1],
    [1, 0],
    [1, 1],
    [1, -1],
  ];
  const out: Move[] = [];
  for (const [r0, c0] of candidates) {
    if (board[r0][c0] !== null) continue;
    let ok = false;
    for (const [dr, dc] of dirs) {
      const hasSide = (sgn: 1 | -1) => {
        for (let step = 1; step <= 3; step++) {
          const r = r0 + sgn * dr * step;
          const c = c0 + sgn * dc * step;
          if (r < 0 || r >= size || c < 0 || c >= size) break;
          const cell = board[r][c];
          if (cell === player) return true;
          if (cell !== null) break; // blocked
        }
        return false;
      };
      if (hasSide(1) && hasSide(-1)) { ok = true; break; }
    }
    if (ok) out.push([r0, c0]);
  }
  return out;
}

// --- Forbidden (3-3 / 4-4) detection and avoidance helpers ---
function cellAt(
  board: (Player | null)[][],
  r: number,
  c: number
): Player | null | 'OOB' {
  const n = getBoardSize(board);
  if (r < 0 || c < 0 || r >= n || c >= n) return 'OOB';
  return board[r][c];
}

function hasOpenThreeInDirection(
  board: (Player | null)[][],
  player: Player,
  r: number,
  c: number,
  dr: number,
  dc: number
): boolean {
  // Scan 6-length segments S0..S5 with S0/S5 empty and within S1..S4 exactly 3 stones and 1 empty (no opponent).
  // The placed stone must lie within S1..S4 so that the open-three was created by this move.
  for (let s = -5; s <= 0; s++) {
    const idxMove = -s; // index where (r,c) falls within S0..S5
    if (idxMove < 1 || idxMove > 4) continue; // ensure inside S1..S4
    const seg: (Player | null | 'OOB')[] = new Array(6);
    let oob = false;
    for (let i = 0; i < 6; i++) {
      const rr = r + (s + i) * dr;
      const cc = c + (s + i) * dc;
      const v = cellAt(board, rr, cc);
      if (v === 'OOB') { oob = true; break; }
      seg[i] = v;
    }
    if (oob) continue;
    if (!(seg[0] === null && seg[5] === null)) continue; // ends must be open
    let stones = 0, empties = 0; let hasOpp = false;
    for (let i = 1; i <= 4; i++) {
      const v = seg[i];
      if (v === player) stones++;
      else if (v === null) empties++;
      else { hasOpp = true; break; }
    }
    if (!hasOpp && stones === 3 && empties === 1) return true;
  }
  return false;
}

function hasFourInDirection(
  board: (Player | null)[][],
  player: Player,
  r: number,
  c: number,
  dr: number,
  dc: number
): boolean {
  // Scan 5-length segments S0..S4 that include (r,c) with exactly 4 stones + 1 empty and no opponent.
  for (let s = -4; s <= 0; s++) {
    const idxMove = -s;
    if (idxMove < 0 || idxMove > 4) continue;
    const seg: (Player | null | 'OOB')[] = new Array(5);
    let oob = false;
    for (let i = 0; i < 5; i++) {
      const rr = r + (s + i) * dr;
      const cc = c + (s + i) * dc;
      const v = cellAt(board, rr, cc);
      if (v === 'OOB') { oob = true; break; }
      seg[i] = v;
    }
    if (oob) continue;
    let stones = 0, empties = 0; let hasOpp = false;
    for (let i = 0; i < 5; i++) {
      const v = seg[i];
      if (v === player) stones++;
      else if (v === null) empties++;
      else { hasOpp = true; break; }
    }
    if (!hasOpp && stones === 4 && empties === 1) return true;
  }
  return false;
}

function createsDoubleOpenThree(board: (Player | null)[][], player: Player, r: number, c: number): boolean {
  const dirs: [number, number][] = [ [0,1], [1,0], [1,1], [1,-1] ];
  let cnt = 0;
  for (const [dr, dc] of dirs) {
    if (hasOpenThreeInDirection(board, player, r, c, dr, dc)) cnt++;
    if (cnt >= 2) return true;
  }
  return false;
}

function createsDoubleFour(board: (Player | null)[][], player: Player, r: number, c: number): boolean {
  const dirs: [number, number][] = [ [0,1], [1,0], [1,1], [1,-1] ];
  let cnt = 0;
  for (const [dr, dc] of dirs) {
    if (hasFourInDirection(board, player, r, c, dr, dc)) cnt++;
    if (cnt >= 2) return true;
  }
  return false;
}

function isForbiddenMove(board: (Player | null)[][], player: Player, move: Move): boolean {
  if (player !== 'black') return false; // forbidden rules apply to Black only
  const [r, c] = move;
  if (board[r][c] !== null) return false;
  board[r][c] = player;
  const wins = checkWin(board, player, move);
  if (wins) { board[r][c] = null; return false; }
  const is33 = createsDoubleOpenThree(board, player, r, c);
  const is44 = createsDoubleFour(board, player, r, c);
  board[r][c] = null;
  return is33 || is44;
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

  constructor(
    player: Player,
    parent: MCTSNodeNN | null = null,
    move: Move | null = null,
    prior = 0
  ) {
    this.player = player;
    this.parent = parent;
    this.move = move;
    this.children = {};
    this.visits = 0;
    this.valueSum = 0;
    this.prior = prior;
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
      const qFromParent = -child.value; // value is from child's perspective
      const u = c_puct * child.prior * (sqrtN / (1 + child.visits));
      const score = qFromParent + u;
      if (score > bestScore) {
        bestScore = score;
        best = child;
      }
    }
    return best;
  }

  expand(board: (Player | null)[][], policy: Float32Array, topK?: number): void {
    const size = getBoardSize(board);
    const possibleMoves = getPossibleMoves(board);
    const candidates: { idx: number; move: Move; p: number }[] = [];
    for (const move of possibleMoves) {
      const [r, c] = move;
      const idx = r * size + c;
      const p = policy[idx] ?? 0;
      if (p > 0) candidates.push({ idx, move, p });
    }
    if (topK && candidates.length > topK) {
      candidates.sort((a, b) => b.p - a.p);
    }
    const limit = topK ? Math.min(topK, candidates.length) : candidates.length;
    for (let i = 0; i < limit; i++) {
      const { idx, move, p } = candidates[i];
      if (!(idx in this.children)) {
        this.children[idx] = new MCTSNodeNN(getOpponent(this.player), this, move, p);
      }
    }
  }

  backpropagate(value: number): void {
    let node: MCTSNodeNN | null = this;
    while (node) {
      node.visits++;
      node.valueSum += value;
      value = -value;
      node = node.parent;
    }
  }
}

function hasEmpty(board: (Player | null)[][]): boolean {
  const size = getBoardSize(board);
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      if (board[r][c] === null) return true;
    }
  }
  return false;
}

// --- Transposition Table (TT) for MCTS ---
type TTEntry = { visits: number; value: number; priors: Float32Array | null };
const TT_CAP = Number(process.env.TT_CAP || 20000);
const tt = new Map<string, TTEntry>();

function ttKey(board: (Player | null)[][], player: Player): string {
  return canonicalKey(board, player);
}

function ttGet(board: (Player | null)[][], player: Player): TTEntry | null {
  const k = ttKey(board, player);
  const v = tt.get(k) || null;
  if (!v) return null;
  tt.delete(k); tt.set(k, v);
  return v;
}

function ttSet(board: (Player | null)[][], player: Player, entry: TTEntry): void {
  const k = ttKey(board, player);
  if (tt.has(k)) tt.delete(k);
  tt.set(k, entry);
  if (tt.size > TT_CAP) {
    const first = tt.keys().next().value as string | undefined;
    if (first) tt.delete(first);
  }
}

function ttAccumulate(board: (Player | null)[][], player: Player, value: number): void {
  const k = ttKey(board, player);
  const e = tt.get(k);
  if (e) {
    const newV = (e.value * e.visits + value) / (e.visits + 1);
    e.visits += 1;
    e.value = newV;
    tt.delete(k); tt.set(k, e);
  } else {
    ttSet(board, player, { visits: 1, value, priors: null });
  }
}

function bootstrapChildrenFromTT(node: MCTSNodeNN, baseBoard: (Player | null)[][]): void {
  const V0 = Math.max(0, Number(process.env.TT_BOOTSTRAP_VISITS || 3));
  if (V0 <= 0) return;
  for (const child of Object.values(node.children)) {
    if (!child || child.visits > 0) continue;
    const [r, c] = child.move!;
    const nb = baseBoard.map((row) => [...row]);
    // Apply parent's move to reach child state
    nb[r][c] = node.player;
    const ent = ttGet(nb, child.player);
    if (ent && ent.visits > 0) {
      const vInit = Math.min(ent.visits, V0);
      child.visits += vInit;
      child.valueSum += ent.value * vInit;
    }
  }
}

// --- Lightweight Prediction Cache (LRU) ---
type PredCacheEntry = { policy: Float32Array; value: number; size: number };
const PRED_CACHE_CAP = 5000;
const predCache = new Map<string, PredCacheEntry>();

function boardStrKey(board: (Player | null)[][]): string {
  return board.map((row) => row.map((c) => (c === 'black' ? 'b' : c === 'white' ? 'w' : '-')).join('')).join('/');
}
function canonicalKey(board: (Player | null)[][], player: Player): string {
  const n = getBoardSize(board);
  const trans: TransformId[] = [0,1,2,3,4,5,6,7];
  let best = '';
  let first = true;
  for (const t of trans) {
    const b = t === 0 ? board : transformBoard(board, t);
    const h = boardStrKey(b);
    if (first || h < best) { best = h; first = false; }
  }
  return `${player}|${best}`;
}

function predCacheGet(board: (Player | null)[][], player: Player): PredCacheEntry | null {
  const k = canonicalKey(board, player);
  const v = predCache.get(k) || null;
  if (!v) return null;
  // refresh LRU
  predCache.delete(k);
  predCache.set(k, v);
  return v;
}

function predCacheSet(board: (Player | null)[][], player: Player, entry: PredCacheEntry): void {
  const k = canonicalKey(board, player);
  if (predCache.has(k)) predCache.delete(k);
  predCache.set(k, entry);
  if (predCache.size > PRED_CACHE_CAP) {
    // delete oldest
    const first = predCache.keys().next().value as string | undefined;
    if (first) predCache.delete(first);
  }
}

export function findThreats(
  board: (Player | null)[][],
  player: Player,
  threatType: 'open-four' | 'four'
): Move[] {
  const size = getBoardSize(board);
  const threats: Move[] = [];
  const directions: [number, number][] = [
    [0, 1],
    [1, 0],
    [1, 1],
    [1, -1],
  ];

  for (let rStart = 0; rStart < size; rStart++) {
    for (let cStart = 0; cStart < size; cStart++) {
      for (const [dr, dc] of directions) {
        const pos: Move[] = [];
        let playerCount = 0;
        let emptyCell: Move | null = null;

        for (let i = 0; i < 5; i++) {
          const r = rStart + i * dr;
          const c = cStart + i * dc;
          if (r < 0 || r >= size || c < 0 || c >= size) {
            playerCount = -1;
            break;
          }
          pos.push([r, c]);
          const cell = board[r][c];
          if (cell === player) playerCount++;
          else if (cell === null) {
            if (emptyCell !== null) {
              playerCount = -1;
              break;
            }
            emptyCell = [r, c];
          } else {
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
            const openEnds =
              board[beforeR]?.[beforeC] === null &&
              board[afterR]?.[afterC] === null;
            if (openEnds) threats.push(emptyCell);
          }
        }
      }
    }
  }
  return threats;
}

// Detect moves that create an open-four immediately if played now (open-three generator)
function listOpenThreeMakers(board: (Player | null)[][], player: Player, candidates: Move[]): Move[] {
  const out: Move[] = [];
  for (const [r, c] of candidates) {
    if (board[r][c] !== null) continue;
    board[r][c] = player;
    const of = findThreats(board, player, 'open-four');
    board[r][c] = null;
    if (of.length > 0) out.push([r, c]);
  }
  return out;
}

export async function findBestMoveNN(
  model: TFT.LayersModel,
  board: (Player | null)[][],
  player: Player,
  timeLimitMs: number
): Promise<{ bestMove: Move; policy: PolicyData[] }> {
  // Tunables (override via ENV)
  const VCT_MAX_DEPTH = getNumber('VCT_MAX_DEPTH', 4);
  const SYM_AUG_ROOT = getNumber('SYM_AUG_ROOT', 8);
  const MCTS_BATCH_SIZE = getNumber('MCTS_BATCH_SIZE', 8);
  const K_ROOT_CAP = getNumber('K_ROOT_CAP', 256);
  const K_CHILD_BASE = getNumber('K_CHILD_BASE', 24);
  const K_CHILD_STEP = getNumber('K_CHILD_STEP', 12);
  const K_CHILD_MAX = getNumber('K_CHILD_MAX', 128);
  const DIRICHLET_ALPHA = getNumber('DIRICHLET_ALPHA', 0.12);
  const DIRICHLET_EPS = getNumber('DIRICHLET_EPS', 0.25);
  const solverSoftDeadline = Date.now() + Math.min(1500, Math.floor(timeLimitMs * 0.3));
  let solverNodes = 0;
  const MAX_SOLVER_NODES = 20000;
  // Adaptive speed profile
  const SPEED = (process.env.SPEED_PRESET || 'balanced').toLowerCase();
  const isFast = SPEED === 'fast' || timeLimitMs <= 900;

  // Try limited-depth forcing line search (VCF-light)
  // Try deeper VCT (threat-space) search first (depth adapt)
  const VCT_ADAPT = isFast ? Math.min(2, VCT_MAX_DEPTH) : (timeLimitMs <= 1200 ? Math.min(Math.max(2, VCT_MAX_DEPTH - 1), VCT_MAX_DEPTH) : VCT_MAX_DEPTH);
  const vct = findVCTWinFirstMove(board, player, VCT_ADAPT, solverSoftDeadline, () => {
    solverNodes++;
    return solverNodes >= MAX_SOLVER_NODES;
  });
  if (vct) return { bestMove: vct, policy: [{ move: vct, visits: 1 }] };

  const forced = findForcedWinFirstMove(board, player, 3, solverSoftDeadline, () => {
    solverNodes++;
    return solverNodes >= MAX_SOLVER_NODES;
  });
  if (forced) return { bestMove: forced, policy: [{ move: forced, visits: 1 }] };

  // Try defensive blocker if opponent has a short forced win (depth 2-3)
  const defense = findDefenseMoveAgainstOpponentForced(board, player, 3, solverSoftDeadline, () => {
    solverNodes++;
    return solverNodes >= MAX_SOLVER_NODES;
  });
  if (defense) return { bestMove: defense, policy: [{ move: defense, visits: 1 }] };
  // --- Helper: count stones and list legal moves ---
  const size = getBoardSize(board);
  // Count stones quickly for early-game heuristics
  let stones = 0;
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) if (board[r][c] !== null) stones++;
  // Use adjacency-limited candidates to reduce branching; widen radius in very early game
  const rootRadius = stones <= Math.max(6, Math.floor(size / 3)) ? 2 : 1;
  const legalMoves: Move[] = getPossibleMoves(board, rootRadius);

  // Precompute forbidden (3-3 / 4-4) set for Black to avoid in selection
  const forbiddenSet = new Set<string>();
  if (player === 'black') {
    for (const [rr, cc] of legalMoves) {
      if (isForbiddenMove(board, player, [rr, cc])) forbiddenSet.add(`${rr},${cc}`);
    }
  }

  // 1) Immediate tactical checks: win or block-win
  // Try winning move
  for (const [r, c] of legalMoves) {
    if (board[r][c] !== null) continue;
    board[r][c] = player;
    const wins = checkWin(board, player, [r, c]);
    board[r][c] = null;
    if (wins) {
      return { bestMove: [r, c], policy: [{ move: [r, c], visits: 1 }] };
    }
  }
  // Try blocking opponent's win
  const opp = getOpponent(player);
  for (const [r, c] of legalMoves) {
    if (board[r][c] !== null) continue;
    board[r][c] = opp;
    const oppWins = checkWin(board, opp, [r, c]);
    board[r][c] = null;
    if (oppWins) {
      // Avoid forbidden blocks for Black (illegal under Renju-like rules)
      if (player === 'black' && isForbiddenMove(board, player, [r, c])) {
        continue;
      }
      return { bestMove: [r, c], policy: [{ move: [r, c], visits: 1 }] };
    }
  }

  // 2) Root policy/value with symmetry averaging + Dirichlet noise for early-game exploration
  const startTime = Date.now();
  const root = new MCTSNodeNN(player);

  // Initial prediction for the root node
  let cached = predCacheGet(board, player);
  let rootPolicy: Float32Array;
  if (cached && cached.size === size) {
    rootPolicy = cached.policy;
  } else {
  // Adaptive root symmetry count for speed
  const SYM_ADAPT = (process.env.SPEED_PRESET || 'balanced').toLowerCase() === 'fast' || timeLimitMs <= 900
    ? 1
    : (timeLimitMs <= 1200 ? Math.min(4, SYM_AUG_ROOT) : SYM_AUG_ROOT);
  const { policy, value } = await predictRootWithSymmetry(model, board, player, SYM_ADAPT);
  rootPolicy = policy;
  predCacheSet(board, player, { policy, value, size });
  }

  // Mask illegal moves and renormalize
  const masked = new Float32Array(size * size);
  for (let i = 0; i < masked.length; i++) masked[i] = 0;
  for (const [r, c] of legalMoves) masked[r * size + c] = rootPolicy[r * size + c];
  // Avoid forbidden moves for Black by down-weighting or zeroing at root
  if (player === 'black' && forbiddenSet.size > 0) {
    const FORBIDDEN_PENALTY = Math.max(0, Math.min(1, Number(process.env.FORBIDDEN_PENALTY || 0)));
    if (FORBIDDEN_PENALTY < 1) {
      for (const [r, c] of legalMoves) {
        if (forbiddenSet.has(`${r},${c}`)) masked[r * size + c] *= FORBIDDEN_PENALTY;
      }
    }
  }
  // Blend TT priors if available
  const ttRoot = ttGet(board, player);
  const TT_PRIOR_MIX = Math.max(0, Math.min(1, getNumber('TT_PRIOR_MIX', 0.2)));
  if (ttRoot && ttRoot.priors && TT_PRIOR_MIX > 0) {
    const blend = new Float32Array(size * size);
    for (const [r, c] of legalMoves) blend[r * size + c] = Math.max(0, ttRoot.priors[r * size + c] || 0);
    // renorm both and mix
    let sA = 0, sB = 0; for (let i = 0; i < masked.length; i++) { sA += Math.max(0, masked[i]); sB += Math.max(0, blend[i]); }
    if (sA > 0 && sB > 0) {
      for (let i = 0; i < masked.length; i++) masked[i] = Math.max(0, masked[i]) / sA;
      for (let i = 0; i < blend.length; i++) blend[i] = Math.max(0, blend[i]) / sB;
      for (let i = 0; i < masked.length; i++) masked[i] = (1 - TT_PRIOR_MIX) * masked[i] + TT_PRIOR_MIX * blend[i];
    }
  }
  // Add small epsilon to avoid all-zero
  let sum = 0;
  for (let i = 0; i < masked.length; i++) sum += Math.max(0, masked[i]);
  if (sum <= 0) {
    const u = 1 / legalMoves.length;
    for (const [r, c] of legalMoves) masked[r * size + c] = u;
  } else {
    for (let i = 0; i < masked.length; i++) masked[i] = Math.max(0, masked[i]) / sum;
  }

  // Dirichlet noise at root (AlphaGo-like): only early phase
  const addNoise = stones <= Math.max(8, Math.floor(size / 2));
  if (addNoise) {
    const alpha = DIRICHLET_ALPHA; // typical 0.03~0.3 depending on branching
    const epsilon = DIRICHLET_EPS;
    const noise = sampleDirichletForLegal(legalMoves.length, alpha);
    // mix into masked prior
    let k = 0;
    for (const [r, c] of legalMoves) {
      const idx = r * size + c;
      masked[idx] = (1 - epsilon) * masked[idx] + epsilon * noise[k++];
    }
  }

  // Tactical prior boost: favor creating/blocking open-fours at root
  // Keeps NN-guided search but nudges priors to focus threats
  const boostCreate = findThreats(board, player, 'open-four');
  const boostBlock = findThreats(board, opp, 'open-four');
  const boostOpen3Create = listOpenThreeMakers(board, player, legalMoves);
  const boostOpen3Block = listOpenThreeMakers(board, opp, legalMoves);
  const boostFourCreate = findThreats(board, player, 'four');
  const boostFourBlock = findThreats(board, opp, 'four');
  const boostConn3Create = listConnectedThreeMakers(board, player, legalMoves);
  const boostConn3Block = listConnectedThreeMakers(board, opp, legalMoves);
  const boostLink = listLongLinkMakers(board, player, legalMoves);
  if (boostCreate.length + boostBlock.length > 0) {
    const BOOST_CREATE = getNumber('BOOST_CREATE', 1.5);
    const BOOST_BLOCK = getNumber('BOOST_BLOCK', 1.3);
    for (const [r, c] of boostCreate) {
      const idx = r * size + c;
      masked[idx] *= BOOST_CREATE;
    }
    for (const [r, c] of boostBlock) {
      const idx = r * size + c;
      masked[idx] *= BOOST_BLOCK;
    }
  }
  if (boostOpen3Create.length + boostOpen3Block.length > 0) {
    const BOOST_OPEN3_CREATE = getNumber('BOOST_OPEN3_ROOT', 1.08);
    const BOOST_OPEN3_BLOCK = getNumber('BOOST_OPEN3_ROOT_BLOCK', 1.05);
    for (const [r, c] of boostOpen3Create) masked[r * size + c] *= BOOST_OPEN3_CREATE;
    for (const [r, c] of boostOpen3Block) masked[r * size + c] *= BOOST_OPEN3_BLOCK;
  }
  if (boostFourCreate.length + boostFourBlock.length > 0) {
    const BOOST_FOUR_CREATE = getNumber('BOOST_FOUR_ROOT', 1.15);
    const BOOST_FOUR_BLOCK = getNumber('BOOST_FOUR_ROOT_BLOCK', 1.1);
    for (const [r, c] of boostFourCreate) masked[r * size + c] *= BOOST_FOUR_CREATE;
    for (const [r, c] of boostFourBlock) masked[r * size + c] *= BOOST_FOUR_BLOCK;
  }
  if (boostConn3Create.length + boostConn3Block.length > 0) {
    const BOOST_CONN3_CREATE = getNumber('BOOST_CONN3_ROOT', 1.05);
    const BOOST_CONN3_BLOCK = getNumber('BOOST_CONN3_ROOT_BLOCK', 1.03);
    for (const [r, c] of boostConn3Create) masked[r * size + c] *= BOOST_CONN3_CREATE;
    for (const [r, c] of boostConn3Block) masked[r * size + c] *= BOOST_CONN3_BLOCK;
  }
  if (boostLink.length > 0) {
    const BOOST_LINK_ROOT = getNumber('BOOST_LINK_ROOT', 1.03);
    for (const [r, c] of boostLink) masked[r * size + c] *= BOOST_LINK_ROOT;
  }
  // Renormalize after boosting
  let s2 = 0; for (let i = 0; i < masked.length; i++) s2 += Math.max(0, masked[i]);
  if (s2 <= 0) {
    const u = 1 / legalMoves.length;
    for (const [r, c] of legalMoves) masked[r * size + c] = u;
  } else {
    for (let i = 0; i < masked.length; i++) masked[i] = Math.max(0, masked[i]) / s2;
  }

  // Top-K pruning to stabilize search
  const K_ROOT = Math.min(K_ROOT_CAP, legalMoves.length);
  root.expand(board, masked, K_ROOT);
  bootstrapChildrenFromTT(root, board);

  let simulationCount = 0;
  // Adaptive batch size
  const BATCH_MAX = ((process.env.SPEED_PRESET || 'balanced').toLowerCase() === 'fast' || timeLimitMs <= 900)
    ? Math.min(4, MCTS_BATCH_SIZE)
    : MCTS_BATCH_SIZE;
  interface PendingItem { node: MCTSNodeNN; board: (Player|null)[][]; }
  while (Date.now() - startTime < timeLimitMs) {
    const pending: PendingItem[] = [];
    for (let k = 0; k < BATCH_MAX && Date.now() - startTime < timeLimitMs; k++) {
      const currentBoard = board.map((row) => [...row]);
      let node: MCTSNodeNN | null = root;
      while (Object.keys(node.children).length > 0) {
        node = node.selectChild();
        if (!node) break;
        currentBoard[node.move![0]][node.move![1]] = node.parent!.player;
      }
      if (!node) continue;
      const lastMove = node.move;
      const lastPlayer = node.parent?.player;
      if (lastMove && lastPlayer && checkWin(currentBoard, lastPlayer, lastMove)) {
        node.backpropagate(-1);
        simulationCount++;
        continue;
      }
      if (!hasEmpty(currentBoard)) {
        node.backpropagate(0);
        simulationCount++;
        continue;
      }
      const childSize = getBoardSize(currentBoard);
      // Transposition table lookup before NN call
      const ttEnt = ttGet(currentBoard, node.player);
      if (ttEnt && ttEnt.priors) {
        const childLegal: Move[] = getPossibleMoves(currentBoard);
        // Base priors from TT
        const priTT = new Float32Array(childSize * childSize);
        for (const [rr, cc] of childLegal) priTT[rr * childSize + cc] = Math.max(0, ttEnt.priors[rr * childSize + cc] || 0);
        // Optional hybrid with cached NN priors
        let maskedChild: Float32Array = priTT;
        const CHILD_TT_MIX = Math.max(0, Math.min(1, getNumber('CHILD_TT_PRIOR_MIX', 0.35)));
        const cachedPred = predCacheGet(currentBoard, node.player);
        if (CHILD_TT_MIX > 0 && cachedPred) {
          const priNN = new Float32Array(childSize * childSize);
          for (const [rr, cc] of childLegal) priNN[rr * childSize + cc] = Math.max(0, cachedPred.policy[rr * childSize + cc] || 0);
          maskedChild = mixPolicies(priNN, priTT, CHILD_TT_MIX) as any as Float32Array;
        } else {
          // Normalize TT-only
          let sTT = 0; for (let i = 0; i < maskedChild.length; i++) sTT += maskedChild[i];
          if (sTT <= 0) { const u = 1 / childLegal.length; for (const [rr, cc] of childLegal) maskedChild[rr * childSize + cc] = u; }
          else { for (let i = 0; i < maskedChild.length; i++) maskedChild[i] = maskedChild[i] / sTT; }
        }
        // Child-level tactical boost
        const BOOST_CREATE_CHILD = getNumber('BOOST_CREATE_CHILD', 1.3);
        const BOOST_BLOCK_CHILD = getNumber('BOOST_BLOCK_CHILD', 1.2);
        const BOOST_OPEN3_CHILD = getNumber('BOOST_OPEN3_CHILD', 1.1);
        const BOOST_OPEN3_BLOCK_CHILD = getNumber('BOOST_OPEN3_BLOCK_CHILD', 1.05);
        const BOOST_CONN3_CHILD = getNumber('BOOST_CONN3_CHILD', 1.05);
        const BOOST_CONN3_BLOCK_CHILD = getNumber('BOOST_CONN3_BLOCK_CHILD', 1.02);
        const BOOST_LINK_CHILD = getNumber('BOOST_LINK_CHILD', 1.02);
        const oppLocal = getOpponent(node.player);
        const bc = findThreats(currentBoard, node.player, 'open-four');
        const bb = findThreats(currentBoard, oppLocal, 'open-four');
        for (const [rr, cc] of bc) maskedChild[rr * childSize + cc] *= BOOST_CREATE_CHILD;
        for (const [rr, cc] of bb) maskedChild[rr * childSize + cc] *= BOOST_BLOCK_CHILD;
        const open3c = listOpenThreeMakers(currentBoard, node.player, childLegal);
        const open3b = listOpenThreeMakers(currentBoard, oppLocal, childLegal);
        for (const [rr, cc] of open3c) maskedChild[rr * childSize + cc] *= BOOST_OPEN3_CHILD;
        for (const [rr, cc] of open3b) maskedChild[rr * childSize + cc] *= BOOST_OPEN3_BLOCK_CHILD;
        const conn3 = listConnectedThreeMakers(currentBoard, node.player, childLegal);
        const conn3Opp = listConnectedThreeMakers(currentBoard, oppLocal, childLegal);
        for (const [rr, cc] of conn3) maskedChild[rr * childSize + cc] *= BOOST_CONN3_CHILD;
        for (const [rr, cc] of conn3Opp) maskedChild[rr * childSize + cc] *= BOOST_CONN3_BLOCK_CHILD;
        const links = listLongLinkMakers(currentBoard, node.player, childLegal);
        for (const [rr, cc] of links) maskedChild[rr * childSize + cc] *= BOOST_LINK_CHILD;
        // Immediate win/block boost
        const BOOST_INSTANT_WIN_CHILD = Number(process.env.BOOST_INSTANT_WIN_CHILD || 5.0);
        const BOOST_BLOCK_IMM_CHILD = Number(process.env.BOOST_BLOCK_IMMEDIATE_CHILD || 2.0);
        const winsNowTT = winningMovesFor(currentBoard, node.player);
        const oppWinsTT = winningMovesFor(currentBoard, oppLocal);
        for (const [rr, cc] of winsNowTT) maskedChild[rr * childSize + cc] *= BOOST_INSTANT_WIN_CHILD;
        for (const [rr, cc] of oppWinsTT) maskedChild[rr * childSize + cc] *= BOOST_BLOCK_IMM_CHILD;
        // Forbidden avoidance for Black in child priors (TT path)
        if (node.player === 'black') {
          const FORBIDDEN_PENALTY = Math.max(0, Math.min(1, Number(process.env.FORBIDDEN_PENALTY || 0)));
          if (FORBIDDEN_PENALTY < 1) {
            for (const [rr, cc] of childLegal) {
              if (isForbiddenMove(currentBoard, node.player, [rr, cc])) maskedChild[rr * childSize + cc] *= FORBIDDEN_PENALTY;
            }
          }
        }
        let sTT2 = 0; for (let i = 0; i < maskedChild.length; i++) sTT2 += Math.max(0, maskedChild[i]);
        if (sTT2 <= 0) { const u = 1 / childLegal.length; for (const [rr, cc] of childLegal) maskedChild[rr * childSize + cc] = u; }
        else { for (let i = 0; i < maskedChild.length; i++) maskedChild[i] = Math.max(0, maskedChild[i]) / sTT2; }
        // Adaptive child widening parameters
        const _FAST = ((process.env.SPEED_PRESET || 'balanced').toLowerCase() === 'fast' || timeLimitMs <= 900);
        const KC_BASE = _FAST ? Math.max(12, Math.floor(getNumber('K_CHILD_BASE', 24) * 0.6)) : getNumber('K_CHILD_BASE', 24);
        const KC_STEP = _FAST ? Math.max(8, Math.floor(getNumber('K_CHILD_STEP', 12) * 0.66)) : getNumber('K_CHILD_STEP', 12);
        const KC_MAX = _FAST ? Math.max(64, Math.floor(getNumber('K_CHILD_MAX', 128) * 0.7)) : getNumber('K_CHILD_MAX', 128);
        const widenTT = Math.min(KC_MAX, childLegal.length, KC_BASE + KC_STEP * Math.floor(Math.sqrt(node.visits + 1)));
        node.expand(currentBoard, maskedChild, widenTT);
        bootstrapChildrenFromTT(node, currentBoard);
        node.backpropagate(ttEnt.value);
        ttAccumulate(currentBoard, node.player, ttEnt.value);
        simulationCount++;
        continue;
      }
      const cachedChild = predCacheGet(currentBoard, node.player);
      if (cachedChild && cachedChild.size === childSize) {
        const policy = cachedChild.policy;
        const value = cachedChild.value;
        const childLegal: Move[] = getPossibleMoves(currentBoard);
        // Base from NN
        let maskedChild: Float32Array = new Float32Array(childSize * childSize);
        for (const [rr, cc] of childLegal) maskedChild[rr * childSize + cc] = Math.max(0, policy[rr * childSize + cc]);
        // Optional hybrid with TT priors
        const CHILD_TT_MIX2 = Math.max(0, Math.min(1, getNumber('CHILD_TT_PRIOR_MIX', 0.35)));
        const ttEnt2 = ttGet(currentBoard, node.player);
        if (ttEnt2 && ttEnt2.priors && CHILD_TT_MIX2 > 0) {
          const priTT2 = new Float32Array(childSize * childSize);
          for (const [rr, cc] of childLegal) priTT2[rr * childSize + cc] = Math.max(0, ttEnt2.priors[rr * childSize + cc] || 0);
          maskedChild = mixPolicies(maskedChild, priTT2, CHILD_TT_MIX2) as any as Float32Array;
        }
        // Normalize
        let s = 0; for (let i = 0; i < maskedChild.length; i++) s += maskedChild[i];
        if (s <= 0) { const u = 1 / childLegal.length; for (const [rr, cc] of childLegal) maskedChild[rr * childSize + cc] = u; }
        else { for (let i = 0; i < maskedChild.length; i++) maskedChild[i] = maskedChild[i] / s; }
        // Child-level tactical boost
        const BOOST_CREATE_CHILD = getNumber('BOOST_CREATE_CHILD', 1.3);
        const BOOST_BLOCK_CHILD = getNumber('BOOST_BLOCK_CHILD', 1.2);
        const BOOST_OPEN3_CHILD = getNumber('BOOST_OPEN3_CHILD', 1.1);
        const BOOST_OPEN3_BLOCK_CHILD = getNumber('BOOST_OPEN3_BLOCK_CHILD', 1.05);
        const BOOST_CONN3_CHILD = getNumber('BOOST_CONN3_CHILD', 1.05);
        const BOOST_CONN3_BLOCK_CHILD = getNumber('BOOST_CONN3_BLOCK_CHILD', 1.02);
        const BOOST_LINK_CHILD = getNumber('BOOST_LINK_CHILD', 1.02);
        const oppLocal = getOpponent(node.player);
        const bc = findThreats(currentBoard, node.player, 'open-four');
        const bb = findThreats(currentBoard, oppLocal, 'open-four');
        for (const [rr, cc] of bc) maskedChild[rr * childSize + cc] *= BOOST_CREATE_CHILD;
        for (const [rr, cc] of bb) maskedChild[rr * childSize + cc] *= BOOST_BLOCK_CHILD;
        const open3pc = listOpenThreeMakers(currentBoard, node.player, childLegal);
        const open3pb = listOpenThreeMakers(currentBoard, oppLocal, childLegal);
        for (const [rr, cc] of open3pc) maskedChild[rr * childSize + cc] *= BOOST_OPEN3_CHILD;
        for (const [rr, cc] of open3pb) maskedChild[rr * childSize + cc] *= BOOST_OPEN3_BLOCK_CHILD;
        const conn3b = listConnectedThreeMakers(currentBoard, node.player, childLegal);
        const conn3bOpp = listConnectedThreeMakers(currentBoard, oppLocal, childLegal);
        for (const [rr, cc] of conn3b) maskedChild[rr * childSize + cc] *= BOOST_CONN3_CHILD;
        for (const [rr, cc] of conn3bOpp) maskedChild[rr * childSize + cc] *= BOOST_CONN3_BLOCK_CHILD;
        const links2 = listLongLinkMakers(currentBoard, node.player, childLegal);
        for (const [rr, cc] of links2) maskedChild[rr * childSize + cc] *= BOOST_LINK_CHILD;
        const BOOST_INSTANT_WIN_CHILD = Number(process.env.BOOST_INSTANT_WIN_CHILD || 5.0);
        const BOOST_BLOCK_IMM_CHILD = Number(process.env.BOOST_BLOCK_IMMEDIATE_CHILD || 2.0);
        const winsNowPC = winningMovesFor(currentBoard, node.player);
        const oppWinsPC = winningMovesFor(currentBoard, oppLocal);
        for (const [rr, cc] of winsNowPC) maskedChild[rr * childSize + cc] *= BOOST_INSTANT_WIN_CHILD;
        for (const [rr, cc] of oppWinsPC) maskedChild[rr * childSize + cc] *= BOOST_BLOCK_IMM_CHILD;
        // Forbidden avoidance for Black in child priors (cache path)
        if (node.player === 'black') {
          const FORBIDDEN_PENALTY = Math.max(0, Math.min(1, Number(process.env.FORBIDDEN_PENALTY || 0)));
          if (FORBIDDEN_PENALTY < 1) {
            for (const [rr, cc] of childLegal) {
              if (isForbiddenMove(currentBoard, node.player, [rr, cc])) maskedChild[rr * childSize + cc] *= FORBIDDEN_PENALTY;
            }
          }
        }
        let s2b = 0; for (let i = 0; i < maskedChild.length; i++) s2b += Math.max(0, maskedChild[i]);
        if (s2b <= 0) { const u = 1 / childLegal.length; for (const [rr, cc] of childLegal) maskedChild[rr * childSize + cc] = u; }
        else { for (let i = 0; i < maskedChild.length; i++) maskedChild[i] = Math.max(0, maskedChild[i]) / s2b; }
      const widen = Math.min(K_CHILD_MAX, childLegal.length, K_CHILD_BASE + K_CHILD_STEP * Math.floor(Math.sqrt(node.visits + 1)));
      node.expand(currentBoard, maskedChild, widen);
      bootstrapChildrenFromTT(node, currentBoard);
      node.backpropagate(value);
      ttAccumulate(currentBoard, node.player, value);
       simulationCount++;
     } else {
        pending.push({ node, board: currentBoard });
      }
    }
    if (pending.length > 0) {
      const inputs: TFT.Tensor4D[] = [];
      for (const item of pending) inputs.push(boardToInputTensor(item.board, item.node.player));
      const batched = tf.concat(inputs, 0) as TFT.Tensor4D;
      inputs.forEach(t => t.dispose());
      const [policyTensor, valueTensor] = model.predict(batched) as TFT.Tensor[];
      const flat = (await policyTensor.data()) as Float32Array;
      const vals = (await valueTensor.data()) as Float32Array;
      tf.dispose([batched, policyTensor, valueTensor]);
      let off = 0;
      for (let i = 0; i < pending.length; i++) {
        const { node, board: cb } = pending[i];
        const childSize = getBoardSize(cb);
        const len = childSize * childSize;
        const pol = flat.subarray(off, off + len);
        const value = vals[i] as number;
        off += len;
        predCacheSet(cb, node.player, { policy: pol.slice() as any as Float32Array, value, size: childSize });
        ttSet(cb, node.player, { visits: 1, value, priors: pol.slice() as any as Float32Array });
        const childLegal: Move[] = getPossibleMoves(cb);
        const maskedChild = new Float32Array(childSize * childSize);
        for (const [rr, cc] of childLegal) maskedChild[rr * childSize + cc] = Math.max(0, pol[rr * childSize + cc]);
        // Normalize
        let s = 0; for (let j = 0; j < maskedChild.length; j++) s += maskedChild[j];
        if (s <= 0) { const u = 1 / childLegal.length; for (const [rr, cc] of childLegal) maskedChild[rr * childSize + cc] = u; }
        else { for (let j = 0; j < maskedChild.length; j++) maskedChild[j] = maskedChild[j] / s; }
        // Child-level tactical boost
        const BOOST_CREATE_CHILD = Number(process.env.BOOST_CREATE_CHILD || 1.3);
        const BOOST_BLOCK_CHILD = Number(process.env.BOOST_BLOCK_CHILD || 1.2);
        const BOOST_OPEN3_CHILD = Number(process.env.BOOST_OPEN3_CHILD || 1.1);
        const BOOST_OPEN3_BLOCK_CHILD = Number(process.env.BOOST_OPEN3_BLOCK_CHILD || 1.05);
        const oppLocal = getOpponent(node.player);
        const bc = findThreats(cb, node.player, 'open-four');
        const bb = findThreats(cb, oppLocal, 'open-four');
        for (const [rr, cc] of bc) maskedChild[rr * childSize + cc] *= BOOST_CREATE_CHILD;
        for (const [rr, cc] of bb) maskedChild[rr * childSize + cc] *= BOOST_BLOCK_CHILD;
        const open3b2 = listOpenThreeMakers(cb, node.player, childLegal);
        const open3b2opp = listOpenThreeMakers(cb, oppLocal, childLegal);
        for (const [rr, cc] of open3b2) maskedChild[rr * childSize + cc] *= BOOST_OPEN3_CHILD;
        for (const [rr, cc] of open3b2opp) maskedChild[rr * childSize + cc] *= BOOST_OPEN3_BLOCK_CHILD;
        const BOOST_INSTANT_WIN_CHILD = Number(process.env.BOOST_INSTANT_WIN_CHILD || 5.0);
        const BOOST_BLOCK_IMM_CHILD = Number(process.env.BOOST_BLOCK_IMMEDIATE_CHILD || 2.0);
        const winsNowB = winningMovesFor(cb, node.player);
        const oppWinsB = winningMovesFor(cb, oppLocal);
        for (const [rr, cc] of winsNowB) maskedChild[rr * childSize + cc] *= BOOST_INSTANT_WIN_CHILD;
        for (const [rr, cc] of oppWinsB) maskedChild[rr * childSize + cc] *= BOOST_BLOCK_IMM_CHILD;
        // Forbidden avoidance for Black in child priors (batched NN path)
        if (node.player === 'black') {
          const FORBIDDEN_PENALTY = Math.max(0, Math.min(1, Number(process.env.FORBIDDEN_PENALTY || 0)));
          if (FORBIDDEN_PENALTY < 1) {
            for (const [rr, cc] of childLegal) {
              if (isForbiddenMove(cb, node.player, [rr, cc])) maskedChild[rr * childSize + cc] *= FORBIDDEN_PENALTY;
            }
          }
        }
        let s2c = 0; for (let j = 0; j < maskedChild.length; j++) s2c += Math.max(0, maskedChild[j]);
        if (s2c <= 0) { const u = 1 / childLegal.length; for (const [rr, cc] of childLegal) maskedChild[rr * childSize + cc] = u; }
        else { for (let j = 0; j < maskedChild.length; j++) maskedChild[j] = Math.max(0, maskedChild[j]) / s2c; }
        // Adaptive child widening parameters
        const _FAST2 = ((process.env.SPEED_PRESET || 'balanced').toLowerCase() === 'fast' || timeLimitMs <= 900);
        const KC_BASE2 = _FAST2 ? Math.max(12, Math.floor(getNumber('K_CHILD_BASE', 24) * 0.6)) : getNumber('K_CHILD_BASE', 24);
        const KC_STEP2 = _FAST2 ? Math.max(8, Math.floor(getNumber('K_CHILD_STEP', 12) * 0.66)) : getNumber('K_CHILD_STEP', 12);
        const KC_MAX2 = _FAST2 ? Math.max(64, Math.floor(getNumber('K_CHILD_MAX', 128) * 0.7)) : getNumber('K_CHILD_MAX', 128);
        const widen = Math.min(KC_MAX2, childLegal.length, KC_BASE2 + KC_STEP2 * Math.floor(Math.sqrt(node.visits + 1)));
        node.expand(cb, maskedChild, widen);
        bootstrapChildrenFromTT(node, cb);
        node.backpropagate(value);
        ttAccumulate(cb, node.player, value);
        simulationCount++;
      }
    }
    // Early-stop: dominant child heuristic
    const FAST = ((process.env.SPEED_PRESET || 'balanced').toLowerCase() === 'fast' || timeLimitMs <= 900);
    const MINV = Number(process.env.EARLY_STOP_MIN_VISITS || (FAST ? 120 : 220));
    const RATIO = Number(process.env.EARLY_STOP_RATIO || (FAST ? 1.8 : 2.2));
    if (Object.keys(root.children).length > 1) {
      let best = 0, second = 0, total = 0;
      for (const ch of Object.values(root.children)) {
        total += ch.visits;
        if (ch.visits >= best) { second = best; best = ch.visits; }
        else if (ch.visits > second) { second = ch.visits; }
      }
      if (total >= MINV && best >= RATIO * Math.max(1, second)) break;
    }
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
    policy: Object.values(root.children).map((child) => ({
      move: child.move!,
      visits: child.visits,
    })),
  };
}

// --- Threat/VCF-light search ---
function listLegalMoves(board: (Player | null)[][]): Move[] {
  // Prefer adjacency-limited candidates for threat searches as well
  return getPossibleMoves(board);
}

function winningMovesFor(board: (Player | null)[][], player: Player): Move[] {
  const legal = listLegalMoves(board);
  const wins: Move[] = [];
  for (const [r, c] of legal) {
    board[r][c] = player;
    const ok = checkWin(board, player, [r, c]);
    board[r][c] = null;
    if (ok) wins.push([r, c]);
  }
  return wins;
}

function findForcedWinFirstMove(
  board: (Player | null)[][],
  player: Player,
  maxDepth: number,
  deadlineMs: number,
  shouldStop: () => boolean
): Move | null {
  const size = getBoardSize(board);
  const legal = listLegalMoves(board);

  // Candidate first moves: prioritize threat-creating moves
  const candidateSet = new Set<string>();
  const threats = findThreats(board, player, 'open-four');
  for (const t of threats) candidateSet.add(`${t[0]},${t[1]}`);
  // Also include makers that create open-four (open-three present now)
  for (const [rr, cc] of listOpenThreeMakers(board, player, legal)) candidateSet.add(`${rr},${cc}`);
  // Also include immediate winning moves
  const wins = winningMovesFor(board, player);
  for (const w of wins) candidateSet.add(`${w[0]},${w[1]}`);
  // If empty, fall back to all legal moves near center (light filter)
  const candidates: Move[] = [];
  if (candidateSet.size === 0) {
    const mid = Math.floor(size / 2);
    legal.sort((a, b) =>
      Math.abs(a[0] - mid) + Math.abs(a[1] - mid) - (Math.abs(b[0] - mid) + Math.abs(b[1] - mid))
    );
    for (let i = 0; i < Math.min(32, legal.length); i++) candidates.push(legal[i]);
  } else {
    for (const s of candidateSet) {
      const [r, c] = s.split(',').map(Number);
      candidates.push([r, c]);
    }
  }

  for (const [r, c] of candidates) {
    if (Date.now() > deadlineMs || shouldStop()) break;
    if (board[r][c] !== null) continue;
    if (player === 'black' && isForbiddenMove(board, player, [r, c])) continue;
    // Immediate win
    board[r][c] = player;
    if (checkWin(board, player, [r, c])) {
      board[r][c] = null;
      return [r, c];
    }

    const wm = winningMovesFor(board, player);
    if (wm.length >= 2) {
      board[r][c] = null;
      return [r, c]; // double threat → forced win
    }

    let ok = false;
    if (wm.length === 1 && maxDepth > 0) {
      // Opponent must block that single winning move
      const [br, bc] = wm[0];
      board[br][bc] = getOpponent(player);
      ok = forcedWinDFS(board, player, maxDepth - 1, deadlineMs, shouldStop);
      board[br][bc] = null;
    }
    board[r][c] = null;
    if (ok) return [r, c];
  }
  return null;
}

function forcedWinDFS(
  board: (Player | null)[][],
  player: Player,
  depth: number,
  deadlineMs: number,
  shouldStop: () => boolean
): boolean {
  if (depth <= 0 || Date.now() > deadlineMs || shouldStop()) return false;

  // Try to create a new double threat chain
  const size = getBoardSize(board);
  const legal = listLegalMoves(board);

  // Prioritize threat-creating moves
  const candidateSet = new Set<string>();
  const threats = findThreats(board, player, 'open-four');
  for (const t of threats) candidateSet.add(`${t[0]},${t[1]}`);
  for (const [rr, cc] of listOpenThreeMakers(board, player, legal)) candidateSet.add(`${rr},${cc}`);
  const wins = winningMovesFor(board, player);
  for (const w of wins) candidateSet.add(`${w[0]},${w[1]}`);
  const candidates: Move[] = [];
  if (candidateSet.size === 0) {
    const mid = Math.floor(size / 2);
    legal.sort((a, b) =>
      Math.abs(a[0] - mid) + Math.abs(a[1] - mid) - (Math.abs(b[0] - mid) + Math.abs(b[1] - mid))
    );
    for (let i = 0; i < Math.min(24, legal.length); i++) candidates.push(legal[i]);
  } else {
    for (const s of candidateSet) {
      const [r, c] = s.split(',').map(Number);
      candidates.push([r, c]);
    }
  }

  for (const [r, c] of candidates) {
    if (Date.now() > deadlineMs || shouldStop()) break;
    if (board[r][c] !== null) continue;
    if (player === 'black' && isForbiddenMove(board, player, [r, c])) continue;
    board[r][c] = player;
    if (checkWin(board, player, [r, c])) {
      board[r][c] = null;
      return true;
    }
    const wm = winningMovesFor(board, player);
    if (wm.length >= 2) {
      board[r][c] = null;
      return true; // double threat established
    }
    let ok = false;
    if (wm.length === 1 && depth > 0) {
      const [br, bc] = wm[0];
      board[br][bc] = getOpponent(player);
      ok = forcedWinDFS(board, player, depth - 1, deadlineMs, shouldStop);
      board[br][bc] = null;
    }
    board[r][c] = null;
    if (ok) return true;
  }
  return false;
}

// --- Threat Space Search (VCT) ---
function listWinningMoves(board: (Player | null)[][], player: Player): Move[] {
  return winningMovesFor(board, player);
}

function listThreatCandidates(board: (Player | null)[][], player: Player): Move[] {
  const size = getBoardSize(board);
  const set = new Set<string>();
  // Highest priority: winning moves now
  for (const [r, c] of listWinningMoves(board, player)) set.add(`${r},${c}`);
  // Open-fours (create five next)
  for (const [r, c] of findThreats(board, player, 'open-four')) set.add(`${r},${c}`);
  // Fours (may be blocked at one end)
  for (const [r, c] of findThreats(board, player, 'four')) set.add(`${r},${c}`);
  // Open-three creators (playing here yields an open-four)
  const legal = listLegalMoves(board);
  for (const [r, c] of listOpenThreeMakers(board, player, legal)) set.add(`${r},${c}`);
  // Return as array (some order)
  const res: Move[] = [];
  for (const s of set) {
    const [r, c] = s.split(',').map(Number);
    if (r >= 0 && r < size && c >= 0 && c < size && board[r][c] === null) res.push([r, c]);
  }
  return res;
}

function findVCTWinFirstMove(
  board: (Player | null)[][],
  player: Player,
  maxDepth: number,
  deadlineMs: number,
  shouldStop: () => boolean
): Move | null {
  const size = getBoardSize(board);
  const candidates = listThreatCandidates(board, player);
  for (const [r, c] of candidates) {
    if (Date.now() > deadlineMs || shouldStop()) break;
    if (board[r][c] !== null) continue;
    if (player === 'black' && isForbiddenMove(board, player, [r, c])) continue;
    board[r][c] = player;
    // Immediate win or double threat
    if (checkWin(board, player, [r, c])) { board[r][c] = null; return [r, c]; }
    const wins = listWinningMoves(board, player);
    if (wins.length >= 2) { board[r][c] = null; return [r, c]; }

    let ok = false;
    if (maxDepth > 0 && wins.length === 1) {
      // Opponent must block this single winning move
      const [br, bc] = wins[0];
      board[br][bc] = getOpponent(player);
      ok = vctDFS(board, player, maxDepth - 1, deadlineMs, shouldStop);
      board[br][bc] = null;
    }
    board[r][c] = null;
    if (ok) return [r, c];
  }
  return null;
}

function vctDFS(
  board: (Player | null)[][],
  player: Player,
  depth: number,
  deadlineMs: number,
  shouldStop: () => boolean
): boolean {
  if (depth <= 0 || Date.now() > deadlineMs || shouldStop()) return false;
  const size = getBoardSize(board);
  const candidates = listThreatCandidates(board, player);
  for (const [r, c] of candidates) {
    if (Date.now() > deadlineMs || shouldStop()) break;
    if (board[r][c] !== null) continue;
    if (player === 'black' && isForbiddenMove(board, player, [r, c])) continue;
    board[r][c] = player;
    if (checkWin(board, player, [r, c])) { board[r][c] = null; return true; }
    const wins = listWinningMoves(board, player);
    if (wins.length >= 2) { board[r][c] = null; return true; }
    if (wins.length === 1 && depth > 0) {
      const [br, bc] = wins[0];
      board[br][bc] = getOpponent(player);
      const cont = vctDFS(board, player, depth - 1, deadlineMs, shouldStop);
      board[br][bc] = null;
      if (cont) { board[r][c] = null; return true; }
    }
    board[r][c] = null;
  }
  return false;
}

function findDefenseMoveAgainstOpponentForced(
  board: (Player | null)[][],
  player: Player,
  maxDepth: number,
  deadlineMs: number,
  shouldStop: () => boolean
): Move | null {
  const opponent = getOpponent(player);
  const size = getBoardSize(board);
  const legal = listLegalMoves(board);

  // If opponent does not have a forced win now (assuming they move next), skip
  // We simulate opponent-to-move from current board to detect looming threats.
  if (!findForcedWinFirstMove(board, opponent, Math.max(2, maxDepth - 1), deadlineMs, shouldStop)) {
    return null;
  }

  // Candidate defensive moves: blocks of opponent threats and central proximity
  const candidateSet = new Set<string>();
  const oppThreats = findThreats(board, opponent, 'open-four');
  for (const t of oppThreats) candidateSet.add(`${t[0]},${t[1]}`);
  // Opponent open-three creators are also urgency to block
  for (const [rr, cc] of listOpenThreeMakers(board, opponent, legal)) candidateSet.add(`${rr},${cc}`);
  const oppWins = winningMovesFor(board, opponent);
  for (const w of oppWins) candidateSet.add(`${w[0]},${w[1]}`);
  const candidates: Move[] = [];
  if (candidateSet.size === 0) {
    const mid = Math.floor(size / 2);
    legal.sort((a, b) =>
      Math.abs(a[0] - mid) + Math.abs(a[1] - mid) - (Math.abs(b[0] - mid) + Math.abs(b[1] - mid))
    );
    for (let i = 0; i < Math.min(32, legal.length); i++) candidates.push(legal[i]);
  } else {
    for (const s of candidateSet) {
      const [r, c] = s.split(',').map(Number);
      candidates.push([r, c]);
    }
  }

  for (const [r, c] of candidates) {
    if (Date.now() > deadlineMs || shouldStop()) break;
    if (board[r][c] !== null) continue;
    if (player === 'black' && isForbiddenMove(board, player, [r, c])) continue;
    // Try playing the block
    board[r][c] = player;
    // After our move, opponent to move: do they still have a forced win?
    const stillForced = findForcedWinFirstMove(board, opponent, Math.max(2, maxDepth - 1), deadlineMs, shouldStop);
    board[r][c] = null;
    if (!stillForced) return [r, c];
  }
  return null;
}

// --- Sampling utilities ---
function sampleDirichletForLegal(n: number, alpha: number): number[] {
  // Dirichlet(alpha) by normalizing Gamma(alpha, 1)
  const arr = new Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const g = sampleGamma(alpha, 1);
    arr[i] = g;
    sum += g;
  }
  if (sum <= 0) {
    const u = 1 / n;
    return arr.map(() => u);
  }
  return arr.map((x) => x / sum);
}

// Marsaglia and Tsang method for k >= 1, and Johnk's transformation for k < 1
function sampleGamma(k: number, theta: number): number {
  if (k < 1) {
    // Use boost: Gamma(k) = Gamma(k+1)*U^(1/k)
    const u = Math.random();
    return sampleGamma(1 + k, theta) * Math.pow(u, 1 / k);
  }
  const d = k - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x: number, v: number;
    do {
      x = gaussian();
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return theta * d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return theta * d * v;
  }
}

// Box-Muller transform
function gaussian(): number {
  let u = 0,
    v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
