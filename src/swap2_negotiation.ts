import tf from './tf';
import type * as TFT from '@tensorflow/tfjs';
import type { Player } from './ai';
import { findBestMoveNN, getOpponent, checkWin } from './ai';
import { getNumber, applyDistancePenalty } from './tuning';

export type Board = (Player | null)[][];

function cloneBoard(b: Board): Board {
  return b.map(row => row.slice()) as Board;
}

function inside(n: number, r: number, c: number) {
  return r >= 0 && c >= 0 && r < n && c < n;
}

function firstEmptyAround(board: Board, r0: number, c0: number, rings = 2): [number, number] {
  const n = board.length;
  for (let rad = 1; rad <= rings; rad++) {
    for (let dr = -rad; dr <= rad; dr++) {
      for (let dc = -rad; dc <= rad; dc++) {
        const r = r0 + dr, c = c0 + dc;
        if (!inside(n, r, c)) continue;
        if (board[r][c] === null) return [r, c];
      }
    }
  }
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) if (board[r][c] === null) return [r, c];
  return [-1, -1];
}

function boardToInputTensor(board: Board, player: Player): TFT.Tensor4D {
  const size = board.length;
  const channels = 3;
  const opp = player === 'black' ? 'white' : 'black';
  const sideVal = player === 'black' ? 1 : 0;
  const data = new Float32Array(size * size * channels);
  let idx = 0;
  for (let r = 0; r < size; r++) {
    const row = board[r];
    for (let c = 0; c < size; c++) {
      const cell = row[c];
      data[idx] = cell === player ? 1 : 0;
      data[idx + 1] = cell === opp ? 1 : 0;
      data[idx + 2] = sideVal;
      idx += channels;
    }
  }
  return tf.tensor4d(data, [1, size, size, channels]) as TFT.Tensor4D;
}

// Ranking and cache helpers
function boardStr(board: Board): string {
  return board.map(row => row.map(cell => (cell === 'black' ? 'b' : cell === 'white' ? 'w' : '-')).join('')).join('|');
}

// 8-way symmetry transforms (same as AI)
type T = 0|1|2|3|4|5|6|7;
function transformRC(r: number, c: number, n: number, t: T): [number, number] {
  switch (t) {
    case 0: return [r, c];
    case 1: return [c, n - 1 - r];
    case 2: return [n - 1 - r, n - 1 - c];
    case 3: return [n - 1 - c, r];
    case 4: return [r, n - 1 - c];
    case 5: return [n - 1 - r, c];
    case 6: return [c, r];
    case 7: return [n - 1 - c, n - 1 - r];
  }
}
function transformBoardLocal(board: Board, t: T): Board {
  const n = board.length;
  const out: Board = Array.from({ length: n }, () => Array<Player|null>(n).fill(null));
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const [rr, cc] = transformRC(r, c, n, t);
    out[rr][cc] = board[r][c];
  }
  return out;
}
function canonicalHash(board: Board): string {
  const trans: T[] = [0,1,2,3,4,5,6,7];
  let best = '';
  let first = true;
  for (const t of trans) {
    const h = boardStr(t === 0 ? board : transformBoardLocal(board, t));
    if (first || h < best) { best = h; first = false; }
  }
  return best;
}

async function policyForBoard(model: TFT.LayersModel, board: Board, player: Player): Promise<Float32Array> {
  const x = boardToInputTensor(board, player);
  const out = model.predict(x) as TFT.Tensor[];
  const p = await out[0].data() as Float32Array;
  x.dispose();
  out.forEach(t => t.dispose());
  return p;
}

async function policyWithSymmetryAvg(model: TFT.LayersModel, board: Board, player: Player, sym: number): Promise<Float32Array> {
  const n = board.length;
  const trans: T[] = sym >= 8 ? [0,1,2,3,4,5,6,7] : (sym >= 4 ? [0,1,2,3] : [0]);
  const acc = new Float64Array(n * n);
  for (const t of trans) {
    const b = (t === 0) ? board : transformBoardLocal(board, t);
    const p = await policyForBoard(model, b, player);
    // map back to original orientation
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const [rr, cc] = transformRC(r, c, n, t);
      const idx = r * n + c;
      const idxT = rr * n + cc;
      acc[idx] += p[idxT] || 0;
    }
  }
  const out = new Float32Array(n * n);
  const k = trans.length;
  for (let i = 0; i < out.length; i++) out[i] = acc[i] / k;
  return out;
}

async function topKPolicyMoves(model: TFT.LayersModel, board: Board, player: Player, k: number): Promise<Array<[number, number]>> {
  const size = board.length;
  const pol = await policyForBoard(model, board, player);
  const items: Array<{ r: number; c: number; p: number }> = [];
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      if (board[r][c] === null) {
        const idx = r * size + c;
        items.push({ r, c, p: pol[idx] ?? 0 });
      }
    }
  }
  items.sort((a, b) => b.p - a.p);
  const out: Array<[number, number]> = [];
  for (let i = 0; i < Math.min(k, items.length); i++) out.push([items[i].r, items[i].c]);
  if (out.length === 0) {
    // Fallback to center proximity
    const n = size, mid = Math.floor(n / 2);
    const list: Array<[number, number]> = [];
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) if (board[r][c] === null) list.push([r, c]);
    list.sort((a, b) => (Math.abs(a[0]-mid)+Math.abs(a[1]-mid)) - (Math.abs(b[0]-mid)+Math.abs(b[1]-mid)));
    return list.slice(0, k);
  }
  return out;
}

// --- Simple tactical pattern helpers (open four / immediate win / block) ---
const DIRS: Array<[number, number]> = [ [0,1], [1,0], [1,1], [1,-1] ];

function countInLine(board: Board, r: number, c: number, dr: number, dc: number, player: Player): { count: number; openLeft: boolean; openRight: boolean } {
  const n = board.length;
  let cnt = 1;
  let rr = r + dr, cc = c + dc;
  while (rr >= 0 && rr < n && cc >= 0 && cc < n && board[rr][cc] === player) { cnt++; rr += dr; cc += dc; }
  const openRight = (rr >= 0 && rr < n && cc >= 0 && cc < n && board[rr][cc] === null);
  rr = r - dr; cc = c - dc;
  while (rr >= 0 && rr < n && cc >= 0 && cc < n && board[rr][cc] === player) { cnt++; rr -= dr; cc -= dc; }
  const openLeft = (rr >= 0 && rr < n && cc >= 0 && cc < n && board[rr][cc] === null);
  return { count: cnt, openLeft, openRight };
}

function isOpenFour(board: Board, r: number, c: number, player: Player): boolean {
  for (const [dr, dc] of DIRS) {
    const { count, openLeft, openRight } = countInLine(board, r, c, dr, dc, player);
    if (count === 4 && openLeft && openRight) return true;
  }
  return false;
}

function listEmpty(board: Board): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (let r = 0; r < board.length; r++) for (let c = 0; c < board.length; c++) if (board[r][c] === null) out.push([r,c]);
  return out;
}

function findImmediateWins(board: Board, player: Player): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (const [r,c] of listEmpty(board)) {
    board[r][c] = player;
    const win = checkWin(board as any, player, [r,c]);
    board[r][c] = null;
    if (win) out.push([r,c]);
  }
  return out;
}

function findOpenFourMoves(board: Board, player: Player, maxCount: number): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (const [r,c] of listEmpty(board)) {
    board[r][c] = player;
    if (isOpenFour(board, r, c, player)) out.push([r,c]);
    board[r][c] = null;
    if (out.length >= maxCount) break;
  }
  return out;
}

function mergeUnique(a: Array<[number, number]>, b: Array<[number, number]>, limit: number): Array<[number, number]> {
  const seen = new Set<string>();
  const push = (r: number, c: number, arr: Array<[number, number]>) => {
    const k = r+','+c; if (!seen.has(k)) { seen.add(k); arr.push([r,c]); }
  };
  const out: Array<[number, number]> = [];
  for (const [r,c] of a) { if (out.length >= limit) break; push(r,c,out); }
  for (const [r,c] of b) { if (out.length >= limit) break; push(r,c,out); }
  return out;
}

function valueFor(model: TFT.LayersModel, board: Board, toMove: Player): number {
  return tf.tidy(() => {
    const x = boardToInputTensor(board, toMove);
    const out = model.predict(x) as TFT.Tensor[];
    const valT = out[1];
    const v = (valT.dataSync() as Float32Array)[0];
    x.dispose();
    // out tensors are disposed by tidy
    return v;
  });
}

/**
 * Rollout with small NN-guided search over a few plies to better estimate value
 * from the perspective of the original 'toMove'.
 * - msBudget split evenly per ply (min 80ms)
 * - early exit on immediate win detection
 */
async function rolloutValueMulti(
  model: TFT.LayersModel,
  board: Board,
  toMove: Player,
  msBudget: number,
  plies: number
): Promise<number> {
  const local = rolloutValueMulti as unknown as { _memo?: Map<string, number> };
  if (!local._memo) local._memo = new Map();
  const memo = local._memo;
  const minMs = 80;
  const budget = Math.max(minMs, Math.floor(msBudget));
  const perPly = Math.max(minMs, Math.floor(budget / Math.max(1, plies)));
  const orig = toMove;
  const b: Board = cloneBoard(board);
  let cur: Player = toMove;
  for (let i = 0; i < plies; i++) {
    const { bestMove } = await findBestMoveNN(model as any, b as any, cur, perPly);
    const [r, c] = bestMove as [number, number];
    if (r == null || r < 0 || b[r][c] !== null) break;
    b[r][c] = cur;
    if (checkWin(b as any, cur, [r, c])) {
      // winner is cur
      return cur === orig ? 1 : -1;
    }
    cur = getOpponent(cur);
  }
  // Evaluate terminal with value head for current to-move
  const key = canonicalHash(b) + '|' + cur;
  let v = memo.get(key);
  if (v === undefined) { v = valueFor(model, b, cur); memo.set(key, v); if (memo.size > 1000) memo.clear(); }
  // Convert to original perspective
  // If an odd number of plies were played, perspective flips
  const flips = plies % 2 === 1 ? -1 : 1;
  return flips * (v as number);
}

function enumerateCandidates(board: Board, rings = 2, limit = 24): Array<[number, number]> {
  const n = board.length;
  const mid = Math.floor(n / 2);
  const list: Array<[number, number]> = [];
  for (let dr = -rings; dr <= rings; dr++) {
    for (let dc = -rings; dc <= rings; dc++) {
      const r = mid + dr, c = mid + dc;
      if (!inside(n, r, c)) continue;
      if (board[r][c] === null) list.push([r, c]);
    }
  }
  list.sort((a, b) => (Math.abs(a[0] - mid) + Math.abs(a[1] - mid)) - (Math.abs(b[0] - mid) + Math.abs(b[1] - mid)));
  return list.slice(0, limit);
}

export function proposeInitialTriple(board: Board): Board {
  const n = board.length;
  const mid = Math.floor(n / 2);
  const b = cloneBoard(board);
  if (b[mid][mid] == null) b[mid][mid] = 'black';
  const w1 = firstEmptyAround(b, mid, mid, 1);
  if (w1[0] !== -1) b[w1[0]][w1[1]] = 'white';
  const b2 = firstEmptyAround(b, mid, mid, 1);
  if (b2[0] !== -1) b[b2[0]][b2[1]] = 'black';
  return b;
}

export async function performSwap2Negotiation(initial: Board, model: TFT.LayersModel): Promise<{ board: Board; toMove: Player; swapColors: boolean }> {
  // Step 1: proposer places three stones B-W-B (heuristic near center)
  const afterThree = proposeInitialTriple(initial);
  const n = afterThree.length;
  const toMoveAfterThree: Player = 'white'; // last placed was black

  const R_MS = Number(process.env.SWAP2_ROLLOUT_MS || 500);
  const R_PLIES = Number(process.env.SWAP2_ROLLOUT_PLIES || 3);
  const polCache = new Map<string, Float32Array>();

  async function topKWithCache(board: Board, player: Player, k: number): Promise<Array<[number, number]>> {
    const key = canonicalHash(board) + '|' + player + '|pol';
    let pol = polCache.get(key);
    if (!pol) {
      const SYM = Math.max(0, Math.floor(getNumber('SWAP2_SYM_AVG', 8)));
      pol = await (SYM > 1 ? policyWithSymmetryAvg(model, board, player, SYM) : policyForBoard(model, board, player));
      polCache.set(key, pol);
    }
    // reuse the ranking logic
    const size = board.length;
    const items: Array<{ r: number; c: number; p: number }> = [];
    // Early-phase distance penalty to reduce noise
    let stones = 0; for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) if (board[r][c] != null) stones++;
    const EARLY_MOVES = Math.max(0, Math.floor(getNumber('SWAP2_DIST_PHASE_MOVES', 8)));
    const LAMBDA = getNumber('SWAP2_DIST_LAMBDA', 0);
    const usePenalty = LAMBDA > 0 && stones <= EARLY_MOVES;
    for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) if (board[r][c] === null) {
      const base = pol[r*size+c] ?? 0;
      const factor = usePenalty ? applyDistancePenalty(size, r, c, LAMBDA, 'exp') : 1;
      items.push({ r, c, p: base * factor });
    }
    items.sort((a, b) => b.p - a.p);
    const out: Array<[number, number]> = [];
    for (let i = 0; i < Math.min(k, items.length); i++) out.push([items[i].r, items[i].c]);
    if (out.length === 0) return topKPolicyMoves(model, board, player, k);
    return out;
  }

  // Helper: rollout-enhanced value for current to-move
  async function rolloutValue(board: Board, toMove: Player): Promise<number> {
    // multi-plies shallow rollout, fallback internally to value head, with canonical memo reuse
    return rolloutValueMulti(model, board, toMove, R_MS, R_PLIES);
  }

  // Option 1: Second player chooses to be Black (swap colors); next to move stays White.
  // Score from second (Black) perspective = -v_white_toMove
  const v1_roll = await rolloutValue(afterThree, toMoveAfterThree);
  const scoreOpt1 = -v1_roll; // second player's (Black) advantage

  // Option 2: Second plays as White and places one more White stone, then Black to move.
  let bestOpt2Score = -Infinity;
  let bestOpt2Board: Board | null = null;
  const W_TOPK = Number(process.env.SWAP2_CAND_W_TOPK || 16);
  const USE_PAT = String(process.env.SWAP2_USE_PATTERNS || 'true').toLowerCase() === 'true';
  let candW = await topKWithCache(afterThree, 'white', W_TOPK);
  if (USE_PAT) {
    // Merge tactical candidates: immediate wins first, then open-fours, then policy top‑k
    const winsW = findImmediateWins(afterThree, 'white');
    const blocksW: Array<[number, number]> = [];
    // urgent blocks of Black immediate wins
    for (const [br,bc] of findImmediateWins(afterThree, 'black')) blocksW.push([br,bc]);
    const ofW = findOpenFourMoves(afterThree, 'white', Math.max(4, Math.floor(W_TOPK/2)));
    candW = mergeUnique([...winsW, ...blocksW], mergeUnique(ofW, candW, W_TOPK), W_TOPK);
  }
  for (const [wr, wc] of candW) {
    if (afterThree[wr][wc] !== null) continue;
    const b2 = cloneBoard(afterThree);
    b2[wr][wc] = 'white';
    // After White extra, Black to move; rollout for Black, then convert to White (second) perspective
    const vBlack = await rolloutValue(b2, 'black');
    const score = -vBlack; // second is White
    if (score > bestOpt2Score) { bestOpt2Score = score; bestOpt2Board = b2; }
  }

  // Option 3: Second places W then B, then First chooses color to maximize own advantage.
  let bestOpt3Score = -Infinity; // from second player's perspective
  let bestOpt3Board: Board | null = null;
  let candW3 = await topKWithCache(afterThree, 'white', Math.max(8, Math.floor(W_TOPK/2)));
  if (USE_PAT) {
    const winsW3 = findImmediateWins(afterThree, 'white');
    const ofW3 = findOpenFourMoves(afterThree, 'white', Math.max(4, Math.floor(W_TOPK/3)));
    candW3 = mergeUnique([...winsW3, ...ofW3], candW3, Math.max(8, Math.floor(W_TOPK/2)));
  }
  for (const [wr, wc] of candW3) {
    if (afterThree[wr][wc] !== null) continue;
    const tmpW = cloneBoard(afterThree);
    tmpW[wr][wc] = 'white';
    const B_TOPK = Number(process.env.SWAP2_CAND_B_TOPK || 12);
    let candB3 = await topKWithCache(tmpW, 'black', B_TOPK);
    if (USE_PAT) {
      const winsB3 = findImmediateWins(tmpW, 'black');
      const blocksB3: Array<[number, number]> = [];
      for (const [wr2, wc2] of findImmediateWins(tmpW, 'white')) blocksB3.push([wr2, wc2]);
      const ofB3 = findOpenFourMoves(tmpW, 'black', Math.max(4, Math.floor(B_TOPK/3)));
      candB3 = mergeUnique([...winsB3, ...blocksB3], candB3, B_TOPK);
    }
    for (const [br, bc] of candB3) {
      if (tmpW[br][bc] !== null) continue;
      const b3 = cloneBoard(tmpW);
      b3[br][bc] = 'black'; // last placed is Black => White to move
      const vW = await rolloutValue(b3, 'white');
      const vB = await rolloutValue(b3, 'black');
      // First player chooses the color (White or Black) giving larger advantage
      const firstPlayerAdv = Math.max(vW, vB);
      const secondPlayerScore = -firstPlayerAdv;
      if (secondPlayerScore > bestOpt3Score) { bestOpt3Score = secondPlayerScore; bestOpt3Board = b3; }
    }
  }

  // Choose best option for second player
  let finalBoard: Board;
  let toMove: Player;
  let swapColors = false;
  let bestScore = scoreOpt1;
  finalBoard = afterThree;
  toMove = 'white';
  // Option 2
  if (bestOpt2Board && bestOpt2Score > bestScore) {
    bestScore = bestOpt2Score;
    finalBoard = bestOpt2Board;
    toMove = 'black';
    swapColors = false;
  }
  // Option 3
  if (bestOpt3Board && bestOpt3Score > bestScore) {
    bestScore = bestOpt3Score;
    finalBoard = bestOpt3Board;
    toMove = 'white';
    // swap depends on first player's choice at this board; P1 picks White if vW >= 0
    const vW = await rolloutValue(finalBoard, 'white');
    const vB = await rolloutValue(finalBoard, 'black');
    // If picking Black yields higher advantage, colors are swapped
    swapColors = (vB > vW);
  }
  // Option 1 remains default if no better
  if (bestScore === scoreOpt1) {
    swapColors = true; // second chose to be Black
    toMove = 'white';
  }

  return { board: finalBoard, toMove, swapColors };
}
