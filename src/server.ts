import fastify from 'fastify';
import cors from '@fastify/cors';
import tf from './tf';
import type * as TFT from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as fsp from 'fs/promises';
import * as path from 'path';
import { pathToFileURL } from 'node:url';
import { findBestMoveNN } from './ai';
import { performSwap2Negotiation, proposeInitialTriple } from './swap2_negotiation';
import type { Player } from './ai';
import * as dotenv from 'dotenv';

dotenv.config();

// --- Configuration ---
const BASE_DIR = process.env.APP_DIR || path.resolve(__dirname, '..');
const PORT = Number(process.env.PORT || 8080);
const MODEL_PATH = process.env.MODEL_PATH || path.resolve(BASE_DIR, 'gomoku_model_prod', 'model.json');
const MODEL_URL = process.env.MODEL_URL; // optional: remote Supabase Storage URL
const MODEL_CHECK_INTERVAL_MS = Number(process.env.MODEL_CHECK_INTERVAL_MS || 5 * 60 * 1000);

const EARLY_GAME_MOVES = Number(process.env.EARLY_GAME_MOVES || 6);
const SERVER_BOARD_SIZE = Number(process.env.BOARD_SIZE || 15);
const EARLY_GAME_THINK_TIME = Number(process.env.EARLY_GAME_THINK_TIME || 1500);
const MID_GAME_THINK_TIME = Number(process.env.MID_GAME_THINK_TIME || 3000);
const LATE_GAME_THINK_TIME = Number(process.env.LATE_GAME_THINK_TIME || 1500);
// Optional time control spec like "5+1" (seconds+increment). Used only as fallback
const TIME_CONTROL = (process.env.TIME_CONTROL || '5+1').trim();

type GetMoveBody = {
  board: (Player | null)[][];
  player: Player;
  moves?: [number, number][];
  turnEndsAt?: number; // epoch ms when turn ends
  timeLeftMs?: number; // remaining time for this turn
  turnLimitMs?: number; // server-provided per-turn limit
  // Optional: client can enforce a specific think time (e.g., easy mode)
  forceThinkTimeMs?: number;
};

let model: TFT.LayersModel | null = null;
let lastModelMtimeMs = 0;
let lastRemoteEtag: string | null = null;
// Opening book stored canonically (by symmetry) for robust lookup
let openingBook: { [hashCanon: string]: [number, number] } | null = null;

// --- Symmetry helpers for opening book ---
type TransformId = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7; // I, R90, R180, R270, FlipH, FlipV, Diag, AntiDiag
const INV_T: Record<TransformId, TransformId> = { 0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7 };

function transformRC(r: number, c: number, size: number, t: TransformId): [number, number] {
  const n = size;
  switch (t) {
    case 0: return [r, c];
    case 1: return [c, n - 1 - r]; // rot90
    case 2: return [n - 1 - r, n - 1 - c]; // rot180
    case 3: return [n - 1 - c, r]; // rot270
    case 4: return [r, n - 1 - c]; // flipH
    case 5: return [n - 1 - r, c]; // flipV
    case 6: return [c, r]; // main diag
    case 7: return [n - 1 - c, n - 1 - r]; // anti diag
  }
}

function hashToBoard(hash: string): (Player | null)[][] {
  const rows = hash.split('|');
  const n = rows.length;
  const b: (Player | null)[][] = Array.from({ length: n }, () => Array<Player | null>(n).fill(null));
  for (let r = 0; r < n; r++) {
    const row = rows[r];
    for (let c = 0; c < n; c++) {
      const ch = row[c];
      b[r][c] = ch === '-' ? null : ch === 'b' ? 'black' : 'white';
    }
  }
  return b;
}

function boardToHashStr(board: (Player | null)[][]): string {
  return board.map(row => row.map(cell => (cell ? cell[0] : '-')).join('')).join('|');
}

function transformBoard(board: (Player | null)[][], t: TransformId): (Player | null)[][] {
  const n = board.length;
  const out: (Player | null)[][] = Array.from({ length: n }, () => Array<Player | null>(n).fill(null));
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const [rr, cc] = transformRC(r, c, n, t);
      out[rr][cc] = board[r][c];
    }
  }
  return out;
}

function canonicalizeBoard(board: (Player | null)[][]): { hash: string; t: TransformId } {
  const trans: TransformId[] = [0, 1, 2, 3, 4, 5, 6, 7];
  let bestHash = '';
  let bestT: TransformId = 0;
  let first = true;
  for (const t of trans) {
    const h = boardToHashStr(transformBoard(board, t));
    if (first || h < bestHash) { bestHash = h; bestT = t; first = false; }
  }
  return { hash: bestHash, t: bestT };
}

async function loadOpeningBook(): Promise<void> {
  const seen = new Set<string>();
  const candidates: Array<{ filePath: string; label: string }> = [];
  const pushCandidate = (inputPath: string, label: string) => {
    const resolved = path.isAbsolute(inputPath) ? inputPath : path.resolve(BASE_DIR, inputPath);
    if (seen.has(resolved)) return;
    seen.add(resolved);
    candidates.push({ filePath: resolved, label });
  };

  const envPath = process.env.OPENING_BOOK_PATH;
  if (envPath) pushCandidate(envPath, 'OPENING_BOOK_PATH');
  pushCandidate(path.resolve(BASE_DIR, '..', 'opening_book_generated.json'), 'opening_book_generated.json');
  pushCandidate(path.resolve(BASE_DIR, '..', 'opening_book.json'), 'opening_book.json');

  for (const { filePath, label } of candidates) {
    try {
      if (!fs.existsSync(filePath)) continue;
      const raw = await fsp.readFile(filePath, 'utf-8');
      const entries = JSON.parse(raw) as Array<{ board_hash: string; best_move: [number, number]; move_count?: number }>;
      const filtered: { [hash: string]: [number, number] } = Object.create(null);
      let kept = 0;
      for (const entry of entries) {
        const board = hashToBoard(entry.board_hash);
        if (board.length !== SERVER_BOARD_SIZE) continue;
        const { hash, t } = canonicalizeBoard(board);
        const [rr, cc] = transformRC(entry.best_move[0], entry.best_move[1], board.length, t);
        filtered[hash] = [rr, cc];
        kept++;
      }
      if (kept > 0) {
        openingBook = filtered;
        const skipped = entries.length - kept;
        console.log(`[Server] Opening book loaded (${kept} entries) from ${label}.`);
        if (skipped > 0) {
          console.warn(`[Server] Skipped ${skipped} entries from ${label} due to board-size mismatch.`);
        }
        return;
      }
      console.warn(`[Server] No entries in ${label} matched board size ${SERVER_BOARD_SIZE}.`);
    } catch (e) {
      console.warn(`[Server] Failed to load opening book from ${label}:`, e);
    }
  }

  openingBook = null;
  console.warn('[Server] Opening book not loaded; all candidates unavailable or mismatched.');
}

async function loadModel(): Promise<TFT.LayersModel> {
  const useRemote = !!(MODEL_URL && MODEL_URL.startsWith('http'));
  let loadHref: string;
  if (useRemote) {
    // Append cache-busting param if we have ETag
    const v = lastRemoteEtag ? `?v=${encodeURIComponent(lastRemoteEtag)}` : '';
    loadHref = `${MODEL_URL}${v}`;
  } else {
    await fsp.stat(MODEL_PATH); // throws if missing
    loadHref = pathToFileURL(MODEL_PATH).href;
  }
  console.log(`[Server] Loading model: ${loadHref}`);
  const m = await tf.loadLayersModel(loadHref);
  // Warm-up with configured BOARD_SIZE
  tf.tidy(() => {
    const dummy = tf.zeros([1, SERVER_BOARD_SIZE, SERVER_BOARD_SIZE, 3]);
    const out = m.predict(dummy);
    void out;
  });
  model = m;
  if (!useRemote) {
    const stat = await fsp.stat(MODEL_PATH);
    lastModelMtimeMs = stat.mtimeMs;
  }
  return m;
}

async function ensureModel(): Promise<TFT.LayersModel> {
  const useRemote = !!(MODEL_URL && MODEL_URL.startsWith('http'));
  if (!model) return loadModel();
  if (useRemote) {
    try {
      const head = await fetch(MODEL_URL!, { method: 'HEAD' });
      const etag = head.headers.get('etag') || head.headers.get('ETag');
      if (etag && etag !== lastRemoteEtag) {
        console.log('[Server] Remote model ETag changed. Reloading...');
        lastRemoteEtag = etag;
        model.dispose();
        return loadModel();
      }
    } catch (e) {
      console.warn('[Server] Failed to HEAD remote model:', e);
    }
  } else {
    try {
      const stat = await fsp.stat(MODEL_PATH);
      if (stat.mtimeMs > lastModelMtimeMs) {
        console.log('[Server] Model file changed. Reloading...');
        model.dispose();
        return loadModel();
      }
    } catch (e) {
      console.warn('[Server] Failed to stat model file:', e);
    }
  }
  return model;
}

async function start() {
  const app = fastify({ logger: false });
  await app.register(cors, { origin: true });

  await loadOpeningBook();

  app.get('/health', async () => {
    try {
      await ensureModel();
      return { ok: true, modelPath: MODEL_PATH };
    } catch (e) {
      return { ok: false, error: String(e) };
    }
  });

  // Swap2 helper endpoints for AI-vs-Player or online matches
  // 1) Propose initial triple (B-W-B) when AI is the first player
  app.post<{ Body: { board: (Player|null)[][] } }>('/swap2/propose', async (req, reply) => {
    try {
      const { board } = req.body || ({} as any);
      if (!board) return reply.code(400).send({ error: "Missing 'board'" });
      const opened = proposeInitialTriple(board);
      // After B-W-B, White to move
      return reply.code(200).send({ board: opened, toMove: 'white' });
    } catch (e) {
      return reply.code(500).send({ error: String(e) });
    }
  });

  // 2) Choose second player's Swap2 option (AI is second player):
  //    Input board should already contain the initial triple B-W-B.
  app.post<{ Body: { board: (Player|null)[][] } }>('/swap2/second', async (req, reply) => {
    try {
      const { board } = req.body || ({} as any);
      if (!board) return reply.code(400).send({ error: "Missing 'board'" });
      const m = await ensureModel();
      const res = await performSwap2Negotiation(board as any, m as any);
      return reply.code(200).send(res);
    } catch (e) {
      return reply.code(500).send({ error: String(e) });
    }
  });

  function parseTimeControl(tc: string): { baseMs: number; incMs: number } {
    const m = tc.match(/^(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)$/);
    if (!m) return { baseMs: 1500, incMs: 0 };
    const base = Math.max(0, Math.round(parseFloat(m[1]) * 1000));
    const inc = Math.max(0, Math.round(parseFloat(m[2]) * 1000));
    return { baseMs: base, incMs: inc };
  }

  function computeThinkTime(nowMs: number, body: GetMoveBody): number {
    // If client forces a specific think time (e.g., easy mode), honor it within sane bounds
    if (typeof body.forceThinkTimeMs === 'number' && isFinite(body.forceThinkTimeMs)) {
      const forced = Math.max(200, Math.min(5000, Math.floor(body.forceThinkTimeMs)));
      return forced;
    }
    const moves = body.moves ?? [];
    // Base by phase as fallback
    let phaseBase = moves.length <= EARLY_GAME_MOVES
      ? EARLY_GAME_THINK_TIME
      : moves.length <= 30
      ? MID_GAME_THINK_TIME
      : LATE_GAME_THINK_TIME;

    // If custom time is provided, scale dynamically
    const safetyMs = 200; // leave some margin to avoid flag fall
    let remain: number | null = null;
    const now = nowMs;
    if (typeof body.timeLeftMs === 'number') remain = Math.max(0, body.timeLeftMs - safetyMs);
    else if (typeof body.turnLimitMs === 'number') remain = Math.max(0, body.turnLimitMs - safetyMs);
    else if (typeof body.turnEndsAt === 'number') remain = Math.max(0, body.turnEndsAt - now - safetyMs);
    
    // If no explicit per-turn time provided, approximate remaining time using TIME_CONTROL (e.g., 5+1)
    if (remain == null) {
      const { baseMs, incMs } = parseTimeControl(TIME_CONTROL);
      // Approximate own moves from board state
      const board = body.board;
      const player = body.player;
      if (board && player) {
        let own = 0;
        for (let r = 0; r < board.length; r++) {
          const row = board[r];
          for (let c = 0; c < row.length; c++) if (row[c] === player) own++;
        }
        // Crude estimate: base + increments earned so far
        const approxRemain = Math.max(0, baseMs + own * incMs - safetyMs);
        const frac = moves.length <= EARLY_GAME_MOVES ? 0.35 : moves.length <= 30 ? 0.55 : 0.5;
        const dynamic = Math.max(800, Math.floor(approxRemain * frac));
        return Math.min(dynamic, Math.max(phaseBase, Math.floor(approxRemain * 0.9)));
      }
      return phaseBase;
    }

    // Allocate a fraction depending on phase; be conservative in early, more in mid
    const frac = moves.length <= EARLY_GAME_MOVES ? 0.35 : moves.length <= 30 ? 0.55 : 0.5;
    const dynamic = Math.max(800, Math.floor(remain * frac));
    return Math.min(dynamic, Math.max(phaseBase, Math.floor(remain * 0.9))); // cap below remaining
  }

  app.post<{ Body: GetMoveBody }>('/get-move', async (req, reply) => {
    try {
      const { board, player, moves = [] } = req.body || ({} as GetMoveBody);
      if (!board || !player) {
        return reply.code(400).send({ error: "Missing 'board' or 'player'" });
      }
      const m = await ensureModel();
      // Validate board size vs model input
      const ishape = (m.inputs?.[0]?.shape || [null, SERVER_BOARD_SIZE, SERVER_BOARD_SIZE, 3]) as number[];
      const expected = ishape[1] || SERVER_BOARD_SIZE;
      if (board.length !== expected) {
        return reply.code(400).send({ error: `Board size ${board.length}x${board.length} does not match model ${expected}x${expected}.` });
      }
      const nowMs = Date.now();
      const thinkTime = computeThinkTime(nowMs, req.body as GetMoveBody);

      // Opening book for first 12 moves if present, canonical matching + inverse transform
      if (openingBook && moves.length <= 12) {
        const { hash, t } = canonicalizeBoard(board);
        const mvCanon = openingBook[hash];
        if (mvCanon) {
          const invT = INV_T[t];
          const [r, c] = transformRC(mvCanon[0], mvCanon[1], board.length, invT);
          if (board[r]?.[c] === null) {
            return reply.code(200).send({ move: [r, c] as [number, number], source: 'book' });
          }
        }
      }

      const { bestMove } = await findBestMoveNN(m, board, player, thinkTime);
      return reply.code(200).send({ move: bestMove });
    } catch (e) {
      req.log?.error(e);
      return reply.code(500).send({ error: String(e) });
    }
  });

  // Prime model at startup
  await ensureModel();

  // Periodic model check
  setInterval(() => {
    void ensureModel();
  }, MODEL_CHECK_INTERVAL_MS).unref();

  await app.listen({ port: PORT, host: '0.0.0.0' });
  console.log(`[Server] Listening on :${PORT}`);
}

start().catch((e) => {
  console.error('Server failed:', e);
  process.exitCode = 1;
});
