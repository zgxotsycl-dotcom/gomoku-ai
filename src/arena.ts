import tf from './tf';
import type * as TFT from '@tensorflow/tfjs';
import * as path from 'path';
import * as fs from 'fs';
import * as fse from 'fs-extra';
import { createClient } from '@supabase/supabase-js';
import { createHash } from 'crypto';
import { findBestMoveNN, checkWin, getOpponent, type Player } from './ai';
import { performSwap2Negotiation } from './swap2_negotiation';
import { generateSwap2Opening } from './rules_swap2';
import { updateStatus } from './status';

type Board = (Player | null)[][];

const BASE_DIR = process.env.APP_DIR || path.resolve(__dirname, '..');
const PROD_DIR = process.env.PROD_MODEL_DIR || path.resolve(BASE_DIR, 'gomoku_model_prod');
const CANDIDATE_DIR = process.env.CANDIDATE_SAVE_DIR || path.resolve(BASE_DIR, 'gomoku_model_candidate');
const PAST_MODELS_DIR = process.env.PAST_MODELS_DIR || path.resolve(BASE_DIR, 'past_models');
const GAMES = Number(process.env.ARENA_GAMES || 200);
const THINK_TIME = Number(process.env.ARENA_THINK_TIME || 3000);
const BOARD_SIZE = Number(process.env.BOARD_SIZE || 15);
const WINRATE_THRESHOLD = Number(process.env.WINRATE_THRESHOLD || 0.6);
const PROMOTE_ON_PASS = (process.env.PROMOTE_ON_PASS || 'true').toLowerCase() === 'true';
const EARLY_STOP = (process.env.ARENA_EARLY_STOP || 'true').toLowerCase() === 'true';
const LOG_EVAL_TO_SUPABASE = (process.env.LOG_EVAL_TO_SUPABASE || 'true').toLowerCase() === 'true';

function ensureSupabaseEnv() {
  if (process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_ROLE_KEY) return;
  const candidate = path.resolve(__dirname, '..', 'gomoku-app-v2', '.env.local');
  try {
    if (fs.existsSync(candidate)) {
      const txt = fs.readFileSync(candidate, 'utf-8');
      for (const line of txt.split(/\r?\n/)) {
        const m = line.match(/^([A-Z0-9_]+)=(.*)$/);
        if (m) {
          const k = m[1];
          const v = m[2];
          if (!process.env[k]) process.env[k] = v;
        }
      }
    }
  } catch {}
}

ensureSupabaseEnv();
const SUPABASE_URL = process.env.SUPABASE_URL || '';
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || '';

function emptyBoard(): Board { return Array.from({ length: BOARD_SIZE }, () => Array<Player | null>(BOARD_SIZE).fill(null)); }

async function loadModelFromDir(dir: string): Promise<TFT.LayersModel> {
  const p = path.join(dir, 'model.json');
  if (!fs.existsSync(p)) throw new Error(`Model not found: ${p}`);
  const url = `file://${p}`;
  const m = await tf.loadLayersModel(url);
  // warmup
  tf.tidy(() => { const x = tf.zeros([1, BOARD_SIZE, BOARD_SIZE, 3]); const y = m.predict(x); void y; });
  return m;
}

async function playSingleGame(blackModel: TFT.LayersModel, whiteModel: TFT.LayersModel): Promise<Player | 0> {
  let board = emptyBoard();
  let player: Player = 'black';
  // Apply Swap2 negotiation if enabled (second player evaluates = initial White)
  try {
    const rule = (process.env.OPENING_RULE || 'swap2').toLowerCase();
    if (rule === 'swap2') {
      const { board: opened, toMove, swapColors } = await performSwap2Negotiation(board, whiteModel);
      board = opened;
      player = toMove;
      if (swapColors) {
        const tmp = blackModel; (blackModel as any) = whiteModel; (whiteModel as any) = tmp;
      }
    }
  } catch {}
  const maxMoves = BOARD_SIZE * BOARD_SIZE;
  for (let move = 0; move < maxMoves; move++) {
    const model = player === 'black' ? blackModel : whiteModel;
    const { bestMove } = await findBestMoveNN(model, board, player, THINK_TIME);
    const [r, c] = bestMove;
    if (r < 0 || c < 0 || board[r][c] !== null) return 0; // invalid -> draw
    board[r][c] = player;
    if (checkWin(board, player, [r, c])) return player;
    player = getOpponent(player);
  }
  return 0;
}

function modelFingerprint(dir: string): string {
  try {
    const mj = fs.readFileSync(path.join(dir, 'model.json'), 'utf-8');
    const hash = createHash('sha256');
    hash.update(mj);
    return hash.digest('hex').slice(0, 12);
  } catch { return 'unknown'; }
}

async function logToSupabase(payload: any) {
  if (!LOG_EVAL_TO_SUPABASE || !SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) return;
  try {
    const client = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, { auth: { persistSession: false } });
    await client.from('ai_model_evaluations').insert(payload);
  } catch (e) {
    console.warn('Supabase logging failed:', e);
  }
}

async function main() {
  console.log('=== Arena Gating ===');
  if (!fs.existsSync(CANDIDATE_DIR)) throw new Error(`Candidate model not found: ${CANDIDATE_DIR}`);
  if (!fs.existsSync(PROD_DIR)) throw new Error(`Prod model not found: ${PROD_DIR}`);

  const prod = await loadModelFromDir(PROD_DIR);
  const cand = await loadModelFromDir(CANDIDATE_DIR);

  let candWins = 0, prodWins = 0, draws = 0;
  for (let i = 0; i < GAMES; i++) {
    const swap = i % 2 === 1; // alternate colors
    const black = swap ? prod : cand;
    const white = swap ? cand : prod;
    const res = await playSingleGame(black, white);
    if (res === 0) draws++;
    else if ((res === 'black' && !swap) || (res === 'white' && swap)) candWins++;
    else prodWins++;

    const played = i + 1;
    if (played % 10 === 0 || played === GAMES) {
      console.log(`Progress: ${played}/${GAMES} | cand=${candWins} prod=${prodWins} draw=${draws}`);
    }
    if (played % 10 === 0) {
      await updateStatus({ arena: { played, total: GAMES, candWins, prodWins, draws } });
    }

    if (EARLY_STOP) {
      const remaining = GAMES - played;
      const bestPossible = candWins + remaining;
      const worstPossible = candWins;
      const maxWinrate = bestPossible / GAMES;
      const minWinrate = worstPossible / GAMES;
      if (minWinrate >= WINRATE_THRESHOLD) {
        console.log(`Early stop: candidate already above threshold with minimum winrate ${(minWinrate * 100).toFixed(1)}%.`);
        break;
      }
      if (maxWinrate < WINRATE_THRESHOLD) {
        console.log(`Early stop: candidate can no longer reach threshold (max ${(maxWinrate * 100).toFixed(1)}%).`);
        break;
      }
    }
  }

  const total = candWins + prodWins + draws;
  const winrate = total > 0 ? candWins / total : 0;
  console.log(`Result: candidate=${candWins}, prod=${prodWins}, draws=${draws}, winrate=${(winrate * 100).toFixed(1)}%`);

  const payload = {
    ts: new Date().toISOString(),
    games: total,
    candidate_wins: candWins,
    prod_wins: prodWins,
    draws,
    winrate,
    candidate_fingerprint: modelFingerprint(CANDIDATE_DIR),
    prod_fingerprint: modelFingerprint(PROD_DIR),
    threshold: WINRATE_THRESHOLD,
  };
  await logToSupabase(payload);

  let promoted = false;
  if (winrate >= WINRATE_THRESHOLD && PROMOTE_ON_PASS) {
    // Dispose models before file ops to avoid busy/locked handles
    try { (prod as any)?.dispose?.(); } catch {}
    try { (cand as any)?.dispose?.(); } catch {}
    // Promote candidate to prod, backup old prod (retry on EBUSY)
    await fse.ensureDir(PAST_MODELS_DIR);
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backup = path.join(PAST_MODELS_DIR, `prod_${stamp}`);
    await fse.copy(PROD_DIR, backup);
    const maxTries = 10;
    let ok = false; let lastErr: any = null;
    for (let t = 0; t < maxTries; t++) {
      try {
        await fse.rm(PROD_DIR, { recursive: true, force: true });
        ok = true; break;
      } catch (e: any) {
        lastErr = e;
        if (e?.code === 'EBUSY') { await new Promise(r => setTimeout(r, 500)); continue; }
        else break;
      }
    }
    if (!ok && lastErr) throw lastErr;
    await fse.copy(CANDIDATE_DIR, PROD_DIR);
    console.log(`Promoted candidate to prod. Backup saved at ${backup}`);
    promoted = true;
  } else {
    console.log('Promotion skipped (did not meet threshold or disabled).');
  }

  await updateStatus({ arena: { played: total, total: GAMES, candWins, prodWins, draws, promoted } });

  // Write arena result file for pipeline to consume
  const resultPath = process.env.ARENA_RESULT_PATH || path.resolve(__dirname, '..', 'arena_result.json');
  const payloadOut = {
    ...payload,
    promoted,
  };
  try {
    await fse.writeJson(resultPath, payloadOut, { spaces: 2 });
    console.log(`Arena result written to ${resultPath}`);
  } catch (e) {
    console.warn('Failed to write arena result file:', e);
  }
}

main().catch((e) => {
  console.error('Arena failed:', e);
  process.exitCode = 1;
});
