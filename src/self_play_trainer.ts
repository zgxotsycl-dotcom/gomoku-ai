import { Worker } from 'node:worker_threads';
import * as path from 'path';
import * as fs from 'node:fs';
import * as fse from 'fs-extra';
import tf from './tf';
import type * as TFT from '@tensorflow/tfjs';
import { createDualResNetModel } from './model';
import { updateStatus } from './status';
import { customAlphabet } from 'nanoid';
import type { TrainingSample } from './types/training';

// -------- Config --------
const BASE_DIR = process.env.APP_DIR || path.resolve(__dirname, '..');
const NUM_WORKERS = Number(process.env.NUM_WORKERS || 4);
const SAVE_INTERVAL_MS = Number(process.env.SAVE_INTERVAL_MS || 30_000);
const OUTPUT_DIR = process.env.REPLAY_BUFFER_DIR || path.resolve(BASE_DIR, 'replay_buffer');
const PROD_MODEL_DIR = process.env.PROD_MODEL_DIR || path.resolve(BASE_DIR, 'gomoku_model_prod');
const PAST_MODELS_DIR = process.env.PAST_MODELS_DIR || path.resolve(BASE_DIR, 'past_models');
const SELF_PLAY_DURATION_MS = Number(process.env.SELF_PLAY_DURATION_MS || 30 * 60 * 1000);
const PAST_MODEL_PROBABILITY = Number(process.env.PAST_MODEL_PROBABILITY || 0.5);
const BOARD_SIZE = Number(process.env.BOARD_SIZE || 15);
const MCTS_THINK_TIME_MS = Number(process.env.MCTS_THINK_TIME_MS || 4000);
const EXPLORATION_MOVES = Number(process.env.EXPLORATION_MOVES || 15);

// -------- State --------
const nanoid = customAlphabet('1234567890abcdef', 10);
const WORKER_PATH = path.resolve(__dirname, './game_worker.js');
let isRunning = false;

async function ensureDirs() {
  await fse.ensureDir(OUTPUT_DIR);
  await fse.ensureDir(PAST_MODELS_DIR);
}

function resolveModelJson(dir: string): string | null {
  const p = path.join(dir, 'model.json');
  return fs.existsSync(p) ? p : null;
}

function pickRandom<T>(arr: T[]): T | null {
  if (arr.length === 0) return null;
  const i = Math.floor(Math.random() * arr.length);
  return arr[i];
}

async function pickOpponentModelPath(): Promise<string | null> {
  try {
    if (!fs.existsSync(PAST_MODELS_DIR)) return null;
    const entries = await fse.readdir(PAST_MODELS_DIR);
    const candidates: string[] = [];
    for (const e of entries) {
      const d = path.join(PAST_MODELS_DIR, e);
      const stat = await fse.stat(d).catch(() => null);
      if (stat && stat.isDirectory()) {
        const m = resolveModelJson(d);
        if (m) candidates.push(m);
      }
    }
    return pickRandom(candidates);
  } catch {
    return null;
  }
}

function createWorker(workerId: number, prodModelPath: string, opponentModelPath: string | null) {
  const worker = new Worker(WORKER_PATH, {
    workerData: {
      workerId,
      prodModelPath,
      opponentModelPath,
      autostart: true,
      boardSize: BOARD_SIZE,
      mctsSimLimit: MCTS_THINK_TIME_MS,
      explorationMoves: EXPLORATION_MOVES,
    },
  });
  return worker;
}

async function main() {
  if (isRunning) return;
  isRunning = true;
  console.log('=== Self-Play Trainer Starting ===');
  await updateStatus({ selfPlay: { workers: NUM_WORKERS }, phase: 'self_play_start' });

  await ensureDirs();
  let prodModelJson = resolveModelJson(PROD_MODEL_DIR);
  const isValidTfjsLayersModel = (p: string | null): boolean => {
    try {
      if (!p) return false;
      const raw = fs.readFileSync(p, 'utf-8');
      const j = JSON.parse(raw);
      // tfjs-layers saved model should include modelTopology and weightsManifest
      if (j && j.modelTopology && j.weightsManifest) return true;
      // Some variants include format marker
      if (typeof j?.format === 'string' && j.format.toLowerCase().includes('layers')) return true;
    } catch {}
    return false;
  };

  if (!isValidTfjsLayersModel(prodModelJson)) {
    console.log(`[Bootstrap] Missing or invalid model under ${PROD_MODEL_DIR}. Creating an initial model...`);
    await fse.ensureDir(PROD_MODEL_DIR);
    const model = createDualResNetModel();
    // Warm-up and save
    tf.tidy(() => {
      const size = Number(process.env.BOARD_SIZE || 15);
      const dummy = tf.zeros([1, size, size, 3]);
      const out = model.predict(dummy as TFT.Tensor) as TFT.Tensor | TFT.Tensor[];
      void out;
    });
    await model.save(`file://${PROD_MODEL_DIR}`);
    prodModelJson = resolveModelJson(PROD_MODEL_DIR);
    if (!isValidTfjsLayersModel(prodModelJson)) {
      throw new Error(`Failed to bootstrap a valid TFJS layers model at ${PROD_MODEL_DIR}`);
    }
    console.log(`[Bootstrap] Initial model saved to ${prodModelJson!}`);
  }

  const buffer: string[] = [];
  let totalSampleCount = 0;
  let workerFailures = 0;
  let fileCounter = 0;

  // Flush buffer to a new JSONL file
  async function flush(now = new Date()) {
    if (buffer.length === 0) return;
    const name = `${now.toISOString().replace(/[:.]/g, '-')}_${nanoid()}_${fileCounter++}.jsonl`;
    const dest = path.join(OUTPUT_DIR, name);
    const data = buffer.splice(0, buffer.length).join('\n') + '\n';
    await fse.writeFile(dest, data, 'utf-8');
    console.log(`Saved ${dest}`);
    await updateStatus({ selfPlay: { samplesTotal: totalSampleCount, lastSave: now.toISOString() } });
  }

  const workers: Worker[] = [];
  const endAt = Date.now() + SELF_PLAY_DURATION_MS;

  for (let i = 0; i < NUM_WORKERS; i++) {
    const usePast = Math.random() < PAST_MODEL_PROBABILITY;
    const opp = usePast ? await pickOpponentModelPath() : null;
    const w = createWorker(i, prodModelJson!, opp);

    w.on('message', (msg: any) => {
      if (msg?.trainingSamples) {
        const samples = msg.trainingSamples as TrainingSample[];
        for (const s of samples) buffer.push(JSON.stringify(s));
        totalSampleCount += samples.length;
        // Start next game unless duration exceeded
        if (Date.now() < endAt) {
          setImmediate(() => w.postMessage('start_new_game'));
        }
      } else if (msg?.reloaded) {
        console.log(`[Worker ${i}] Models reloaded:`, msg.reloaded);
      }
    });

    w.on('error', (err) => {
      console.error(`[Worker ${i}] error:`, err);
    });
    w.on('exit', (code) => {
      console.log(`[Worker ${i}] exited with code ${code}`);
      if (typeof code === 'number' && code !== 0) workerFailures++;
    });

    workers.push(w);
  }

  const saveInterval = setInterval(() => void flush(), SAVE_INTERVAL_MS);

  // Stop after duration
  const remain = Math.max(0, endAt - Date.now());
  await new Promise<void>((resolve) => setTimeout(resolve, remain));

  // Signal no more restarts and shutdown workers after final game completes
  console.log('Stopping workers...');
  for (const w of workers) w.terminate().catch(() => {});
  clearInterval(saveInterval);
  await flush();

  console.log('=== Self-Play Trainer Finished ===');
  if (totalSampleCount === 0) {
    console.error('No self-play samples were generated. Failing this cycle to avoid empty training.');
    process.exitCode = 2;
    throw new Error('Self-play produced zero samples');
  }
  await updateStatus({ phase: 'self_play_done', selfPlay: { samplesTotal: totalSampleCount } });
  isRunning = false;
}

main().catch((e) => {
  console.error('Self-play trainer failed:', e);
  process.exitCode = 1;
});


