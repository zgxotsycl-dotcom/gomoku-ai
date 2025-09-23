// Use unified TF loader that prefers GPU (@tensorflow/tfjs-node-gpu) when available.
import tf from './tf';
import * as fs from 'fs/promises';
import * as fss from 'fs';
import * as path from 'path';
import * as fse from 'fs-extra';
import * as readline from 'node:readline';
import { updateStatus } from './status';

type Player = 'black' | 'white';

type TrainingSample = {
  state: (Player | null)[][];
  player: Player;
  mcts_policy: number[];
  teacher_policy: number[];
  teacher_value: number;
  final_value: -1 | 0 | 1;
};

// ---------- Config ----------
const BASE_DIR = process.env.APP_DIR || path.resolve(__dirname, '..');
const REPLAY_BUFFER_DIR = process.env.REPLAY_BUFFER_DIR || path.resolve(BASE_DIR, 'replay_buffer');
const SAVE_DIR = process.env.SAVE_DIR || path.resolve(BASE_DIR, 'gomoku_model_prod');
const CANDIDATE_SAVE_DIR = process.env.CANDIDATE_SAVE_DIR || path.resolve(BASE_DIR, 'gomoku_model_candidate');
const GATING_ENABLED = (process.env.GATING_ENABLED || 'true').toLowerCase() === 'true';
const PAST_MODELS_DIR = process.env.PAST_MODELS_DIR || path.resolve(BASE_DIR, 'past_models');
const SAVE_PAST_MODEL = (process.env.SAVE_PAST_MODEL || 'true').toLowerCase() === 'true';
const BATCH_SIZE = Number(process.env.BATCH_SIZE || 64);
const EPOCHS = Number(process.env.EPOCHS || 4);
const STEPS_PER_EPOCH = Number(process.env.STEPS_PER_EPOCH || 4000);
const LEARNING_RATE = Number(process.env.LEARNING_RATE || 5e-4);
const TEACHER_TEMP = Number(process.env.TEACHER_TEMP || 1.5);
const ALPHA_TEACHER_POLICY = Number(process.env.ALPHA_TEACHER_POLICY || 0.7); // vs MCTS policy
const BETA_TEACHER_VALUE = Number(process.env.BETA_TEACHER_VALUE || 0.7);       // vs final value
const SHUFFLE_FILES = (process.env.SHUFFLE_FILES || 'true').toLowerCase() === 'true';

// ---------- Utils ----------
function getBoardSize(board: (Player | null)[][]): number { return board.length; }

function toInput(board: (Player | null)[][], player: Player): any {
  const size = getBoardSize(board);
  const opp = player === 'black' ? 'white' : 'black';
  const playerCh = new Float32Array(size * size);
  const oppCh = new Float32Array(size * size);
  const sideCh = new Float32Array(size * size);
  let k = 0;
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const v = board[r][c];
      playerCh[k] = v === player ? 1 : 0;
      oppCh[k] = v === opp ? 1 : 0;
      sideCh[k] = player === 'black' ? 1 : 0;
      k++;
    }
  }
  const stacked = tf.stack([
    tf.tensor2d(playerCh, [size, size]),
    tf.tensor2d(oppCh, [size, size]),
    tf.tensor2d(sideCh, [size, size])
  ], 2); // [H,W,3]
  return stacked;
}

function applyTemperature(dist: number[], temp: number): number[] {
  if (temp === 1) return dist.slice();
  const out = new Array(dist.length);
  let sum = 0;
  for (let i = 0; i < dist.length; i++) {
    const v = Math.max(1e-20, dist[i]);
    const s = Math.exp(Math.log(v) / temp);
    out[i] = s; sum += s;
  }
  if (sum <= 0) {
    const u = 1 / out.length; return out.map(() => u);
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

function mixPolicy(teacher: number[], mcts: number[], alpha: number, temp: number): number[] {
  const t = applyTemperature(teacher, temp);
  const out = new Array(t.length);
  for (let i = 0; i < t.length; i++) out[i] = alpha * t[i] + (1 - alpha) * (mcts[i] ?? 0);
  // renorm
  let s = 0; for (let i = 0; i < out.length; i++) s += Math.max(0, out[i]);
  if (s <= 0) { const u = 1 / out.length; return out.map(() => u); }
  for (let i = 0; i < out.length; i++) out[i] = Math.max(0, out[i]) / s;
  return out;
}

// --- Symmetry augmentation ---
type TransformId = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7;
function transformRC(r: number, c: number, size: number, t: TransformId): [number, number] {
  const n = size;
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
function transformBoard(board: (Player | null)[][], t: TransformId): (Player | null)[][] {
  const n = board.length;
  const out: (Player | null)[][] = Array.from({ length: n }, () => Array<Player | null>(n).fill(null));
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) { const [rr, cc] = transformRC(r, c, n, t); out[rr][cc] = board[r][c]; }
  return out;
}
function transformPolicyFlat(pol: number[], size: number, t: TransformId): number[] {
  const out = new Array(pol.length).fill(0);
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) {
    const [rr, cc] = transformRC(r, c, size, t);
    out[rr * size + cc] = pol[r * size + c];
  }
  return out;
}

async function* readJsonlStream(fullPath: string): AsyncGenerator<TrainingSample> {
  const rl = readline.createInterface({ input: fss.createReadStream(fullPath), crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line) continue;
    try { yield JSON.parse(line) as TrainingSample; } catch {}
  }
  rl.close();
}

async function* sampleStream(files: string[]): AsyncGenerator<TrainingSample> {
  for (const f of files) {
    if (!f.endsWith('.jsonl')) continue;
    const full = path.join(REPLAY_BUFFER_DIR, f);
    try {
      for await (const s of readJsonlStream(full)) yield s;
    } catch {}
  }
}

async function listFiles(dir: string): Promise<string[]> {
  const prioritizeRecent = (process.env.PRIORITIZE_RECENT_FILES || 'true').toLowerCase() === 'true';
  const limit = Number(process.env.MAX_FILES_PER_EPOCH || 0); // 0 = no limit
  const names = await fs.readdir(dir).catch(() => [] as string[]);
  let files = names.filter((f) => f.endsWith('.jsonl'));
  if (prioritizeRecent) {
    const stats = await Promise.all(files.map(async (f) => ({ f, m: (await fs.stat(path.join(dir, f))).mtimeMs })));
    stats.sort((a, b) => b.m - a.m);
    files = stats.map((s) => s.f);
  }
  if (limit > 0 && files.length > limit) files = files.slice(0, limit);
  if (SHUFFLE_FILES) return files.sort(() => Math.random() - 0.5);
  return files;
}

async function train() {
  console.log('--- Distillation Training Start ---');
  await updateStatus({ phase: 'distill_start' });
  // Ensure we are using the native TensorFlow binding (tensorflow backend).
  // If not (e.g., pure JS backend), skip heavy training and fall back to copying.
  const backend = (tf as any)?.getBackend?.() as string | undefined;
  if (backend !== 'tensorflow') {
    console.warn(`[TF] Non-native backend detected (backend=${backend}). Skipping training and copying current prod model as candidate.`);
    try {
      await fs.rm(CANDIDATE_SAVE_DIR, { recursive: true, force: true });
      if (fss.existsSync(SAVE_DIR)) {
        await fse.copy(SAVE_DIR, CANDIDATE_SAVE_DIR, { overwrite: true });
        console.log(`[Fallback] Copied existing model to candidate at ${CANDIDATE_SAVE_DIR}`);
      } else {
        console.warn('[Fallback] No existing model to copy. Skipping.');
      }
    } catch (copyErr) {
      console.error('Fallback copy failed:', copyErr);
    }
    return;
  }
  if (!fss.existsSync(REPLAY_BUFFER_DIR)) {
    throw new Error(`REPLAY_BUFFER_DIR not found: ${REPLAY_BUFFER_DIR}`);
  }

  const { createDualResNetModel } = await import('./model');
  const model = createDualResNetModel();
  // Re-compile with custom optimizer and lossWeights for stronger policy emphasis
  model.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: { policy_head: 'categoricalCrossentropy', value_head: 'meanSquaredError' },
    metrics: { policy_head: 'accuracy', value_head: 'mae' },
  });

  const files = await listFiles(REPLAY_BUFFER_DIR);
  const boardSize = Number(process.env.BOARD_SIZE || 15);
  const flat = boardSize * boardSize;

  for (let epoch = 1; epoch <= EPOCHS; epoch++) {
    console.log(`\n[Epoch ${epoch}/${EPOCHS}]`);
    await updateStatus({ distill: { epoch, epochs: EPOCHS } });
    let step = 0;
    let totalSamples = 0;
    const inputs: any[] = [];
    const polTargets: number[][] = [];
    const valTargets: number[] = [];

    for await (const s of sampleStream(files)) {
      // Filter to unified 15x15 (BOARD_SIZE)
      if (!s?.state || s.state.length !== boardSize) continue;
      // Random symmetry augmentation
      const t: TransformId = (Math.floor(Math.random() * 8) as TransformId);
      const stateT = transformBoard(s.state, t);
      const inp = toInput(stateT, s.player);
      inputs.push(inp);
      let pol = mixPolicy(s.teacher_policy ?? s.mcts_policy, s.mcts_policy ?? s.teacher_policy, ALPHA_TEACHER_POLICY, TEACHER_TEMP);
      pol = transformPolicyFlat(pol, boardSize, t);
      // Label smoothing
      const epsilon = Number(process.env.LABEL_SMOOTHING || 0.05);
      if (epsilon > 0) {
        const u = 1 / flat;
        for (let i = 0; i < pol.length; i++) pol[i] = (1 - epsilon) * pol[i] + epsilon * u;
      }
      if (pol.length !== flat) { inp.dispose(); continue; }
      polTargets.push(pol);
      const val = BETA_TEACHER_VALUE * (s.teacher_value ?? 0) + (1 - BETA_TEACHER_VALUE) * (s.final_value ?? 0);
      valTargets.push(val);
      totalSamples++;

      if (inputs.length === BATCH_SIZE) {
        const x = tf.stack(inputs); // [B,H,W,3]
        const yPolicy = tf.tensor2d(polTargets, [BATCH_SIZE, flat]);
        const yValue = tf.tensor2d(valTargets.map((v) => [v]), [BATCH_SIZE, 1]);
        inputs.splice(0).forEach((t) => t.dispose());
        polTargets.splice(0); valTargets.splice(0);

        const h = await model.trainOnBatch(x, { policy_head: yPolicy, value_head: yValue }) as number[];
        x.dispose(); yPolicy.dispose(); yValue.dispose();
        step++;
        if (step % 50 === 0) console.log(`  step=${step} loss=${h?.[0]?.toFixed?.(4)} policyAcc=${h?.[3]?.toFixed?.(4)}`);
        if (step >= STEPS_PER_EPOCH) break;
      }
    }

    // Flush tail if any
    if (inputs.length > 0) {
      const b = inputs.length;
      const x = tf.stack(inputs);
      const yPolicy = tf.tensor2d(polTargets, [b, flat]);
      const yValue = tf.tensor2d(valTargets.map((v) => [v]), [b, 1]);
      inputs.splice(0).forEach((t) => t.dispose());
      polTargets.splice(0); valTargets.splice(0);
      await model.trainOnBatch(x, { policy_head: yPolicy, value_head: yValue });
      x.dispose(); yPolicy.dispose(); yValue.dispose();
    }

    // Save after each epoch (candidate if gating enabled)
    const targetDir = GATING_ENABLED ? CANDIDATE_SAVE_DIR : SAVE_DIR;
    await fs.rm(targetDir, { recursive: true, force: true });
    await model.save(`file://${targetDir}`);
    console.log(`[Epoch ${epoch}] Saved model to ${targetDir}`);

    if (SAVE_PAST_MODEL) {
      try {
        const stamp = new Date().toISOString().replace(/[:.]/g, '-');
        const dest = path.join(PAST_MODELS_DIR, `model_${stamp}`);
        await fse.ensureDir(PAST_MODELS_DIR);
        await fse.copy(SAVE_DIR, dest, { overwrite: true });
        console.log(`[Epoch ${epoch}] Copied snapshot to ${dest}`);
      } catch (e) {
        console.warn('Failed to copy to past_models:', e);
      }
    }
  }

  console.log('--- Distillation Training Finished ---');
  await updateStatus({ phase: 'distill_done', distill: { epoch: EPOCHS, epochs: EPOCHS, lastSaved: new Date().toISOString() } });
}

train().catch((e) => {
  console.error('Distillation failed:', e);
  process.exitCode = 1;
});
