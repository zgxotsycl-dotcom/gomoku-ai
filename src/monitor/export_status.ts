import * as fs from 'fs';
import * as path from 'path';

interface StatusPayload {
  ts?: string;
  phase?: string;
  cycle?: number;
  selfPlay?: Record<string, unknown>;
  distill?: Record<string, unknown>;
  arena?: Record<string, unknown>;
  upload?: Record<string, unknown>;
  error?: string;
}

const STATUS_PATH = process.env.STATUS_PATH || path.resolve(process.cwd(), 'logs', 'status.json');
const OUTPUT_FORMAT = (process.env.EXPORT_FORMAT || 'prometheus').toLowerCase();

function readStatus(): StatusPayload | null {
  try {
    const raw = fs.readFileSync(STATUS_PATH, 'utf-8');
    return JSON.parse(raw) as StatusPayload;
  } catch (err) {
    console.error(`[Export] Failed to read ${STATUS_PATH}:`, err);
    return null;
  }
}

function printPrometheus(status: StatusPayload): void {
  const phase = status.phase ?? 'unknown';
  const cycle = status.cycle ?? 0;
  const errorFlag = status.error ? 1 : 0;
  console.log(`# TYPE gomoku_phase gauge`);
  console.log(`gomoku_phase{phase="${phase}"} 1`);
  console.log(`# TYPE gomoku_cycle counter`);
  console.log(`gomoku_cycle ${cycle}`);
  console.log(`# TYPE gomoku_error gauge`);
  console.log(`gomoku_error ${errorFlag}`);

  const toNumber = (value: unknown): number => typeof value === 'number' && Number.isFinite(value) ? value : NaN;

  if (status.selfPlay) {
    const workers = toNumber(status.selfPlay['workers']);
    const samples = toNumber(status.selfPlay['samplesTotal']);
    if (!Number.isNaN(workers)) console.log(`gomoku_selfplay_workers ${workers}`);
    if (!Number.isNaN(samples)) console.log(`gomoku_selfplay_samples_total ${samples}`);
  }
  if (status.distill) {
    const epoch = toNumber(status.distill['epoch']);
    const epochs = toNumber(status.distill['epochs']);
    if (!Number.isNaN(epoch)) console.log(`gomoku_distill_epoch ${epoch}`);
    if (!Number.isNaN(epochs)) console.log(`gomoku_distill_epochs ${epochs}`);
  }
  if (status.arena) {
    const played = toNumber(status.arena['played']);
    const candWins = toNumber(status.arena['candWins']);
    const prodWins = toNumber(status.arena['prodWins']);
    if (!Number.isNaN(played)) console.log(`gomoku_arena_played ${played}`);
    if (!Number.isNaN(candWins)) console.log(`gomoku_arena_candidate_wins ${candWins}`);
    if (!Number.isNaN(prodWins)) console.log(`gomoku_arena_prod_wins ${prodWins}`);
  }
}

function printJson(status: StatusPayload): void {
  console.log(JSON.stringify({ ts: new Date().toISOString(), status }));
}

function main(): void {
  const status = readStatus();
  if (!status) {
    process.exitCode = 1;
    return;
  }
  if (OUTPUT_FORMAT === 'jsonl' || OUTPUT_FORMAT === 'json') {
    printJson(status);
  } else {
    printPrometheus(status);
  }
}

main();
