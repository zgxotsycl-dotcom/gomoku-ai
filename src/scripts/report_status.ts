import * as fs from "fs";
import * as path from "path";

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

interface ArenaResult {
  ts?: string;
  games?: number;
  candidate_wins?: number;
  prod_wins?: number;
  draws?: number;
  winrate?: number;
  promoted?: boolean;
}

const STATUS_PATH = process.env.STATUS_PATH || path.resolve(process.cwd(), 'logs', 'status.json');
const ARENA_RESULT_PATH = process.env.ARENA_RESULT_PATH || path.resolve(process.cwd(), 'arena_result.json');

function loadJson<T>(file: string): T | null {
  if (!fs.existsSync(file)) return null;
  try {
    const raw = fs.readFileSync(file, 'utf-8');
    return JSON.parse(raw) as T;
  } catch (err) {
    console.warn(`[Report] Failed to read ${file}:`, err);
    return null;
  }
}

function percent(v: number | undefined): string {
  if (typeof v !== 'number' || !Number.isFinite(v)) return '-';
  return (v * 100).toFixed(1) + '%';
}

function main(): void {
  const status = loadJson<StatusPayload>(STATUS_PATH);
  if (!status) {
    console.error('[Report] status.json not found.');
    process.exitCode = 1;
    return;
  }
  const arena = loadJson<ArenaResult>(ARENA_RESULT_PATH);

  console.log('=== Pipeline Status ===');
  console.log(`Timestamp  : ${status.ts ?? '-'}`);
  console.log(`Phase      : ${status.phase ?? '-'}`);
  console.log(`Cycle      : ${status.cycle ?? '-'}`);
  if (status.error) console.log(`Error      : ${status.error}`);

  const workers = status.selfPlay?.['workers'];
  const samples = status.selfPlay?.['samplesTotal'];
  console.log('\n-- Self-Play --');
  console.log(`Workers    : ${workers ?? '-'}`);
  console.log(`Samples    : ${samples ?? '-'}`);

  console.log('\n-- Distillation --');
  console.log(`Epoch      : ${status.distill?.['epoch'] ?? '-'}/${status.distill?.['epochs'] ?? '-'}`);
  if (status.distill?.['lastSaved']) {
    console.log(`Last saved : ${status.distill['lastSaved']}`);
  }

  console.log('\n-- Arena --');
  console.log(`Played     : ${status.arena?.['played'] ?? '-'}/${status.arena?.['total'] ?? '-'}`);
  console.log(`Wins (cand/prod): ${status.arena?.['candWins'] ?? '-'} / ${status.arena?.['prodWins'] ?? '-'}`);
  console.log(`Draws      : ${status.arena?.['draws'] ?? '-'}`);
  console.log(`Promoted   : ${status.arena?.['promoted'] ?? '-'}`);

  if (arena) {
    console.log('\nLast Arena Result');
    console.log(`Time       : ${arena.ts ?? '-'}`);
    console.log(`Games      : ${arena.games ?? '-'}`);
    console.log(`Winrate    : ${percent(arena.winrate)}`);
    console.log(`Promoted   : ${arena.promoted ?? '-'}`);
  }

  if (status.upload?.['publicUrl']) {
    console.log('\n-- Upload --');
    console.log(`Model URL  : ${status.upload['publicUrl']}`);
  }
}

main();
