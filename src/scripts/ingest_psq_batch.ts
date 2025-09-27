import { spawnSync } from "child_process";
import * as fs from "fs";
import * as path from "path";

interface BatchState {
  processed: Record<string, string>;
}

const ROOT_DIR = process.env.PSQ_BATCH_ROOT || path.resolve(process.cwd(), 'data', 'psq_batches');
const OUTPUT_DIR = process.env.PSQ_BATCH_OUTPUT || path.resolve(process.cwd(), 'replay_buffer');
const STATE_FILE = process.env.PSQ_BATCH_STATE || path.resolve(process.cwd(), '.psq_processed.json');

function loadState(): BatchState {
  if (!fs.existsSync(STATE_FILE)) {
    return { processed: {} };
  }
  try {
    const raw = fs.readFileSync(STATE_FILE, 'utf-8');
    const parsed = JSON.parse(raw) as BatchState;
    if (!parsed.processed) parsed.processed = {};
    return parsed;
  } catch {
    return { processed: {} };
  }
}

function saveState(state: BatchState): void {
  fs.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
}

function collectBatches(root: string): string[] {
  if (!fs.existsSync(root)) return [];
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => path.join(root, d.name))
    .sort();
}

function ingestDirectory(dir: string): boolean {
  console.log(`[PSQ-Batch] Ingesting ${dir}`);
  const result = spawnSync('npm', ['run', 'data:ingest:psq', '--', '--input', dir, '--output', OUTPUT_DIR], {
    stdio: 'inherit',
    shell: process.platform === 'win32',
  });
  if (result.status !== 0) {
    console.error(`[PSQ-Batch] Failed with code ${result.status ?? 'unknown'}`);
    return false;
  }
  return true;
}

function main(): void {
  console.log(`[PSQ-Batch] Root directory: ${ROOT_DIR}`);
  const dirs = collectBatches(ROOT_DIR);
  if (dirs.length === 0) {
    console.warn('[PSQ-Batch] No batch directories found.');
    return;
  }
  const state = loadState();
  let processedAny = false;
  for (const dir of dirs) {
    const key = path.relative(ROOT_DIR, dir);
    if (state.processed[key]) {
      console.log(`[PSQ-Batch] Skipping already processed directory: ${key}`);
      continue;
    }
    const ok = ingestDirectory(dir);
    if (ok) {
      state.processed[key] = new Date().toISOString();
      processedAny = true;
    } else {
      console.error(`[PSQ-Batch] Directory failed: ${key}. Stopping.`);
      break;
    }
  }
  if (processedAny) {
    saveState(state);
    console.log('[PSQ-Batch] State updated.');
  } else {
    console.log('[PSQ-Batch] Nothing new processed.');
  }
}

main();
