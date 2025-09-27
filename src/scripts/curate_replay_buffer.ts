import * as fs from "fs";
import * as fsp from "fs/promises";
import * as path from "path";

const INPUT_DIR = process.env.CURATE_INPUT_DIR || path.resolve(process.cwd(), 'replay_buffer');
const OUTPUT_PATH = process.env.CURATE_OUTPUT || path.resolve(process.cwd(), 'replay_buffer', 'curated.jsonl');
const MAX_SAMPLES = Number(process.env.CURATE_MAX_SAMPLES || 0);

interface Sample {
  state: (string | null)[][] | (number | null)[][] | any;
  player: string;
  mcts_policy: number[];
  teacher_policy: number[];
  teacher_value: number;
  final_value: number;
}

async function collectJsonl(dir: string): Promise<string[]> {
  const entries = await fsp.readdir(dir, { withFileTypes: true }).catch(() => []);
  const files: string[] = [];
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      const nested = await collectJsonl(full);
      files.push(...nested);
    } else if (entry.isFile() && entry.name.toLowerCase().endsWith('.jsonl')) {
      files.push(full);
    }
  }
  return files.sort();
}

function stateKey(state: any, player: string): string {
  if (!Array.isArray(state)) return JSON.stringify({ state, player });
  const rows = state.map((row: any) => Array.isArray(row) ? row.map((cell) => (cell === null ? '-' : typeof cell === 'string' ? cell[0] : cell)).join('') : String(row)).join('|');
  return `${rows}|${player}`;
}

async function main(): Promise<void> {
  console.log(`[Curate] Source: ${INPUT_DIR}`);
  const files = await collectJsonl(INPUT_DIR);
  if (files.length === 0) {
    console.warn('[Curate] No JSONL files found.');
    return;
  }
  await fsp.mkdir(path.dirname(OUTPUT_PATH), { recursive: true });
  const outStream = fs.createWriteStream(OUTPUT_PATH, { encoding: 'utf-8' });
  const seen = new Set<string>();
  let kept = 0;
  let total = 0;

  for (const file of files) {
    console.log(`[Curate] Processing ${file}`);
    const data = await fsp.readFile(file, 'utf-8');
    for (const line of data.split(/\r?\n/)) {
      if (!line) continue;
      total += 1;
      try {
        const sample = JSON.parse(line) as Sample;
        const key = stateKey(sample.state, sample.player);
        if (seen.has(key)) continue;
        seen.add(key);
        outStream.write(`${JSON.stringify(sample)}\n`);
        kept += 1;
        if (MAX_SAMPLES > 0 && kept >= MAX_SAMPLES) {
          console.log('[Curate] Reached max samples limit.');
          outStream.end();
          console.log(`[Curate] Kept ${kept} of ${total} processed entries.`);
          return;
        }
      } catch (err) {
        console.warn(`[Curate] Failed to parse line in ${file}:`, err);
      }
    }
  }

  outStream.end();
  console.log(`[Curate] Kept ${kept} of ${total} processed entries. Output: ${OUTPUT_PATH}`);
}

main().catch((err) => {
  console.error('[Curate] Fatal error:', err);
  process.exitCode = 1;
});
