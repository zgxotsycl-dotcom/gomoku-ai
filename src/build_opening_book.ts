import * as fsp from 'fs/promises';
import * as path from 'path';
import { updateStatus } from './status';

const BOARD_SIZE = Number(process.env.BOARD_SIZE || 15);
const BASE_DIR = process.env.APP_DIR || path.resolve(__dirname, '..');
const OUTPUT_PATH = process.env.OUTPUT_PATH || path.resolve(BASE_DIR, 'opening_book_generated.json');
const NEIGHBOR_RADIUS = Math.max(1, Math.floor(Number(process.env.BOOK_NEIGHBOR_RADIUS || 2)));
const INCLUDE_DIAGONALS = (process.env.BOOK_INCLUDE_DIAGONALS || 'true').toLowerCase() === 'true';
const MAX_NEIGHBOR_COUNT = Number(process.env.BOOK_MAX_NEIGHBORS || 0); // 0 = no limit

function emptyHash(size: number): string {
  const row = '-'.repeat(size);
  return Array.from({ length: size }, () => row).join('|');
}

async function main() {
  const center = Math.floor(BOARD_SIZE / 2);
  const entries: Array<{ board_hash: string; best_move: [number, number] }> = [];
  // Minimal book: empty board -> center
  entries.push({ board_hash: emptyHash(BOARD_SIZE), best_move: [center, center] });
  // Optionally: add a few symmetric replies around center for move_count 1
  // Skipped for simplicity; server/edge will rely on search for others.

  await fsp.writeFile(OUTPUT_PATH, JSON.stringify(entries, null, 2), 'utf-8');
  console.log(`[Book] Wrote ${entries.length} entries to ${OUTPUT_PATH}`);
  try { await updateStatus({ book: { entries: entries.length } }); } catch {}
}

main().catch((e) => {
  console.error('Building opening book failed:', e);
  process.exitCode = 1;
});
