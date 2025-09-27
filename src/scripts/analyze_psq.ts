import * as fs from "fs";
import * as fsp from "fs/promises";
import * as path from "path";

interface HeaderMap { [key: string]: string; }

interface StatsEntry {
  games: number;
  blackWins: number;
  whiteWins: number;
  draws: number;
}

const ROOT_DIR = process.env.PSQ_ANALYZE_DIR || path.resolve(process.cwd(), 'data', 'psq_batches');
const PLAYER1_IS_BLACK = (process.env.PSQ_ANALYZE_PLAYER1_IS_BLACK || 'true').toLowerCase() === 'true';

function parsePsq(text: string): { header: HeaderMap; moves: Array<[number, number]> } {
  const header: HeaderMap = {};
  const moves: Array<[number, number]> = [];
  let section: 'header' | 'game' | '' = '';
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith('[') && line.endsWith(']')) {
      const s = line.slice(1, -1).toLowerCase();
      section = s === 'header' ? 'header' : s === 'game' ? 'game' : '';
      continue;
    }
    if (section === 'header') {
      const idx = line.indexOf('=');
      if (idx > 0) {
        const key = line.slice(0, idx).trim();
        const value = line.slice(idx + 1).trim();
        header[key] = value;
      }
      continue;
    }
    if (section === 'game') {
      const parts = line.split(',').map((p) => p.trim());
      if (parts.length < 2) continue;
      const x = Number(parts[0]);
      const y = Number(parts[1]);
      if (Number.isFinite(x) && Number.isFinite(y)) moves.push([x, y]);
    }
  }
  return { header, moves };
}

async function collectPsqFiles(dir: string): Promise<string[]> {
  const entries = await fsp.readdir(dir, { withFileTypes: true }).catch(() => []);
  const files: string[] = [];
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      const nested = await collectPsqFiles(full);
      files.push(...nested);
    } else if (entry.isFile() && entry.name.toLowerCase().endsWith('.psq')) {
      files.push(full);
    }
  }
  return files;
}

function coordKey(x: number, y: number): string {
  return `${x},${y}`;
}

function updateStats(map: Map<string, StatsEntry>, key: string, winner: 'black' | 'white' | 'draw'): void {
  const entry = map.get(key) || { games: 0, blackWins: 0, whiteWins: 0, draws: 0 };
  entry.games += 1;
  if (winner === 'black') entry.blackWins += 1;
  else if (winner === 'white') entry.whiteWins += 1;
  else entry.draws += 1;
  map.set(key, entry);
}

async function main(): Promise<void> {
  console.log(`[Analyze] Root: ${ROOT_DIR}`);
  if (!fs.existsSync(ROOT_DIR)) {
    console.error('[Analyze] Directory not found.');
    process.exitCode = 1;
    return;
  }
  const files = await collectPsqFiles(ROOT_DIR);
  if (files.length === 0) {
    console.warn('[Analyze] No .psq files located.');
    return;
  }
  const stats = new Map<string, StatsEntry>();
  for (const file of files) {
    try {
      const text = await fsp.readFile(file, 'utf-8');
      const { header, moves } = parsePsq(text);
      if (moves.length === 0) continue;
      const firstMove = moves[0];
      const key = coordKey(firstMove[0], firstMove[1]);
      const result = Number(header['Result'] || header['result'] || 0);
      let winner: 'black' | 'white' | 'draw' = 'draw';
      if (result === 1) winner = PLAYER1_IS_BLACK ? 'black' : 'white';
      else if (result === 2) winner = PLAYER1_IS_BLACK ? 'white' : 'black';
      updateStats(stats, key, winner);
    } catch (err) {
      console.warn(`[Analyze] Failed to parse ${file}:`, err);
    }
  }

  const sorted = Array.from(stats.entries()).sort((a, b) => {
    const lossRateA = a[1].whiteWins / Math.max(1, a[1].games);
    const lossRateB = b[1].whiteWins / Math.max(1, b[1].games);
    return lossRateB - lossRateA;
  });

  console.log('\n=== First-move loss hot spots ===');
  for (const [coord, entry] of sorted.slice(0, 20)) {
    const whiteLossRate = (entry.whiteWins / entry.games) * 100;
    const blackLossRate = (entry.blackWins / entry.games) * 100;
    console.log(`${coord} -> games: ${entry.games}, blackWin%: ${blackLossRate.toFixed(1)}, whiteWin%: ${whiteLossRate.toFixed(1)}, draws: ${entry.draws}`);
  }
}

main().catch((err) => {
  console.error('[Analyze] Fatal error:', err);
  process.exitCode = 1;
});
