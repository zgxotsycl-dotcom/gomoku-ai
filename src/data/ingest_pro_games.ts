import * as fs from 'fs';
import * as fsp from 'fs/promises';
import * as path from 'path';

import type { Player } from '../ai';

interface RawMove { r: number; c: number; player?: Player; }
interface RawGameJson {
  board_size?: number;
  moves: Array<RawMove | [number, number] | [number, number, Player]>;
  winner?: Player | 'draw' | null;
  first_player?: Player;
}

interface TrainingSample {
  state: (Player | null)[][];
  player: Player;
  mcts_policy: number[];
  teacher_policy: number[];
  teacher_value: number;
  final_value: -1 | 0 | 1;
}

const BOARD_SIZE = Number(process.env.BOARD_SIZE || 15);
const INPUT_DIR = process.env.PRO_GAME_DIR || path.resolve(process.cwd(), 'data', 'pro_games');
const OUTPUT_DIR = process.env.PRO_GAME_OUTPUT || path.resolve(process.cwd(), 'replay_buffer');
const DEFAULT_FIRST_PLAYER: Player = (process.env.PRO_FIRST_PLAYER || 'black') === 'white' ? 'white' : 'black';

function parseArgs(): { input: string; output: string } {
  const args = process.argv.slice(2);
  let input = INPUT_DIR;
  let output = OUTPUT_DIR;
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if ((arg === '--input' || arg === '-i') && args[i + 1]) {
      input = path.resolve(args[i + 1]);
      i++;
      continue;
    }
    if ((arg === '--output' || arg === '-o') && args[i + 1]) {
      output = path.resolve(args[i + 1]);
      i++;
      continue;
    }
  }
  return { input, output };
}

function normalizeMove(move: RawMove | [number, number] | [number, number, Player], turnIndex: number, startPlayer: Player): RawMove {
  if (Array.isArray(move)) {
    if (move.length === 3) {
      return { r: move[0], c: move[1], player: move[2] as Player };
    }
    return { r: move[0], c: move[1], player: turnIndex % 2 === 0 ? startPlayer : (startPlayer === 'black' ? 'white' : 'black') };
  }
  if (!move.player) {
    move.player = turnIndex % 2 === 0 ? startPlayer : (startPlayer === 'black' ? 'white' : 'black');
  }
  return move;
}

function createEmptyBoard(size: number): (Player | null)[][] {
  return Array.from({ length: size }, () => Array<Player | null>(size).fill(null));
}

function cloneBoard(board: (Player | null)[][]): (Player | null)[][] {
  return board.map((row) => row.slice()) as (Player | null)[][];
}

function applyMove(board: (Player | null)[][], move: RawMove): void {
  if (board[move.r] && board[move.r][move.c] === null) {
    board[move.r][move.c] = move.player ?? 'black';
  }
}

function policyFromMove(size: number, r: number, c: number): number[] {
  const flat = size * size;
  const policy = new Array<number>(flat).fill(0);
  if (r >= 0 && c >= 0 && r < size && c < size) {
    policy[r * size + c] = 1;
  }
  return policy;
}

function valueFromWinner(winner: Player | 'draw' | null | undefined, player: Player): number {
  if (!winner || winner === 'draw') return 0;
  return winner === player ? 1 : -1;
}

async function readJsonFile(fullPath: string): Promise<RawGameJson[]> {
  const text = await fsp.readFile(fullPath, 'utf-8');
  const data = JSON.parse(text);
  if (Array.isArray(data)) return data as RawGameJson[];
  return [data as RawGameJson];
}

async function readJsonlFile(fullPath: string): Promise<RawGameJson[]> {
  const lines = await fsp.readFile(fullPath, 'utf-8');
  const games: RawGameJson[] = [];
  for (const line of lines.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      const parsed = JSON.parse(line);
      games.push(parsed as RawGameJson);
    } catch (err) {
      console.warn(`[Skip] Failed to parse line in ${fullPath}:`, err);
    }
  }
  return games;
}

async function readCsvFile(fullPath: string): Promise<RawGameJson[]> {
  const text = await fsp.readFile(fullPath, 'utf-8');
  const [header, ...rows] = text.split(/\r?\n/).filter(Boolean);
  const columns = header.split(',');
  const games: RawGameJson[] = [];
  for (const row of rows) {
    if (!row.trim()) continue;
    const cells = row.split(',');
    const record: Record<string, string> = {};
    columns.forEach((col, idx) => {
      record[col.trim()] = cells[idx]?.trim?.() ?? '';
    });
    const movesRaw = record['moves'] || record['sequence'];
    if (!movesRaw) continue;
    const moves: RawGameJson['moves'] = [];
    for (const token of movesRaw.split(';')) {
      const parts = token.split(/\s+/).filter(Boolean);
      if (parts.length >= 2) {
        const r = Number(parts[0]);
        const c = Number(parts[1]);
        const player = parts[2] === 'white' ? 'white' : parts[2] === 'black' ? 'black' : undefined;
        moves.push(player ? { r, c, player } : { r, c });
      }
    }
    const winner = record['winner'] === 'white' ? 'white' : record['winner'] === 'black' ? 'black' : record['winner'] === 'draw' ? 'draw' : undefined;
    const boardSize = Number(record['board_size'] || record['size'] || BOARD_SIZE);
    games.push({ moves, winner, board_size: boardSize });
  }
  return games;
}

async function loadGamesFromFile(fullPath: string): Promise<RawGameJson[]> {
  const ext = path.extname(fullPath).toLowerCase();
  if (ext === '.jsonl') return readJsonlFile(fullPath);
  if (ext === '.json') return readJsonFile(fullPath);
  if (ext === '.csv') return readCsvFile(fullPath);
  throw new Error(`Unsupported file extension: ${ext}`);
}

async function collectGameFiles(dir: string): Promise<string[]> {
  const entries = await fsp.readdir(dir, { withFileTypes: true }).catch(() => [] as fs.Dirent[]);
  const files: string[] = [];
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      const nested = await collectGameFiles(full);
      files.push(...nested);
    } else if (/\.(jsonl|json|csv)$/i.test(entry.name)) {
      files.push(full);
    }
  }
  return files;
}

async function ensureDir(dir: string): Promise<void> {
  await fsp.mkdir(dir, { recursive: true });
}

function buildSamples(game: RawGameJson, defaultBoardSize: number): TrainingSample[] {
  const boardSize = game.board_size || defaultBoardSize;
  const board = createEmptyBoard(boardSize);
  const moves = game.moves || [];
  const samples: TrainingSample[] = [];
  const winner = game.winner ?? null;
  const firstPlayer = game.first_player || DEFAULT_FIRST_PLAYER;

  moves.forEach((rawMove, idx) => {
    const move = normalizeMove(rawMove, idx, firstPlayer);
    const stateBefore = cloneBoard(board);
    applyMove(board, move);
    const policy = policyFromMove(boardSize, move.r, move.c);
    const teacherValue = valueFromWinner(winner, move.player!);
    const finalValue = teacherValue as -1 | 0 | 1;
    samples.push({
      state: stateBefore,
      player: move.player!,
      mcts_policy: policy,
      teacher_policy: policy,
      teacher_value: teacherValue,
      final_value: finalValue,
    });
  });

  return samples;
}

async function main() {
  const { input, output } = parseArgs();
  console.log(`[Ingest] Reading games from ${input}`);
  const files = await collectGameFiles(input);
  if (files.length === 0) {
    console.warn('[Ingest] No supported files found (json/jsonl/csv).');
    return;
  }
  await ensureDir(output);
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const outPath = path.join(output, `${timestamp}_human.jsonl`);
  const writeStream = fs.createWriteStream(outPath, { encoding: 'utf-8' });

  let totalGames = 0;
  let totalSamples = 0;
  for (const file of files) {
    try {
      const games = await loadGamesFromFile(file);
      for (const game of games) {
        const samples = buildSamples(game, BOARD_SIZE);
        for (const sample of samples) {
          writeStream.write(`${JSON.stringify(sample)}\n`);
          totalSamples++;
        }
        totalGames++;
      }
    } catch (err) {
      console.warn(`[Ingest] Failed to process ${file}:`, err);
    }
  }

  writeStream.end();
  await new Promise<void>((resolve) => writeStream.on('close', () => resolve()));
  console.log(`[Ingest] Wrote ${totalSamples} samples from ${totalGames} games to ${outPath}`);
}

main().catch((err) => {
  console.error('[Ingest] Fatal error:', err);
  process.exitCode = 1;
});

