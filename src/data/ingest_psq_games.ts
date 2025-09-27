import * as fs from 'fs';
import * as fsp from 'fs/promises';
import * as path from 'path';

import type { Player } from '../ai';
import type { TrainingSample } from '../types/training';

interface HeaderMap {
  [key: string]: string;
}


const DEFAULT_BOARD_SIZE = Number(process.env.PSQ_BOARD_SIZE || process.env.BOARD_SIZE || 15);
const DEFAULT_INPUT_DIR = process.env.PSQ_INPUT_DIR || path.resolve(process.cwd(), 'data', 'psq');
const DEFAULT_OUTPUT_DIR = process.env.PSQ_OUTPUT_DIR || path.resolve(process.cwd(), 'replay_buffer');
const PLAYER1_IS_BLACK = (process.env.PSQ_PLAYER1_IS_BLACK || 'true').toLowerCase() === 'true';

function parseArgs(): { input: string; output: string; boardSize: number; allowedRules: string[] } {
  const args = process.argv.slice(2);
  let input = DEFAULT_INPUT_DIR;
  let output = DEFAULT_OUTPUT_DIR;
  let boardSize = DEFAULT_BOARD_SIZE;
  let allowedRules = (process.env.PSQ_RULE_FILTER || '')
    .split(',')
    .map((rule) => rule.trim().toLowerCase())
    .filter(Boolean);
  for (let i = 0; i < args.length; i++) {
    const token = args[i];
    if ((token === '--input' || token === '-i') && args[i + 1]) {
      input = path.resolve(args[i + 1]);
      i++;
      continue;
    }
    if ((token === '--output' || token === '-o') && args[i + 1]) {
      output = path.resolve(args[i + 1]);
      i++;
      continue;
    }
    if ((token === '--board-size' || token === '-b') && args[i + 1]) {
      const parsed = Number(args[i + 1]);
      if (Number.isFinite(parsed) && parsed > 0) {
        boardSize = parsed;
      }
      i++;
      continue;
    }
    if ((token === '--rules' || token === '--allow-rules') && args[i + 1]) {
      allowedRules = args[i + 1]
        .split(',')
        .map((rule) => rule.trim().toLowerCase())
        .filter(Boolean);
      i++;
      continue;
    }
  }
  return { input, output, boardSize, allowedRules };
}

async function collectPsqFiles(root: string): Promise<string[]> {
  const entries = await fsp.readdir(root, { withFileTypes: true }).catch(() => []);
  const files: string[] = [];
  for (const entry of entries) {
    const full = path.join(root, entry.name);
    if (entry.isDirectory()) {
      const nested = await collectPsqFiles(full);
      files.push(...nested);
    } else if (entry.isFile() && entry.name.toLowerCase().endsWith('.psq')) {
      files.push(full);
    }
  }
  return files;
}



function parsePsq(content: string): { header: HeaderMap; moves: Array<[number, number]> } {
  const lines = content.split(/\r?\n/);
  const trimmedLines = lines.map((line) => line.trim()).filter(Boolean);
  const hasSectionMarkers = trimmedLines.some((line) => {
    const lower = line.toLowerCase();
    return lower === '[header]' || lower === '[game]';
  });

  if (hasSectionMarkers) {
    const parsed = parseSectionedPsq(lines);
    if (parsed.moves.length > 0 || Object.keys(parsed.header).length > 0) {
      return parsed;
    }
  }

  return parseGomocupPsq(lines);
}

function parseSectionedPsq(lines: string[]): { header: HeaderMap; moves: Array<[number, number]> } {
  const header: HeaderMap = {};
  const moves: Array<[number, number]> = [];
  let section: 'header' | 'game' | '' = '';
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith('[') && line.endsWith(']')) {
      const sectionName = line.slice(1, -1).toLowerCase();
      if (sectionName === 'header') section = 'header';
      else if (sectionName === 'game') section = 'game';
      else section = '';
      continue;
    }
    if (section === 'header') {
      const idx = line.indexOf('=');
      if (idx <= 0) continue;
      const key = line.slice(0, idx).trim();
      const value = line.slice(idx + 1).trim();
      header[key] = value;
      continue;
    }
    if (section === 'game') {
      const parts = line.split(',').map((p) => p.trim());
      if (parts.length < 2) continue;
      const x = Number(parts[0]);
      const y = Number(parts[1]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      moves.push([x, y]);
    }
  }
  return { header, moves };
}

function parseGomocupPsq(lines: string[]): { header: HeaderMap; moves: Array<[number, number]> } {
  const header: HeaderMap = {};
  const moves: Array<[number, number]> = [];
  if (lines.length === 0) return { header, moves };

  const firstLine = lines[0].trim();
  if (firstLine) {
    const sizeMatch = firstLine.match(/(\d+)\s*x\s*(\d+)/i);
    if (sizeMatch) {
      const width = Number(sizeMatch[1]);
      const height = Number(sizeMatch[2]);
      if (Number.isFinite(width)) {
        if (Number.isFinite(height) && width === height) {
          header['BoardSize'] = String(width);
        } else {
          header['BoardSize'] = String(width);
          if (Number.isFinite(height)) {
            header['BoardHeight'] = String(height);
          }
        }
      }
    }
  }

  for (const raw of lines.slice(1)) {
    const line = raw.trim();
    if (!line) continue;

    const parts = line.split(',').map((p) => p.trim());
    if (parts.length >= 2) {
      const x = Number(parts[0]);
      const y = Number(parts[1]);
      if (Number.isFinite(x) && Number.isFinite(y)) {
        moves.push([x, y]);
        continue;
      }
    }

    if (/\.zip$/i.test(line)) {
      const name = line.replace(/\.zip$/i, '');
      if (!header['Player1']) header['Player1'] = name;
      else if (!header['Player2']) header['Player2'] = name;
      continue;
    }

    if (parts.length >= 2) {
      const winner = Number(parts[0]);
      const secondValue = Number(parts[1]);
      if ((winner === 0 || winner === 1 || winner === 2) && !Number.isFinite(secondValue)) {
        header['Result'] = String(winner);
        if (parts[1]) {
          header['Rule'] = parts[1];
          header['rule'] = parts[1];
        }
        continue;
      }
    }

    if (/^-?\d+$/.test(line) && !header['Result']) {
      const numericResult = Number(line);
      if (numericResult === 0 || numericResult === 1 || numericResult === 2) {
        header['Result'] = String(numericResult);
      }
    }
  }

  return { header, moves };
}

function createBoard(size: number): (Player | null)[][] {
  return Array.from({ length: size }, () => Array<Player | null>(size).fill(null));
}

function cloneBoard(board: (Player | null)[][]): (Player | null)[][] {
  return board.map((row) => row.slice()) as (Player | null)[][];
}

function policyFromMove(size: number, r: number, c: number): number[] {
  const flat = size * size;
  const policy = new Array<number>(flat).fill(0);
  if (r >= 0 && r < size && c >= 0 && c < size) {
    policy[r * size + c] = 1;
  }
  return policy;
}

function outcomeForPlayer(winner: Player | null, player: Player): -1 | 0 | 1 {
  if (!winner) return 0;
  return winner === player ? 1 : -1;
}

async function writeJsonl(outputDir: string, samples: TrainingSample[]): Promise<string> {
  await fsp.mkdir(outputDir, { recursive: true });
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const outPath = path.join(outputDir, `${timestamp}_psq.jsonl`);
  const stream = fs.createWriteStream(outPath, { encoding: 'utf-8' });
  for (const sample of samples) {
    stream.write(`${JSON.stringify(sample)}\n`);
  }
  await new Promise<void>((resolve, reject) => {
    stream.end();
    stream.on('close', resolve);
    stream.on('error', reject);
  });
  return outPath;
}

async function main(): Promise<void> {
  const { input, output, boardSize: targetBoardSize, allowedRules } = parseArgs();
  console.log(`[PSQ] Reading from ${input}`);
  const files = await collectPsqFiles(input);
  if (files.length === 0) {
    console.warn('[PSQ] No .psq files found.');
    return;
  }

  const allowedRuleSet = new Set(allowedRules);
  const allSamples: TrainingSample[] = [];
  for (const file of files) {
    try {
      const text = await fsp.readFile(file, 'utf-8');
      const { header, moves } = parsePsq(text);
      if (moves.length === 0) continue;

      const rawBoardSize = Number(header['BoardSize'] || header['board_size'] || DEFAULT_BOARD_SIZE);
      const boardSize = Number.isFinite(rawBoardSize) && rawBoardSize > 0 ? rawBoardSize : DEFAULT_BOARD_SIZE;
      if (boardSize !== targetBoardSize) {
        console.warn(`[PSQ] Skipping ${file}: board size ${boardSize} != target ${targetBoardSize}.`);
        continue;
      }

      const ruleLabel = String(header['Rule'] || header['rule'] || '').trim();
      const normalizedRule = ruleLabel.toLowerCase();
      if (allowedRuleSet.size > 0 && (!normalizedRule || !allowedRuleSet.has(normalizedRule))) {
        console.warn(`[PSQ] Skipping ${file}: rule '${ruleLabel || 'unknown'}' not allowed.`);
        continue;
      }

      const result = Number(header['Result'] || header['result'] || 0);

      const player1IsBlack = PLAYER1_IS_BLACK;

      let winner: Player | null = null;
      if (result === 1) winner = player1IsBlack ? 'black' : 'white';
      else if (result === 2) winner = player1IsBlack ? 'white' : 'black';

      const board = createBoard(boardSize);
      let current: Player = 'black';
      const fileSamples: TrainingSample[] = [];
      let invalidMove = false;
      const relativePath = path.isAbsolute(file) ? path.relative(input, file) : file;
      const gameId = path.basename(file, path.extname(file));
      const baseTags = ['psq'];
      if (normalizedRule) baseTags.push(`rule:${normalizedRule}`);

      for (const [rawX, rawY] of moves) {
        const col = rawX - 1;
        const row = rawY - 1;
        if (row < 0 || col < 0 || row >= boardSize || col >= boardSize) {
          console.warn(`[PSQ] Move out of bounds (${rawX},${rawY}) in ${file}`);
          invalidMove = true;
          break;
        }
        const stateBefore = cloneBoard(board);
        const policy = policyFromMove(boardSize, row, col);
        const teacherValue = outcomeForPlayer(winner, current);
        const sample: TrainingSample = {
          state: stateBefore,
          player: current,
          mcts_policy: policy,
          teacher_policy: policy,
          teacher_value: teacherValue,
          final_value: teacherValue,
          meta: {
            source: "psq",
            gameId,
            moveIndex: fileSamples.length,
            tags: [...baseTags],
            extra: { file: relativePath },
          },
        };
        fileSamples.push(sample);
        board[row][col] = current;
        current = current === 'black' ? 'white' : 'black';
      }

      if (invalidMove) {
        continue;
      }

      const totalMoves = fileSamples.length;
      const blackResult = winner ? (winner === "black" ? 1 : -1) : 0;
      const resultTag = winner ? `winner:${winner}` : "result:draw";
      for (let idx = 0; idx < fileSamples.length; idx++) {
        const sample = fileSamples[idx];
        const existingTags = sample.meta?.tags ?? [];
        const tags = Array.from(new Set([...existingTags, resultTag]));
        sample.meta = {
          ...(sample.meta ?? {}),
          moveIndex: idx,
          totalMoves,
          result: blackResult,
          tags,
        };
      }

      allSamples.push(...fileSamples);
    } catch (err) {
      console.warn(`[PSQ] Failed to process ${file}:`, err);
    }
  }

  if (allSamples.length === 0) {
    console.warn('[PSQ] No samples produced.');
    return;
  }

  const outFile = await writeJsonl(output, allSamples);
  console.log(`[PSQ] Wrote ${allSamples.length} samples to ${outFile}`);
}

main().catch((err) => {
  console.error('[PSQ] Fatal error:', err);
  process.exitCode = 1;
});









