import * as fs from 'fs';
import * as fsp from 'fs/promises';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';
import { updateStatus } from './status';

function ensureSupabaseEnv() {
  if (process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_ROLE_KEY) return;
  const candidate = path.resolve(__dirname, '..', 'gomoku-app-v2', '.env.local');
  try {
    if (fs.existsSync(candidate)) {
      const txt = fs.readFileSync(candidate, 'utf-8');
      for (const line of txt.split(/\r?\n/)) {
        const m = line.match(/^([A-Z0-9_]+)=(.*)$/);
        if (m) {
          const k = m[1];
          const v = m[2];
          if (!process.env[k]) process.env[k] = v;
        }
      }
    }
  } catch {}
}

ensureSupabaseEnv();

const SUPABASE_URL = process.env.SUPABASE_URL || '';
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || '';
const BOOK_FILE = process.env.BOOK_FILE || path.resolve(__dirname, '..', 'opening_book_generated.json');
const FUNCTION_NAME = process.env.OPENING_BOOK_FUNCTION || 'import-opening-book';

type BookEntry = {
  board_hash: string;
  best_move: [number, number];
  move_count?: number | null;
};

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

function hashToBoard(hash: string): string[][] {
  return hash.split('|').map((row) => row.split(''));
}

function boardToHash(board: string[][]): string {
  return board.map((row) => row.join('')).join('|');
}

function transformBoardHash(hash: string, t: TransformId): string {
  const rows = hashToBoard(hash);
  const n = rows.length;
  const out: string[][] = Array.from({ length: n }, () => Array(n).fill('-'));
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const [rr, cc] = transformRC(r, c, n, t);
      out[rr][cc] = rows[r][c];
    }
  }
  return boardToHash(out);
}

function canonicalizeHash(hash: string): { hash: string; t: TransformId } {
  let best = '';
  let bestT: TransformId = 0;
  let first = true;
  for (const t of [0, 1, 2, 3, 4, 5, 6, 7] as TransformId[]) {
    const candidate = transformBoardHash(hash, t);
    if (first || candidate < best) {
      best = candidate;
      bestT = t;
      first = false;
    }
  }
  return { hash: best, t: bestT };
}

function canonicalizeEntries(entries: BookEntry[]): BookEntry[] {
  return entries.map((entry) => {
    const { hash, t } = canonicalizeHash(entry.board_hash);
    const size = Math.sqrt(hash.split('|').join('').length);
    const [r, c] = transformRC(entry.best_move[0], entry.best_move[1], size, t);
    return {
      board_hash: hash,
      best_move: [r, c],
      move_count: entry.move_count ?? 0,
    };
  });
}

async function upsertDirect(entries: BookEntry[]): Promise<number> {
  const client = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, { auth: { persistSession: false } });
  const payload = canonicalizeEntries(entries);
  const { error } = await client
    .from('ai_opening_book')
    .upsert(payload, { onConflict: 'board_hash' });
  if (error) {
    const message = error.message || 'Unknown Supabase error';
    const details = 'details' in error && error.details ? `: ${error.details}` : '';
    throw new Error(message + details);
  }
  return payload.length;
}

async function main() {
  if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
    throw new Error('Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY');
  }
  if (!fs.existsSync(BOOK_FILE)) {
    throw new Error(`Opening book file not found: ${BOOK_FILE}`);
  }
  const content = await fsp.readFile(BOOK_FILE, 'utf-8');
  const parsed = JSON.parse(content);
  if (!Array.isArray(parsed)) {
    throw new Error('Opening book data must be an array of entries.');
  }
  const entries = parsed as BookEntry[];
  const url = `${SUPABASE_URL}/functions/v1/${FUNCTION_NAME}`;

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(entries),
  });

  if (!res.ok) {
    const txt = await res.text();
    console.warn(`[Book] Edge function import failed: ${res.status} ${res.statusText} - ${txt}`);
    try {
      const inserted = await upsertDirect(entries);
      console.log(`[Book] Direct Supabase upsert succeeded (${inserted} entries).`);
      try { await updateStatus({ book: { imported: true, entries: inserted } }); } catch {}
      return;
    } catch (fallbackErr) {
      try { await updateStatus({ book: { imported: false } }); } catch {}
      const errMsg = fallbackErr instanceof Error ? fallbackErr.message : JSON.stringify(fallbackErr);
      throw new Error(`Import failed via edge function (${res.status} ${res.statusText}) - ${txt}; direct upsert error: ${errMsg}`);
    }
  }
  const data = await res.json();
  console.log('Import opening book success:', data);
  try { await updateStatus({ book: { imported: true, entries: entries.length } }); } catch {}
}

main().catch((e) => {
  console.error('Import opening book failed:', e);
  process.exitCode = 1;
});
