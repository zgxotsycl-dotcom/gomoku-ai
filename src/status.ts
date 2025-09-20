import * as fs from 'fs';
import * as fsp from 'fs/promises';
import * as path from 'path';

const STATUS_PATH = process.env.STATUS_PATH || path.resolve(__dirname, '..', 'logs', 'status.json');

export type PipelinePhase =
  | 'idle'
  | 'self_play_start'
  | 'self_play_done'
  | 'distill_start'
  | 'distill_done'
  | 'arena_start'
  | 'arena_done'
  | 'upload_start'
  | 'upload_done'
  | 'book_build_start'
  | 'book_build_done'
  | 'book_import_start'
  | 'book_import_done'
  | 'error';

export type Status = {
  ts: string;
  phase?: PipelinePhase;
  cycle?: number;
  selfPlay?: {
    samplesTotal?: number;
    lastSave?: string;
    workers?: number;
    durationMs?: number;
  };
  distill?: {
    epoch?: number;
    epochs?: number;
    lastSaved?: string;
  };
  arena?: {
    played?: number;
    total?: number;
    candWins?: number;
    prodWins?: number;
    draws?: number;
    promoted?: boolean;
  };
  upload?: {
    publicUrl?: string;
    ok?: boolean;
  };
  book?: {
    entries?: number;
    imported?: boolean;
  };
  error?: string;
};

function merge(a: any, b: any) {
  const out: any = Array.isArray(a) ? [...a] : { ...(a || {}) };
  for (const k of Object.keys(b || {})) {
    if (b[k] && typeof b[k] === 'object' && !Array.isArray(b[k])) out[k] = merge(out[k], b[k]);
    else out[k] = b[k];
  }
  return out;
}

export async function updateStatus(patch: Partial<Status>) {
  try {
    let prev: Status | {} = {};
    if (fs.existsSync(STATUS_PATH)) {
      const raw = await fsp.readFile(STATUS_PATH, 'utf-8');
      prev = JSON.parse(raw);
    } else {
      await fsp.mkdir(path.dirname(STATUS_PATH), { recursive: true });
    }
    const next: Status = merge(prev, { ts: new Date().toISOString(), ...patch });
    await fsp.writeFile(STATUS_PATH, JSON.stringify(next, null, 2), 'utf-8');
  } catch {
    // ignore
  }
}

export function getStatusPath() {
  return STATUS_PATH;
}

