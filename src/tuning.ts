import * as fs from 'fs';
import * as path from 'path';

// Lightweight persistent numeric parameter store with ENV override.
// - getNumber(key, fallback): number
// - setNumber(key, value): void (persists to tuning_params.json)
// - updateMany(map): void

const BASE_DIR = process.env.APP_DIR || path.resolve(__dirname, '..');
const TUNING_PATH = process.env.TUNING_PARAMS_PATH || path.join(BASE_DIR, 'tuning_params.json');

type ParamMap = Record<string, number>;

function loadFile(): ParamMap {
  try {
    if (fs.existsSync(TUNING_PATH)) {
      const raw = fs.readFileSync(TUNING_PATH, 'utf-8');
      const obj = JSON.parse(raw);
      if (obj && typeof obj === 'object') return obj as ParamMap;
    }
  } catch {}
  return {};
}

function saveFile(obj: ParamMap) {
  try {
    fs.writeFileSync(TUNING_PATH, JSON.stringify(obj, null, 2), 'utf-8');
  } catch {}
}

let cache: ParamMap | null = null;

function ensureCache() {
  if (!cache) cache = loadFile();
}

export function getNumber(key: string, fallback: number): number {
  // ENV overrides tuning file
  const env = process.env[key];
  if (env != null && env !== '') {
    const v = Number(env);
    if (!Number.isNaN(v)) return v;
  }
  ensureCache();
  const v = (cache as ParamMap)[key];
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  return fallback;
}

export function setNumber(key: string, value: number): void {
  ensureCache();
  (cache as ParamMap)[key] = value;
  saveFile(cache as ParamMap);
}

export function updateMany(updates: ParamMap): void {
  ensureCache();
  cache = { ...(cache as ParamMap), ...updates };
  saveFile(cache as ParamMap);
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

export function mixPolicies(a: Float32Array, b: Float32Array, mix: number): Float32Array {
  // Normalize a and b to sum=1 over non-negative entries, then return (1-mix)*a + mix*b
  const out = new Float32Array(a.length);
  let sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) { sa += Math.max(0, a[i]); sb += Math.max(0, b[i]); }
  const ia = sa > 0 ? 1 / sa : 0;
  const ib = sb > 0 ? 1 / sb : 0;
  const m = clamp(mix, 0, 1);
  for (let i = 0; i < a.length; i++) {
    const aa = ia > 0 ? Math.max(0, a[i]) * ia : 0;
    const bb = ib > 0 ? Math.max(0, b[i]) * ib : 0;
    out[i] = (1 - m) * aa + m * bb;
  }
  return out;
}

export function applyDistancePenalty(
  size: number,
  r: number,
  c: number,
  lambda: number,
  kind: 'exp' | 'linear' = 'exp'
): number {
  if (!Number.isFinite(lambda) || lambda <= 0) return 1;
  const mid = Math.floor(size / 2);
  const d = Math.abs(r - mid) + Math.abs(c - mid);
  if (kind === 'linear') return 1 / (1 + lambda * d);
  // default: exponential
  return Math.exp(-lambda * d);
}

