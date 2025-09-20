import { createClient } from '@supabase/supabase-js';
import { updateStatus } from './status';
import * as fs from 'fs';
import * as fsp from 'fs/promises';
import * as path from 'path';

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
const SUPABASE_BUCKET = process.env.SUPABASE_BUCKET || 'models';
const SUPABASE_MODEL_PREFIX = process.env.SUPABASE_MODEL_PREFIX || `gomoku_model_${process.env.BOARD_SIZE || 15}`;
const MODEL_DIR = process.env.MODEL_DIR || path.resolve(__dirname, '..', 'gomoku_model_prod');
const PUBLIC_CACHE_CONTROL = process.env.PUBLIC_CACHE_CONTROL || 'public, max-age=60';

async function uploadFile(client: any, localPath: string, remotePath: string, contentType: string) {
  const data = await fsp.readFile(localPath);
  const { error } = await client.storage.from(SUPABASE_BUCKET).upload(remotePath, data, {
    contentType,
    cacheControl: PUBLIC_CACHE_CONTROL,
    upsert: true,
  });
  if (error) throw error;
  console.log(`Uploaded: ${remotePath}`);
}

async function main() {
  if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
    throw new Error('Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env.');
  }
  if (!fs.existsSync(MODEL_DIR)) {
    throw new Error(`MODEL_DIR not found: ${MODEL_DIR}`);
  }

  const client = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, { auth: { persistSession: false } });

  const modelJsonPath = path.join(MODEL_DIR, 'model.json');
  if (!fs.existsSync(modelJsonPath)) throw new Error(`model.json not found in ${MODEL_DIR}`);

  // Versioned folder, e.g., gomoku_model/2025-09-01T12-00-00Z
  const versionTag = new Date().toISOString().replace(/[:.]/g, '-');
  const versionedPrefix = `${SUPABASE_MODEL_PREFIX}/${versionTag}`;
  const latestPrefix = `${SUPABASE_MODEL_PREFIX}`; // also upsert latest

  // Upload model.json
  await uploadFile(client, modelJsonPath, `${versionedPrefix}/model.json`, 'application/json');
  await uploadFile(client, modelJsonPath, `${latestPrefix}/model.json`, 'application/json');

  // Upload all weight files (*.bin)
  const files = await fsp.readdir(MODEL_DIR);
  const binFiles = files.filter((f) => f.endsWith('.bin'));
  for (const bf of binFiles) {
    const lp = path.join(MODEL_DIR, bf);
    await uploadFile(client, lp, `${versionedPrefix}/${bf}`, 'application/octet-stream');
    await uploadFile(client, lp, `${latestPrefix}/${bf}`, 'application/octet-stream');
  }

  // Log public URL (assuming public bucket policy)
  const publicUrl = `${SUPABASE_URL}/storage/v1/object/public/${SUPABASE_BUCKET}/${latestPrefix}/model.json`;
  console.log('Public model URL:', publicUrl);
  try { await updateStatus({ upload: { publicUrl, ok: true } }); } catch {}
}

main().catch((e) => {
  console.error('Upload failed:', e);
  process.exitCode = 1;
});
