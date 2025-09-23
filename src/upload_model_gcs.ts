import * as path from 'path';
import * as fs from 'fs';
import * as fsp from 'fs/promises';
import { Storage } from '@google-cloud/storage';
import { updateStatus } from './status';

const BASE_DIR = process.env.APP_DIR || path.resolve(__dirname, '..');
const MODEL_DIR = process.env.MODEL_DIR || path.resolve(BASE_DIR, 'gomoku_model_prod');
const GCS_BUCKET = process.env.GCS_BUCKET || '';
const BOARD_SIZE = Number(process.env.BOARD_SIZE || 15);
const GCS_MODEL_PREFIX = process.env.GCS_MODEL_PREFIX || `gomoku_model_${BOARD_SIZE}`;
const GCS_LATEST_PREFIX = process.env.GCS_LATEST_PREFIX || `${GCS_MODEL_PREFIX}/latest`;
const GCS_CACHE_CONTROL = process.env.GCS_CACHE_CONTROL || 'public, max-age=60';
const PUBLIC_BASE = process.env.GCS_PUBLIC_BASE_URL || '';

const storage = new Storage();

async function uploadFile(objectPath: string, localPath: string, contentType: string) {
  const bucket = storage.bucket(GCS_BUCKET);
  const file = bucket.file(objectPath);
  const data = await fsp.readFile(localPath);
  await file.save(data, {
    contentType,
    resumable: false,
    metadata: {
      cacheControl: GCS_CACHE_CONTROL,
    },
    public: false,
  });
  console.log(`Uploaded gs://${GCS_BUCKET}/${objectPath}`);
}

async function main() {
  if (!GCS_BUCKET) {
    throw new Error('Missing GCS_BUCKET environment variable');
  }
  if (!fs.existsSync(MODEL_DIR)) {
    throw new Error(`MODEL_DIR not found: ${MODEL_DIR}`);
  }
  const modelJson = path.join(MODEL_DIR, 'model.json');
  if (!fs.existsSync(modelJson)) {
    throw new Error(`model.json not found in ${MODEL_DIR}`);
  }
  const files = await fsp.readdir(MODEL_DIR);
  const binFiles = files.filter((f) => f.endsWith('.bin'));
  if (binFiles.length === 0) {
    throw new Error(`No weight .bin files found in ${MODEL_DIR}`);
  }

  const versionTag = new Date().toISOString().replace(/[:.]/g, '-');
  const versionedPrefix = `${GCS_MODEL_PREFIX}/${versionTag}`;

  await uploadFile(`${versionedPrefix}/model.json`, modelJson, 'application/json');
  await uploadFile(`${GCS_LATEST_PREFIX}/model.json`, modelJson, 'application/json');

  for (const file of binFiles) {
    const localPath = path.join(MODEL_DIR, file);
    await uploadFile(`${versionedPrefix}/${file}`, localPath, 'application/octet-stream');
    await uploadFile(`${GCS_LATEST_PREFIX}/${file}`, localPath, 'application/octet-stream');
  }

  const publicUrl = PUBLIC_BASE ? `${PUBLIC_BASE.replace(/\/$/, '')}/${GCS_LATEST_PREFIX}/model.json` : undefined;
  console.log('GCS upload complete.');
  if (publicUrl) {
    console.log('Public model URL:', publicUrl);
  }
  try {
    await updateStatus({
      upload: {
        gcs: {
          ok: true,
          versionPath: `gs://${GCS_BUCKET}/${versionedPrefix}`,
          latestPath: `gs://${GCS_BUCKET}/${GCS_LATEST_PREFIX}`,
          publicUrl,
        },
      },
    });
  } catch {}
}

main().catch((e) => {
  console.error('GCS upload failed:', e);
  process.exitCode = 1;
});
