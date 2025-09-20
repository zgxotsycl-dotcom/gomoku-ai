import * as path from 'path';
import * as fs from 'fs';
import tf from './tf';
import type * as TFT from '@tensorflow/tfjs';
import { createDualResNetModel } from './model';

async function main() {
  const BASE_DIR = process.env.APP_DIR || path.resolve(__dirname, '..');
  const PROD_MODEL_DIR = process.env.PROD_MODEL_DIR || path.resolve(BASE_DIR, 'gomoku_model_prod');
  const BOARD_SIZE = Number(process.env.BOARD_SIZE || 15);
  const modelJson = path.join(PROD_MODEL_DIR, 'model.json');

  const isValid = (p: string): boolean => {
    try {
      const raw = fs.readFileSync(p, 'utf-8');
      const j = JSON.parse(raw);
      return !!(j && (j.modelTopology && j.weightsManifest));
    } catch { return false; }
  };

  if (fs.existsSync(modelJson) && isValid(modelJson)) {
    console.log(`[bootstrap_model] Model already present at ${modelJson}`);
    return;
  }
  fs.mkdirSync(PROD_MODEL_DIR, { recursive: true });
  console.log('[bootstrap_model] Creating initial model...');
  const model: TFT.LayersModel = createDualResNetModel();
  // warmup
  tf.tidy(() => {
    const x = tf.zeros([1, BOARD_SIZE, BOARD_SIZE, 3]);
    const y = model.predict(x as any);
    void y;
  });
  await model.save(`file://${PROD_MODEL_DIR}`);
  console.log(`[bootstrap_model] Saved to ${modelJson}`);
}

main().catch((e) => { console.error(e); process.exit(1); });

