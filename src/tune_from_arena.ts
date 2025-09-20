import * as fs from 'fs';
import * as path from 'path';
import { updateMany as updateTuning, getNumber as getTuned, clamp as clampNum } from './tuning';

function main() {
  const resPath = process.env.ARENA_RESULT_PATH || path.resolve(__dirname, '..', 'arena_result.json');
  if (!fs.existsSync(resPath)) {
    console.error(`[tune_from_arena] Result not found: ${resPath}`);
    process.exit(1);
  }
  const raw = fs.readFileSync(resPath, 'utf-8');
  const obj = JSON.parse(raw);
  if (!obj || typeof obj.winrate !== 'number') {
    console.error('[tune_from_arena] Invalid result JSON (missing winrate).');
    process.exit(2);
  }
  const winrate: number = obj.winrate;
  const thr: number = typeof obj.threshold === 'number' ? obj.threshold : getTuned('WINRATE_THRESHOLD', 0.55);
  const lr = Number(process.env.TUNE_LR || 0.03);
  const delta = (winrate - thr);
  const mul = clampNum(1 + lr * delta, 0.9, 1.1);
  const mixDelta = clampNum(0.1 * delta, -0.05, 0.05);
  const CHILD_TT_PRIOR_MIX = clampNum(getTuned('CHILD_TT_PRIOR_MIX', 0.35) + mixDelta, 0, 0.6);

  updateTuning({
    // Root boosts
    BOOST_CREATE: getTuned('BOOST_CREATE', 1.5) * mul,
    BOOST_BLOCK: getTuned('BOOST_BLOCK', 1.3) * mul,
    BOOST_OPEN3_ROOT: getTuned('BOOST_OPEN3_ROOT', 1.08) * mul,
    BOOST_OPEN3_ROOT_BLOCK: getTuned('BOOST_OPEN3_ROOT_BLOCK', 1.05) * mul,
    BOOST_FOUR_ROOT: getTuned('BOOST_FOUR_ROOT', 1.15) * mul,
    BOOST_FOUR_ROOT_BLOCK: getTuned('BOOST_FOUR_ROOT_BLOCK', 1.1) * mul,
    BOOST_CONN3_ROOT: getTuned('BOOST_CONN3_ROOT', 1.05) * mul,
    BOOST_CONN3_ROOT_BLOCK: getTuned('BOOST_CONN3_ROOT_BLOCK', 1.03) * mul,
    BOOST_LINK_ROOT: getTuned('BOOST_LINK_ROOT', 1.03) * mul,
    // Child boosts
    BOOST_CREATE_CHILD: getTuned('BOOST_CREATE_CHILD', 1.3) * mul,
    BOOST_BLOCK_CHILD: getTuned('BOOST_BLOCK_CHILD', 1.2) * mul,
    BOOST_OPEN3_CHILD: getTuned('BOOST_OPEN3_CHILD', 1.1) * mul,
    BOOST_OPEN3_BLOCK_CHILD: getTuned('BOOST_OPEN3_BLOCK_CHILD', 1.05) * mul,
    BOOST_CONN3_CHILD: getTuned('BOOST_CONN3_CHILD', 1.05) * mul,
    BOOST_CONN3_BLOCK_CHILD: getTuned('BOOST_CONN3_BLOCK_CHILD', 1.02) * mul,
    BOOST_LINK_CHILD: getTuned('BOOST_LINK_CHILD', 1.02) * mul,
    CHILD_TT_PRIOR_MIX,
  });
  console.log(`[tune_from_arena] Updated tuning with winrate=${winrate.toFixed(3)} thr=${thr.toFixed(3)} mul=${mul.toFixed(3)} mix=${CHILD_TT_PRIOR_MIX.toFixed(3)}`);
}

main();

