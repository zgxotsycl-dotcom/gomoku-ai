import * as fs from "fs";
import * as path from "path";

interface ExperimentSpec {
  name: string;
  env: Record<string, string>;
}

function parseList(name: string, fallback: string[]): string[] {
  const raw = process.env[name];
  if (!raw) return fallback;
  return raw.split(',').map((v) => v.trim()).filter(Boolean);
}

function cartesian<T>(arrays: T[][]): T[][] {
  return arrays.reduce<T[][]>((acc, curr) => {
    const next: T[][] = [];
    for (const a of acc) {
      for (const c of curr) {
        next.push([...a, c]);
      }
    }
    return next;
  }, [[]]);
}

function main(): void {
  const blocks = parseList('TUNE_RESIDUAL_BLOCKS', ['7', '9']);
  const filters = parseList('TUNE_CONV_FILTERS', ['96', '128']);
  const lr = parseList('TUNE_LEARNING_RATES', ['0.0005', '0.0003']);
  const mcts = parseList('TUNE_MCTS_THINK_MS', ['3600', '4800']);

  const combos = cartesian([blocks, filters, lr, mcts]);
  if (combos.length === 0) {
    console.warn('[Tune] No combinations generated.');
    return;
  }

  const experiments: ExperimentSpec[] = combos.map(([b, f, lrValue, mctsValue], idx) => ({
    name: `tune-${idx + 1}-b${b}-f${f}-lr${lrValue}-m${mctsValue}`,
    env: {
      RESIDUAL_BLOCKS: b,
      CONV_FILTERS: f,
      MODEL_LEARNING_RATE: lrValue,
      MCTS_THINK_TIME_MS: mctsValue,
    }
  }));

  const outFile = process.env.TUNE_OUTPUT || path.resolve(process.cwd(), 'experiments.auto.json');
  fs.writeFileSync(outFile, JSON.stringify(experiments, null, 2));
  console.log(`[Tune] Wrote ${experiments.length} experiments to ${outFile}`);

  if ((process.env.TUNE_RUN || '').toLowerCase() === 'true') {
    const result = require('child_process').spawnSync('npm', ['run', 'experiments'], {
      stdio: 'inherit',
      env: { ...process.env, EXPERIMENT_CONFIG_PATH: outFile },
      shell: process.platform === 'win32',
    });
    if (result.status !== 0) {
      console.error('[Tune] experiments run failed');
      process.exitCode = result.status ?? 1;
    }
  }
}

main();
