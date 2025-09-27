import { spawnSync } from "child_process";
import * as fs from "fs";
import * as path from "path";

interface ExperimentSpec {
  name: string;
  env?: Record<string, string>;
  args?: string[];
}

const DEFAULT_CONFIG = path.resolve(process.cwd(), 'experiments.json');
const RESULTS_DIR = process.env.EXPERIMENT_RESULTS_DIR || path.resolve(process.cwd(), 'experiments/results');

function loadExperiments(): ExperimentSpec[] {
  const explicit = process.env.EXPERIMENT_CONFIG_PATH;
  const filePath = explicit ? path.resolve(explicit) : DEFAULT_CONFIG;
  if (!fs.existsSync(filePath)) {
    console.warn(`[Experiments] Config file missing: ${filePath}. Create a JSON array of experiments.`);
    return [];
  }
  try {
    const raw = fs.readFileSync(filePath, "utf-8");
    const data = JSON.parse(raw) as ExperimentSpec[];
    return Array.isArray(data) ? data : [];
  } catch (err) {
    console.error("[Experiments] Failed to parse config:", err);
    return [];
  }
}

function ensureResultsDir(): string {
  fs.mkdirSync(RESULTS_DIR, { recursive: true });
  return RESULTS_DIR;
}

function writeResult(exp: ExperimentSpec, status: 'success' | 'failure', details: Record<string, any>): void {
  const dir = ensureResultsDir();
  const ts = new Date().toISOString().replace(/[:.]/g, '-');
  const safeName = exp.name.replace(/[^a-z0-9_-]+/gi, '_');
  const file = path.join(dir, `${ts}_${safeName}_${status}.json`);
  const payload = {
    name: exp.name,
    status,
    timestamp: ts,
    env: exp.env ?? {},
    args: exp.args ?? [],
    ...details,
  };
  fs.writeFileSync(file, JSON.stringify(payload, null, 2));
}

function runExperiment(exp: ExperimentSpec, distRoot: string): void {
  const cycleScript = path.join(distRoot, "scripts", "pipeline_cycle.js");
  const args = exp.args ?? [];
  const env = { ...process.env, ...(exp.env ?? {}), EXPERIMENT_NAME: exp.name };
  console.log(`\n===== Experiment: ${exp.name} =====`);
  const startedAt = Date.now();
  const result = spawnSync(process.execPath, [cycleScript, ...args], { stdio: 'inherit', env });
  const durationMs = Date.now() - startedAt;
  if (result.status !== 0) {
    writeResult(exp, 'failure', { code: result.status ?? 'unknown', durationMs });
    throw new Error(`Experiment ${exp.name} failed with code ${result.status ?? 'unknown'}`);
  }
  writeResult(exp, 'success', { durationMs });
}

function main(): void {
  const experiments = loadExperiments();
  if (experiments.length === 0) {
    console.warn("[Experiments] No experiments to run.");
    return;
  }
  const distRoot = path.resolve(__dirname, "..");
  for (const exp of experiments) {
    runExperiment(exp, distRoot);
  }
  console.log("\nAll experiments completed.");
}

main();
