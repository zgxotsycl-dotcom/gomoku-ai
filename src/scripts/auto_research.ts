import { spawnSync } from "child_process";
import * as fs from "fs";
import * as path from "path";

interface TaskConfig {
  name?: string;
  npm?: string;      // npm script name
  args?: string[];   // extra args (after --)
  command?: string;  // arbitrary shell command
  env?: Record<string, string>;
  continueOnError?: boolean;
}

const DEFAULT_CONFIG_PATH = path.resolve(process.cwd(), "automation.json");

function loadTasks(): TaskConfig[] {
  const explicit = process.env.AUTO_CONFIG_PATH;
  const filePath = explicit ? path.resolve(explicit) : DEFAULT_CONFIG_PATH;
  if (fs.existsSync(filePath)) {
    try {
      const raw = fs.readFileSync(filePath, "utf-8");
      const data = JSON.parse(raw) as TaskConfig[];
      if (Array.isArray(data) && data.length > 0) return data;
    } catch (err) {
      console.warn(`[Auto] Failed to parse ${filePath}:`, err);
    }
  }

  const tuneEnv: Record<string, string> = {};
  tuneEnv.TUNE_RUN = process.env.TUNE_RUN && process.env.TUNE_RUN.length > 0 ? process.env.TUNE_RUN : 'true';

  const pipelineEnv: Record<string, string> = {};
  const override = process.env.AUTO_PIPELINE_TF_USE_GPU;
  if (override && override.length > 0) pipelineEnv.TF_USE_GPU = override;
  else if (process.env.TF_USE_GPU && process.env.TF_USE_GPU.length > 0) pipelineEnv.TF_USE_GPU = process.env.TF_USE_GPU as string;
  else pipelineEnv.TF_USE_GPU = '0';

  return [
    { name: "Build dist", npm: "build" },
    { name: "Ingest PSQ batches", npm: "data:ingest:psq:batch", continueOnError: true },
    { name: "Curate replay buffer", npm: "curate:replay", continueOnError: true },
    { name: "Generate tuning experiments", npm: "tune:hparams", env: tuneEnv, continueOnError: true },
    { name: "Run experiments", npm: "experiments", continueOnError: true },
    { name: "Pipeline cycle", npm: "pipeline:cycle", env: pipelineEnv },
    { name: "Analyze PSQ hotspots", npm: "analyze:psq", continueOnError: true },
    { name: "Status report", npm: "report:status" },
    { name: "Inference health check", npm: "check:inference", continueOnError: false },
  ];
}

function runCommand(label: string, task: TaskConfig): void {
  const mergedEnv = { ...process.env, ...(task.env ?? {}) };
  let result;
  if (task.npm) {
    const npmArgs = ['run', task.npm];
    if (task.args && task.args.length > 0) npmArgs.push('--', ...task.args);
    result = spawnSync('npm', npmArgs, {
      stdio: 'inherit',
      env: mergedEnv,
      shell: process.platform === 'win32',
    });
  } else if (task.command) {
    result = spawnSync(task.command, {
      stdio: 'inherit',
      env: mergedEnv,
      shell: true,
    });
  } else {
    console.warn(`[Auto] Task ${label} has no npm/command definition. Skipping.`);
    return;
  }

  if (result.status !== 0) {
    const code = result.status ?? 'unknown';
    const message = `[Auto] Task ${label} failed with code ${code}`;
    if (task.continueOnError) {
      console.warn(message);
    } else {
      throw new Error(message);
    }
  }
}

function main(): void {
  const tasks = loadTasks();
  if (tasks.length === 0) {
    console.log('[Auto] No tasks configured.');
    return;
  }
  console.log('[Auto] Starting automation sequence');
  tasks.forEach((task, index) => {
    const label = task.name ?? `Task ${index + 1}`;
    console.log(`\n=== ${label} ===`);
    runCommand(label, task);
  });
  console.log('\n[Auto] Automation sequence complete.');
}

main();
