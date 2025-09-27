import { spawnSync } from "child_process";
import * as path from "path";

interface Step {
  name: string;
  file: string;
  args?: string[];
  env?: NodeJS.ProcessEnv;
}

function runStep(step: Step, distRoot: string): void {
  const scriptPath = path.join(distRoot, step.file);
  const result = spawnSync(process.execPath, [scriptPath, ...(step.args ?? [])], {
    stdio: "inherit",
    env: { ...process.env, ...(step.env ?? {}) },
  });

  if (result.status !== 0) {
    throw new Error(`${step.name} failed with code ${result.status ?? "unknown"}`);
  }
}

function main(): void {
  const distRoot = path.resolve(__dirname, "..");
  const steps: Step[] = [
    { name: "Self-Play Data Generation", file: "self_play_trainer.js" },
    { name: "Distillation Training", file: "distill_student.js" },
    { name: "Arena Gating", file: "arena.js" },
    { name: "Model Upload", file: "upload_model.js" },
  ];

  console.log("=== Pipeline Cycle Start ===");
  for (const step of steps) {
    console.log(`\n>>> Running: ${step.name}`);
    runStep(step, distRoot);
  }
  console.log("\n=== Pipeline Cycle Completed ===");
}

main();
