import { fork } from 'child_process';
import * as path from 'path';

// --- Configuration ---
const NUM_WORKERS = 4;

const SCRIPTS = {
    worker: path.resolve(__dirname, '../dist/worker_selfplay.js'),
    trainer: path.resolve(__dirname, '../dist/trainer.js'),
    evaluator: path.resolve(__dirname, '../dist/evaluator.js'),
};

function startProcess(scriptPath: string, name: string, args: string[] = []) {
    console.log(`Starting ${name}...`);
    const child = fork(scriptPath, args, { stdio: 'inherit' }); // stdio: 'inherit' allows us to see the logs from the child process

    child.on('exit', (code) => {
        if (code !== 0) {
            console.error(`
--- ${name} has crashed with exit code ${code}. Restarting in 10 seconds... ---
`);
            setTimeout(() => startProcess(scriptPath, name, args), 10000);
        } else {
            console.log(`--- ${name} has exited cleanly. ---`);
        }
    });

    return child;
}

function main() {
    console.log('--- Starting Go AI Training Pipeline ---');

    // Start all self-play workers
    for (let i = 0; i < NUM_WORKERS; i++) {
        startProcess(SCRIPTS.worker, `Worker-${i}`, [String(i)]);
    }

    // Start the trainer
    startProcess(SCRIPTS.trainer, 'Trainer');

    // Start the evaluator
    startProcess(SCRIPTS.evaluator, 'Evaluator');

    console.log(`
--- All ${NUM_WORKERS} workers, 1 trainer, and 1 evaluator have been started. ---
`);
}

main();
