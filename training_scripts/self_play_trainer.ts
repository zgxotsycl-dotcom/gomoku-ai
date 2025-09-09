import { Worker } from 'node:worker_threads';
import * as path from 'node:path';
import * as fs from 'node:fs';

const NUM_WORKERS = 7; // Adjust based on your CPU cores
const SAVE_INTERVAL_MS = 60000; // Save data every 60 seconds
const OUTPUT_FILE = './training_data.jsonl';
const MODEL_DIR = './gomoku_model';

async function runManager() {
    console.log('--- AI Self-Play Training Manager ---');

    if (!fs.existsSync(MODEL_DIR)) {
        console.error(`Error: Model directory not found at ${MODEL_DIR}`);
        console.error('Please run the create_initial_model.ts script first.');
        return;
    }

    let trainingDataBatch: any[] = [];

    function saveBatchToFile() {
        if (trainingDataBatch.length === 0) return;
        console.log(`[Manager] Saving batch of ${trainingDataBatch.length} new training samples...`);
        const data = trainingDataBatch.map(s => JSON.stringify(s)).join('\n') + '\n';
        trainingDataBatch = []; // Clear the batch
        try {
            fs.appendFileSync(OUTPUT_FILE, data);
            console.log(`[Manager] Successfully saved samples to ${OUTPUT_FILE}`);
        } catch (err) {
            console.error("[Manager] Failed to save batch:", err);
        }
    }

    setInterval(saveBatchToFile, SAVE_INTERVAL_MS);

    const createWorker = (workerId: number) => {
        console.log(`[Manager] Creating worker ${workerId}...`);
        const workerScriptPath = path.resolve(__dirname, '../dist/game_worker.js');
        const worker = new Worker(workerScriptPath, { workerData: { workerId } });

        worker.on('message', (data) => {
            if (data.trainingSamples) {
                trainingDataBatch.push(...data.trainingSamples);
                console.log(`[Worker ${workerId}] Game finished. Received ${data.trainingSamples.length} samples. Batch size: ${trainingDataBatch.length}`);
                // Start a new game in the same worker
                worker.postMessage('start_new_game');
            }
        });

        worker.on('error', (err) => {
            console.error(`[Worker ${workerId}] FATAL ERROR:`, err);
        });

        worker.on('exit', (code) => {
            console.log(`[Worker ${workerId}] exited with code ${code}.`);
            if (code !== 0) {
                console.error(`[Worker ${workerId}] stopped unexpectedly. Restarting after 5 seconds...`);
                setTimeout(() => createWorker(workerId), 5000);
            }
        });
    };

    console.log(`[Manager] Starting ${NUM_WORKERS} parallel game workers...`);
    for (let i = 0; i < NUM_WORKERS; i++) {
        createWorker(i);
    }

    process.on('SIGINT', () => {
        console.log('\n[Manager] Shutdown signal received. Saving remaining data...');
        saveBatchToFile();
        process.exit(0);
    });
}

runManager();