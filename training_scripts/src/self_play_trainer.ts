import { Worker } from 'node:worker_threads';
import * as path from 'path';
import * as fs from 'fs';

const NUM_WORKERS = 3; // Adjusted for stability
const SAVE_INTERVAL_MS = 60000; // Save data every 60 seconds
const OUTPUT_FILE = './training_data.jsonl';
const MODEL_DIR = './model_main';

async function runManager() {
    console.log('--- AI Self-Play Training Manager ---');

    if (!fs.existsSync(MODEL_DIR)) {
        console.error(`Error: Model directory not found at ${MODEL_DIR}`);
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

    console.log(`[Manager] Starting ${NUM_WORKERS} parallel game workers...`);

    for (let i = 0; i < NUM_WORKERS; i++) {
        const workerScriptPath = path.resolve(__dirname, '../dist/worker_selfplay.js');
        const worker = new Worker(workerScriptPath, { workerData: { workerId: i } });

        worker.on('message', (data) => {
            if (data.trainingSamples) {
                trainingDataBatch.push(...data.trainingSamples);
                console.log(`[Worker ${i}] Game finished. Received ${data.trainingSamples.length} samples. Batch size: ${trainingDataBatch.length}`);
            }
        });

        worker.on('error', (err) => console.error(`[Worker ${i}] Error:`, err));
        worker.on('exit', (code) => {
            if (code !== 0) {
                console.error(`[Worker ${i}] stopped with exit code ${code}. Restarting...`);
            }
        });
    }

    process.on('SIGINT', () => {
        console.log('\n[Manager] Shutdown signal received. Saving remaining data...');
        saveBatchToFile();
        process.exit(0);
    });
}

runManager();