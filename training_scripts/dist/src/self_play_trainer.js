"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const node_path_1 = __importDefault(require("node:path"));
const node_worker_threads_1 = require("node:worker_threads");
const node_fs_1 = __importDefault(require("node:fs"));
const NUM_WORKERS = 4; // Start with a reasonable number
const SAVE_INTERVAL_MS = 60000; // 1 minute
// Output file in the root of the training_scripts directory
const OUTPUT_FILE = node_path_1.default.resolve(__dirname, '../../training_data.jsonl');
async function runParallelSelfPlay() {
    console.log('--- Go-Moku AI Self-Play Trainer ---');
    console.log(`Outputting training data to: ${OUTPUT_FILE}`);
    let trainingDataBatch = [];
    function ensureFileSync(filePath) {
        const dirname = node_path_1.default.dirname(filePath);
        if (!node_fs_1.default.existsSync(dirname)) {
            node_fs_1.default.mkdirSync(dirname, { recursive: true });
        }
    }
    function saveBatchToFile() {
        if (trainingDataBatch.length === 0)
            return;
        console.log(`Saving ${trainingDataBatch.length} new training samples...`);
        const data = trainingDataBatch.map(s => JSON.stringify(s)).join('\n') + '\n';
        trainingDataBatch = [];
        try {
            ensureFileSync(OUTPUT_FILE);
            node_fs_1.default.appendFileSync(OUTPUT_FILE, data);
            console.log(`Successfully saved to ${OUTPUT_FILE}`);
        }
        catch (err) {
            console.error("Failed to save batch:", err);
        }
    }
    setInterval(saveBatchToFile, SAVE_INTERVAL_MS);
    console.log(`Starting ${NUM_WORKERS} parallel game workers...`);
    for (let i = 0; i < NUM_WORKERS; i++) {
        // The worker path should be the compiled JS file in the same directory
        const workerPath = node_path_1.default.join(__dirname, 'game_worker.js');
        const worker = new node_worker_threads_1.Worker(workerPath);
        worker.on('message', (data) => {
            const { trainingSamples } = data;
            if (trainingSamples) {
                trainingDataBatch.push(...trainingSamples);
                console.log(`Worker ${i}: Game finished. Received ${trainingSamples.length} samples. Total in batch: ${trainingDataBatch.length}`);
                // Start a new game in the same worker
                worker.postMessage('start_new_game');
            }
        });
        worker.on('error', (err) => console.error(`Worker ${i} error:`, err));
        worker.on('exit', (code) => {
            if (code !== 0)
                console.error(`Worker ${i} stopped with exit code ${code}`);
        });
    }
    // Keep the main thread alive
    // This is a bit of a hack, but it works for a long-running script.
    await new Promise(() => { });
}
if (node_worker_threads_1.isMainThread) {
    runParallelSelfPlay().catch(e => console.error("Critical error in manager:", e));
}
