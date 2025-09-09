"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
const node_worker_threads_1 = require("node:worker_threads");
const path = __importStar(require("node:path"));
const fs = __importStar(require("node:fs"));
const NUM_WORKERS = 4; // Adjust based on your CPU cores
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
    let trainingDataBatch = [];
    function saveBatchToFile() {
        if (trainingDataBatch.length === 0)
            return;
        console.log(`[Manager] Saving batch of ${trainingDataBatch.length} new training samples...`);
        const data = trainingDataBatch.map(s => JSON.stringify(s)).join('\n') + '\n';
        trainingDataBatch = []; // Clear the batch
        try {
            fs.appendFileSync(OUTPUT_FILE, data);
            console.log(`[Manager] Successfully saved samples to ${OUTPUT_FILE}`);
        }
        catch (err) {
            console.error("[Manager] Failed to save batch:", err);
        }
    }
    // Save data periodically
    setInterval(saveBatchToFile, SAVE_INTERVAL_MS);
    console.log(`[Manager] Starting ${NUM_WORKERS} parallel game workers...`);
    for (let i = 0; i < NUM_WORKERS; i++) {
        // Node.js requires the compiled .js file for the worker.
        const workerScriptPath = path.resolve(__dirname, '../dist/game_worker.js');
        const worker = new node_worker_threads_1.Worker(workerScriptPath, { workerData: { workerId: i } });
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
                // Optional: Restart worker on failure
            }
        });
    }
    // Handle graceful shutdown
    process.on('SIGINT', () => {
        console.log('\n[Manager] Shutdown signal received. Saving remaining data...');
        saveBatchToFile();
        process.exit(0);
    });
}
runManager();
