import { Worker } from 'worker_threads';
import * as path from 'path';

console.log("Attempting to load the game worker...");
try {
    const worker = new Worker(path.resolve(__dirname, 'game_worker.js'));
    console.log("Worker script loaded successfully. Terminating worker.");
    worker.terminate();
} catch (e) {
    console.error("Failed to load worker:", e);
}