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
const child_process_1 = require("child_process");
const path = __importStar(require("path"));
// --- Configuration ---
const NUM_WORKERS = 5;
const SCRIPTS = {
    worker: path.resolve(__dirname, '../dist/worker_selfplay.js'),
    trainer: path.resolve(__dirname, '../dist/trainer.js'),
    evaluator: path.resolve(__dirname, '../dist/evaluator.js'),
};
function startProcess(scriptPath, name, args = []) {
    console.log(`Starting ${name}...`);
    const child = (0, child_process_1.fork)(scriptPath, args, { stdio: 'inherit' }); // stdio: 'inherit' allows us to see the logs from the child process
    child.on('exit', (code) => {
        if (code !== 0) {
            console.error(`
--- ${name} has crashed with exit code ${code}. Restarting in 10 seconds... ---
`);
            setTimeout(() => startProcess(scriptPath, name, args), 10000);
        }
        else {
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
