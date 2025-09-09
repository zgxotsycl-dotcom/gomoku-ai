"use strict";
/**
 * @file Value Head Pre-training Script (Node.js, Memory Optimized v2)
 * This script uses a pre-existing knowledge base to pre-train the value head.
 * It reads the entire dataset into string lines, then processes everything in chunks
 * (parsing, tensor conversion, training) to keep memory usage minimal.
 */
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
const tf = __importStar(require("@tensorflow/tfjs-node-gpu"));
const node_fs_1 = require("node:fs");
const path = __importStar(require("path"));
const model_1 = require("./src/model");
// --- Configuration ---
const INPUT_CSV_PATH = './ai_knowledge_export.csv';
const MODEL_SAVE_PATH = './gomoku_model';
const BOARD_SIZE = 19;
// Training Hyperparameters
const EPOCHS = 5;
const CHUNK_SIZE = 8192; // Number of records to process at a time
const BATCH_SIZE = 128;
function parseLinesToRecords(lines, header) {
    const hashIndex = header.indexOf('pattern_hash');
    const winRateIndex = header.indexOf('win_rate');
    const records = [];
    for (const line of lines) {
        if (!line)
            continue;
        const values = line.split(',');
        const boardStr = values[hashIndex];
        const winRateStr = values[winRateIndex];
        if (!boardStr || !winRateStr || boardStr.length < 19)
            continue;
        const winRate = parseFloat(winRateStr);
        if (isNaN(winRate))
            continue;
        const board = boardStr.split('|').map(rowStr => rowStr.split('').map(char => {
            if (char === 'b')
                return 'black';
            if (char === 'w')
                return 'white';
            return null;
        }));
        if (board.length !== BOARD_SIZE || !board[0] || board[0].length !== BOARD_SIZE)
            continue;
        records.push({ state: board, value: (winRate * 2) - 1 });
    }
    return records;
}
function convertRecordsToTensors(records) {
    return tf.tidy(() => {
        const states = [];
        const values = [];
        for (const record of records) {
            const player = 'black';
            const opponent = 'white';
            const playerChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));
            const opponentChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));
            for (let r = 0; r < BOARD_SIZE; r++) {
                for (let c = 0; c < BOARD_SIZE; c++) {
                    if (record.state[r][c] === player)
                        playerChannel[r][c] = 1;
                    else if (record.state[r][c] === opponent)
                        opponentChannel[r][c] = 1;
                }
            }
            const colorChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(1));
            const stacked = tf.stack([tf.tensor2d(playerChannel), tf.tensor2d(opponentChannel), tf.tensor2d(colorChannel)], 2);
            states.push(stacked.expandDims(0));
            values.push([record.value]);
        }
        const xs = tf.concat(states);
        const ys = tf.tensor2d(values);
        return { xs, ys };
    });
}
async function pretrain() {
    console.log(`Loading lines from ${INPUT_CSV_PATH}...`);
    let fileContent;
    try {
        fileContent = (0, node_fs_1.readFileSync)(INPUT_CSV_PATH, 'utf-8');
    }
    catch (error) {
        console.error(`Error reading file: ${error}`);
        return;
    }
    const allLines = fileContent.trim().split('\n');
    if (allLines.length < 2) {
        console.error('CSV file has no data.');
        return;
    }
    const header = allLines.shift().split(','); // Remove header and get column names
    console.log(`Loaded ${allLines.length} data lines.`);
    const fullModel = (0, model_1.createDualResNetModel)();
    const valueHeadOutput = fullModel.getLayer('value').output;
    const valueModel = tf.model({ inputs: fullModel.inputs, outputs: valueHeadOutput });
    valueModel.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'meanSquaredError',
        metrics: [tf.metrics.meanAbsoluteError]
    });
    console.log('Model compiled for VALUE-ONLY training.');
    const NUM_CHUNKS = Math.ceil(allLines.length / CHUNK_SIZE);
    for (let epoch = 0; epoch < EPOCHS; epoch++) {
        console.log(`\n--- Epoch ${epoch + 1} / ${EPOCHS} ---
`);
        tf.util.shuffle(allLines);
        for (let i = 0; i < NUM_CHUNKS; i++) {
            const start = i * CHUNK_SIZE;
            const end = Math.min(start + CHUNK_SIZE, allLines.length);
            const lineChunk = allLines.slice(start, end);
            console.log(`\nProcessing chunk ${i + 1}/${NUM_CHUNKS} (lines ${start}-${end})`);
            const records = parseLinesToRecords(lineChunk, header);
            if (records.length === 0) {
                console.log('No valid records in this chunk, skipping.');
                continue;
            }
            const { xs, ys } = convertRecordsToTensors(records);
            await valueModel.fit(xs, ys, {
                batchSize: BATCH_SIZE,
                epochs: 1,
                shuffle: true,
            });
            xs.dispose();
            ys.dispose();
            console.log(`Memory freed for chunk ${i + 1}. Num Tensors: ${tf.memory().numTensors}`);
        }
    }
    console.log('\nValue pre-training finished.');
    // ... (Code to transfer weights and save the full model)
    console.log(`Saving pre-trained full model to ${MODEL_SAVE_PATH}...`);
    await fullModel.save(`file:///${path.resolve(MODEL_SAVE_PATH)}`);
    console.log('Model saved successfully.');
}
pretrain().catch(console.error);
