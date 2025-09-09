/**
 * @file Value Head Pre-training Script (Node.js, Memory Optimized v2)
 * This script uses a pre-existing knowledge base to pre-train the value head.
 * It reads the entire dataset into string lines, then processes everything in chunks
 * (parsing, tensor conversion, training) to keep memory usage minimal.
 */

import * as tf from '@tensorflow/tfjs-node-gpu';
import { readFileSync } from 'node:fs';
import { promises as fs } from 'node:fs';
import * as path from 'path';
import { createDualResNetModel } from './src/model';
import type { Player } from './src/ai';

// --- Configuration ---
const INPUT_CSV_PATH = './ai_knowledge_export.csv';
const MODEL_SAVE_PATH = './gomoku_model';
const BOARD_SIZE = 19;

// Training Hyperparameters
const EPOCHS = 5;
const CHUNK_SIZE = 8192; // Number of records to process at a time
const BATCH_SIZE = 128;

interface KnowledgeRecord {
    state: (Player | null)[][];
    value: number;
}

function parseLinesToRecords(lines: string[], header: string[]): KnowledgeRecord[] {
    const hashIndex = header.indexOf('pattern_hash');
    const winRateIndex = header.indexOf('win_rate');
    
    const records: KnowledgeRecord[] = [];
    for (const line of lines) {
        if (!line) continue;
        const values = line.split(',');
        const boardStr = values[hashIndex];
        const winRateStr = values[winRateIndex];

        if (!boardStr || !winRateStr || boardStr.length < 19) continue;
        const winRate = parseFloat(winRateStr);
        if (isNaN(winRate)) continue;

        const board = boardStr.split('|').map(rowStr =>
            rowStr.split('').map(char => {
                if (char === 'b') return 'black';
                if (char === 'w') return 'white';
                return null;
            })
        ) as (Player | null)[][];

        if (board.length !== BOARD_SIZE || !board[0] || board[0].length !== BOARD_SIZE) continue;
        records.push({ state: board, value: (winRate * 2) - 1 });
    }
    return records;
}

function convertRecordsToTensors(records: KnowledgeRecord[]): { xs: tf.Tensor4D, ys: tf.Tensor2D } {
    return tf.tidy(() => {
        const states: tf.Tensor4D[] = [];
        const values: number[][] = [];

        for (const record of records) {
            const player: Player = 'black';
            const opponent: Player = 'white';
            const playerChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));
            const opponentChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));

            for (let r = 0; r < BOARD_SIZE; r++) {
                for (let c = 0; c < BOARD_SIZE; c++) {
                    if (record.state[r][c] === player) playerChannel[r][c] = 1;
                    else if (record.state[r][c] === opponent) opponentChannel[r][c] = 1;
                }
            }
            const colorChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(1));

            const stacked = tf.stack([tf.tensor2d(playerChannel), tf.tensor2d(opponentChannel), tf.tensor2d(colorChannel)], 2);
            states.push(stacked.expandDims(0) as tf.Tensor4D);
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
        fileContent = readFileSync(INPUT_CSV_PATH, 'utf-8');
    } catch (error) {
        console.error(`Error reading file: ${error}`);
        return;
    }
    const allLines = fileContent.trim().split('\n');
    if (allLines.length < 2) {
        console.error('CSV file has no data.');
        return;
    }

    const header = allLines.shift()!.split(','); // Remove header and get column names
    console.log(`Loaded ${allLines.length} data lines.`);

    const fullModel = createDualResNetModel();
    const valueHeadOutput = fullModel.getLayer('value').output as tf.SymbolicTensor;
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
            if(records.length === 0) {
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
