"use strict";
/**
 * @file Standalone Evaluation Worker
 * This script runs in an infinite loop, continuously checking for new model checkpoints,
 * evaluating them against the current best model, and promoting the challenger if it's stronger,
 * and automatically deploying the new champion model to Supabase Storage.
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
const fs = __importStar(require("fs/promises"));
const path = __importStar(require("path"));
const supabase_js_1 = require("@supabase/supabase-js");
const ai_1 = require("./src/ai");
// --- Configuration ---
const MAIN_MODEL_PATH = './model_main';
const CHECKPOINT_PATH = './training_checkpoints';
const ARCHIVE_PATH = './training_checkpoints_archive';
const EVAL_INTERVAL_MS = 15 * 60 * 1000; // Check for new checkpoints every 15 minutes
const BOARD_SIZE = 19;
// --- Evaluation Parameters ---
const NUM_GAMES = 50; // Must be an even number
const MCTS_THINK_TIME = 1000; // 1 second per move for fast evaluation
const PROMOTION_THRESHOLD = 0.55; // Challenger must win >55% of games
// --- Supabase Configuration ---
const SUPABASE_URL = 'https://xkwgfidiposftwwasdqs.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhrd2dmaWRpcG9zZnR3d2FzZHFzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwODM3NzMsImV4cCI6MjA3MDY1OTc3M30.-9n_26ga07dXFiFOShP78_p9cEcIKBxHBEYJ1A1gaiE';
const supabase = (0, supabase_js_1.createClient)(SUPABASE_URL, SUPABASE_ANON_KEY);
// --- Helper Functions ---
async function loadModel(modelPath) {
    try {
        const model = await tf.loadLayersModel(`file://${path.resolve(modelPath)}/model.json`);
        return model;
    }
    catch (e) {
        console.error(`[Evaluator] Could not load model from ${modelPath}.`);
        return null;
    }
}
async function runGame(blackModel, whiteModel) {
    let board = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(null));
    let player = 'black';
    const players = { black: blackModel, white: whiteModel };
    for (let moveCount = 0; moveCount < BOARD_SIZE * BOARD_SIZE; moveCount++) {
        const currentModel = players[player];
        const { bestMove } = await (0, ai_1.findBestMoveNN)(currentModel, board, player, MCTS_THINK_TIME);
        if (!bestMove || bestMove[0] === -1)
            return null; // Draw
        board[bestMove[0]][bestMove[1]] = player;
        if ((0, ai_1.checkWin)(board, player, bestMove))
            return player; // Winner
        player = (0, ai_1.getOpponent)(player);
    }
    return null; // Draw
}
async function uploadModelToSupabase(modelDir) {
    console.log(`[Uploader] Starting upload of new model from ${modelDir} to Supabase Storage...`);
    try {
        const modelJsonContent = await fs.readFile(path.join(modelDir, 'model.json'), 'utf-8');
        const weightsBinContent = await fs.readFile(path.join(modelDir, 'weights.bin'));
        const { error: jsonError } = await supabase.storage
            .from('models')
            .upload('gomoku_model/model.json', modelJsonContent, { upsert: true });
        if (jsonError)
            throw new Error(`Failed to upload model.json: ${jsonError.message}`);
        const { error: weightsError } = await supabase.storage
            .from('models')
            .upload('gomoku_model/weights.bin', weightsBinContent, { upsert: true });
        if (weightsError)
            throw new Error(`Failed to upload weights.bin: ${weightsError.message}`);
        console.log('[Uploader] Successfully uploaded new model to Supabase Storage.');
    }
    catch (e) {
        console.error('[Uploader] An error occurred during upload:', e);
    }
}
async function evaluateCheckpoint(challengerPath) {
    console.log(`\n--- Evaluating new checkpoint: ${path.basename(challengerPath)} ---`);
    const championModel = await loadModel(MAIN_MODEL_PATH);
    const challengerModel = await loadModel(challengerPath);
    if (!challengerModel) {
        console.log('Could not load challenger model, skipping evaluation.');
        await fs.rename(challengerPath, path.join(ARCHIVE_PATH, path.basename(challengerPath)));
        return;
    }
    if (!championModel) {
        console.log('No champion model found. Promoting challenger automatically.');
        await fs.cp(challengerPath, MAIN_MODEL_PATH, { recursive: true });
        await uploadModelToSupabase(MAIN_MODEL_PATH);
        await fs.rename(challengerPath, path.join(ARCHIVE_PATH, path.basename(challengerPath)));
        return;
    }
    let challengerWins = 0;
    for (let i = 0; i < NUM_GAMES; i++) {
        console.log(`-- Starting evaluation game ${i + 1} / ${NUM_GAMES} --`);
        const challengerIsBlack = i % 2 === 0;
        const blackPlayer = challengerIsBlack ? challengerModel : championModel;
        const whitePlayer = challengerIsBlack ? championModel : challengerModel;
        const winner = await runGame(blackPlayer, whitePlayer);
        if (winner) {
            if ((challengerIsBlack && winner === 'black') || (!challengerIsBlack && winner === 'white')) {
                challengerWins++;
                console.log(`Game ${i + 1}: Challenger wins.`);
            }
            else {
                console.log(`Game ${i + 1}: Champion wins.`);
            }
        }
    }
    const winRate = challengerWins / NUM_GAMES;
    console.log(`\n--- Evaluation for ${path.basename(challengerPath)} Finished ---`);
    console.log(`Challenger Win Rate: ${(winRate * 100).toFixed(2)}%`);
    if (winRate > PROMOTION_THRESHOLD) {
        console.log(`PASSED! New model is stronger. Promoting to main model.`);
        const oldChampionPath = `${MAIN_MODEL_PATH}_archive_${Date.now()}`;
        await fs.rename(MAIN_MODEL_PATH, oldChampionPath);
        console.log(`Old champion model archived to ${oldChampionPath}`);
        await fs.cp(challengerPath, MAIN_MODEL_PATH, { recursive: true });
        console.log(`New champion model copied to ${MAIN_MODEL_PATH}`);
        await uploadModelToSupabase(MAIN_MODEL_PATH);
    }
    else {
        console.log(`FAILED. New model is not significantly stronger.`);
    }
    await fs.rename(challengerPath, path.join(ARCHIVE_PATH, path.basename(challengerPath)));
}
async function main() {
    console.log('--- Evaluation Worker Started ---');
    while (true) {
        try {
            const checkpoints = await fs.readdir(CHECKPOINT_PATH);
            const sortedCheckpoints = checkpoints.filter(f => f.startsWith('model_checkpoint')).sort();
            if (sortedCheckpoints.length > 0) {
                const checkpointToTest = sortedCheckpoints[0];
                const checkpointFullPath = path.join(CHECKPOINT_PATH, checkpointToTest);
                await evaluateCheckpoint(checkpointFullPath);
            }
            else {
                console.log('No new checkpoints found. Waiting...');
            }
        }
        catch (e) {
            console.error('[Evaluator] An error occurred in the main loop:', e);
        }
        await new Promise(resolve => setTimeout(resolve, EVAL_INTERVAL_MS));
    }
}
main();
