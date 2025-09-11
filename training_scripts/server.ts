import fastify, { FastifyRequest, FastifyReply } from 'fastify';
import cors from '@fastify/cors';
import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs/promises';
import * as path from 'path';
import { createClient } from '@supabase/supabase-js';
import { findBestMoveNN } from './src/ai';
import type { Player } from './src/ai';

// --- Configuration ---
const MODEL_DIR = './model_gcp';
const MCTS_THINK_TIME = 15000;
const PORT = 8080;
const MODEL_CHECK_INTERVAL_MS = 5 * 60 * 1000;

// --- Supabase Configuration ---
const SUPABASE_URL = 'https://xkwgfidiposftwwasdqs.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhrd2dmaWRpcG9zZnR3d2FzZHFzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwODM3NzMsImV4cCI6MjA3MDY1OTc3M30.-9n_26ga07dXFiFOShP78_p9cEcIKBxHBEYJ1A1gaiE';
const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

let model: tf.LayersModel | null = null;
let currentModelTimestamp: string | null = null;

// --- Model Loading and Auto-Update ---

async function downloadAndLoadModel(): Promise<boolean> {
    console.log('[Model Syncer] Downloading latest model from Supabase...');
    try {
        await fs.mkdir(MODEL_DIR, { recursive: true });

        const { data: jsonBlob, error: jsonError } = await supabase.storage.from('models').download('gomoku_model/model.json');
        if (jsonError) throw jsonError;
        await fs.writeFile(path.join(MODEL_DIR, 'model.json'), Buffer.from(await jsonBlob.arrayBuffer()));

        const { data: weightsBlob, error: weightsError } = await supabase.storage.from('models').download('gomoku_model/weights.bin');
        if (weightsError) throw weightsError;
        await fs.writeFile(path.join(MODEL_DIR, 'weights.bin'), Buffer.from(await weightsBlob.arrayBuffer()));

        console.log('[Model Syncer] Loading model into memory...');
        model = await tf.loadLayersModel(`file://${path.resolve(MODEL_DIR)}/model.json`);
        console.log('[Model Syncer] New model loaded successfully!');
        return true;
    } catch (e) {
        console.error('[Model Syncer] Failed to download or load model:', e);
        return false;
    }
}

async function checkForNewModel() {
    console.log('[Model Syncer] Checking for new model version in Supabase...');
    try {
        const { data, error } = await supabase.storage.from('models').list('gomoku_model', { limit: 1, offset: 0, sortBy: { column: 'updated_at', order: 'desc' } });
        if (error) throw error;
        if (data && data.length > 0) {
            const latestTimestamp = data[0].updated_at;
            if (!currentModelTimestamp || latestTimestamp > currentModelTimestamp) {
                console.log(`[Model Syncer] New version detected! (New: ${latestTimestamp})`);
                const success = await downloadAndLoadModel();
                if (success) currentModelTimestamp = latestTimestamp;
            } else {
                console.log('[Model Syncer] No new model detected.');
            }
        }
    } catch (e) {
        console.error('[Model Syncer] Error checking for new model:', e);
    }
}

// --- Server Setup ---

const server = fastify({ logger: true });
server.register(cors, { origin: "*" });

interface GetMoveRequestBody { board: (Player | null)[][]; player: Player; moves: any[]; }

server.post('/get-move', async (request: FastifyRequest<{ Body: GetMoveRequestBody }>, reply: FastifyReply) => {
    if (!model) return reply.status(503).send({ error: 'AI model is not ready or still loading.' });
    try {
        const { board, player, moves } = request.body;
        if (!board || !player || !moves) return reply.status(400).send({ error: 'Missing request body' });
        const { bestMove } = await findBestMoveNN(model, board, player, MCTS_THINK_TIME);
        return reply.send({ move: bestMove });
    } catch (e: any) {
        server.log.error(e, 'Error during get-move');
        return reply.status(500).send({ error: 'Internal error' });
    }
});

async function start() {
    try {
        await checkForNewModel();
        setInterval(checkForNewModel, MODEL_CHECK_INTERVAL_MS);
        await server.listen({ port: PORT, host: '0.0.0.0' });
    } catch (err: any) {
        server.log.error(err, 'Server startup error');
        process.exit(1);
    }
}

start();
