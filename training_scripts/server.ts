import fastify from 'fastify';
import * as tf from '@tensorflow/tfjs-node-gpu';
import * as path from 'path';
import * as chokidar from 'chokidar';
import { findBestMoveNN } from './src/ai';
import type { Player } from './src/ai';

// --- Configuration ---
const MAIN_MODEL_PATH = './model_main';
const MCTS_THINK_TIME = 15000; // 15 seconds, since we are on a dedicated server
const PORT = 8080;

let model: tf.LayersModel | null = null;

// --- Model Loading with Hot-Reload ---

async function loadModel(): Promise<tf.LayersModel> {
    console.log(`Loading model from ${MAIN_MODEL_PATH}...`);
    try {
        const loadedModel = await tf.loadLayersModel(`file://${path.resolve(MAIN_MODEL_PATH)}/model.json`);
        console.log('Model loaded successfully.');
        return loadedModel;
    } catch (e) {
        console.error(`Could not load model. Error: ${e}`);
        throw e;
    }
}

function setupModelWatcher() {
    console.log(`Watching for model changes in ${MAIN_MODEL_PATH}`);
    const watcher = chokidar.watch(path.resolve(MAIN_MODEL_PATH), {
        ignoreInitial: true,
        persistent: true,
        awaitWriteFinish: {
            stabilityThreshold: 2000,
            pollInterval: 100
        }
    });

    watcher.on('change', async (filePath) => {
        console.log(`Detected model file change: ${filePath}`);
        console.log('Attempting to hot-reload the model...');
        try {
            const newModel = await loadModel();
            model = newModel; // Atomically swap to the new model
            console.log('--- Model hot-reload successful! ---');
        } catch (e) {
            console.error('--- Model hot-reload failed. Keeping the old model. Error: ---', e);
        }
    });
}

// --- Server Setup ---

const server = fastify({ logger: true });

server.post('/get-move', async (request, reply) => {
    if (!model) {
        return reply.status(503).send({ error: 'AI model is not ready.' });
    }

    try {
        const { board, player, moves } = request.body as { board: (Player | null)[][], player: Player, moves: any[] };
        if (!board || !player || !moves) {
            return reply.status(400).send({ error: 'Missing or invalid request body' });
        }

        const { bestMove } = await findBestMoveNN(model, board, player, MCTS_THINK_TIME);
        return reply.send({ move: bestMove });

    } catch (e) {
        server.log.error(e);
        return reply.status(500).send({ error: 'An internal error occurred' });
    }
});

async function start() {
    try {
        model = await loadModel();
        setupModelWatcher(); // Start watching for file changes
        await server.listen({ port: PORT, host: '0.0.0.0' });
    } catch (err) {
        server.log.error(err);
        process.exit(1);
    }
}

start();