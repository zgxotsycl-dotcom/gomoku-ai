import fastify from 'fastify';
import * as tf from '@tensorflow/tfjs-node-gpu';
import * as path from 'path';
import { findBestMoveNN } from './src/ai';
import type { Player } from './src/ai';

// --- Configuration ---
const MAIN_MODEL_PATH = './model_main';
const MCTS_THINK_TIME = 3000; // 3 seconds per move for the server
const PORT = 8080;

let model: tf.LayersModel | null = null;

async function loadModel(): Promise<tf.LayersModel> {
    if (model) return model;
    console.log(`Loading model from ${MAIN_MODEL_PATH}...`);
    try {
        const loadedModel = await tf.loadLayersModel(`file://${path.resolve(MAIN_MODEL_PATH)}/model.json`);
        console.log('Model loaded successfully.');
        return loadedModel;
    } catch (e) {
        console.error(`Could not load model. Error: ${e}`);
        process.exit(1);
    }
}

// --- Server Setup ---

const server = fastify({ logger: true });

server.post('/get-move', async (request, reply) => {
    if (!model) {
        return reply.status(503).send({ error: 'AI model is not ready.' });
    }

    try {
        const { board, player } = request.body as { board: (Player | null)[][], player: Player };
        if (!board || !player) {
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
        // Listen on all network interfaces inside a container
        await server.listen({ port: PORT, host: '0.0.0.0' });
    } catch (err) {
        server.log.error(err);
        process.exit(1);
    }
}

start();
