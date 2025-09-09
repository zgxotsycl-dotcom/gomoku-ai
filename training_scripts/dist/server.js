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
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const fastify_1 = __importDefault(require("fastify"));
const tf = __importStar(require("@tensorflow/tfjs-node-gpu"));
const path = __importStar(require("path"));
const ai_1 = require("./src/ai");
// --- Configuration ---
const MAIN_MODEL_PATH = './model_main';
const MCTS_THINK_TIME = 3000; // 3 seconds per move for the server
const PORT = 8080;
let model = null;
async function loadModel() {
    if (model)
        return model;
    console.log(`Loading model from ${MAIN_MODEL_PATH}...`);
    try {
        const loadedModel = await tf.loadLayersModel(`file://${path.resolve(MAIN_MODEL_PATH)}/model.json`);
        console.log('Model loaded successfully.');
        return loadedModel;
    }
    catch (e) {
        console.error(`Could not load model. Error: ${e}`);
        process.exit(1);
    }
}
// --- Server Setup ---
const server = (0, fastify_1.default)({ logger: true });
server.post('/get-move', async (request, reply) => {
    if (!model) {
        return reply.status(503).send({ error: 'AI model is not ready.' });
    }
    try {
        const { board, player } = request.body;
        if (!board || !player) {
            return reply.status(400).send({ error: 'Missing or invalid request body' });
        }
        const { bestMove } = await (0, ai_1.findBestMoveNN)(model, board, player, MCTS_THINK_TIME);
        return reply.send({ move: bestMove });
    }
    catch (e) {
        server.log.error(e);
        return reply.status(500).send({ error: 'An internal error occurred' });
    }
});
async function start() {
    try {
        model = await loadModel();
        // Listen on all network interfaces inside a container
        await server.listen({ port: PORT, host: '0.0.0.0' });
    }
    catch (err) {
        server.log.error(err);
        process.exit(1);
    }
}
start();
