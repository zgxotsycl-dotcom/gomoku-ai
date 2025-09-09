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
const tf = __importStar(require("@tensorflow/tfjs-node-gpu"));
const fs = __importStar(require("node:fs"));
const model_1 = require("./src/model");
const DATA_FILE_PATH = './training_data.jsonl';
const MODEL_SAVE_PATH = './gomoku_model';
const BOARD_SIZE = 19;
// Training Hyperparameters
const EPOCHS = 10;
const BATCH_SIZE = 32;
const VALIDATION_SPLIT = 0.1; // Use 10% of the data for validation
// This function is now included directly in this file.
function boardToChannels(board, player) {
    const opponent = player === 'black' ? 'white' : 'black';
    const playerChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));
    const opponentChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(0));
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            if (board[r][c] === player) {
                playerChannel[r][c] = 1;
            }
            else if (board[r][c] === opponent) {
                opponentChannel[r][c] = 1;
            }
        }
    }
    const colorChannelValue = player === 'black' ? 1 : 0;
    const colorChannel = Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(colorChannelValue));
    return [tf.tensor2d(playerChannel), tf.tensor2d(opponentChannel), tf.tensor2d(colorChannel)];
}
async function augmentAndProcessData() {
    console.log(`Loading training data from ${DATA_FILE_PATH}...`);
    let fileContent;
    try {
        fileContent = fs.readFileSync(DATA_FILE_PATH, 'utf-8');
    }
    catch (error) {
        if (error.code === 'ENOENT') {
            console.error(`Error: Training data file not found at ${DATA_FILE_PATH}`);
            return null;
        }
        throw error;
    }
    const lines = fileContent.trim().split('\n');
    if (lines.length === 0 || (lines.length === 1 && lines[0].trim() === '')) {
        console.error('Error: Training data file is empty.');
        return null;
    }
    const samples = lines.map((line) => JSON.parse(line));
    console.log(`Loaded ${samples.length} original training samples.`);
    console.log('Augmenting data (x8) and converting to tensors...');
    const augmentedStates = [];
    const augmentedPolicies = [];
    const augmentedValues = [];
    for (const sample of samples) {
        const symmetries = getSymmetries(sample.state, sample.policy);
        for (const sym of symmetries) {
            const player = sample.player || 'black';
            const [playerChannel, opponentChannel, colorChannel] = boardToChannels(sym.state, player);
            const stackedChannels = tf.stack([playerChannel, opponentChannel, colorChannel], 2);
            augmentedStates.push(stackedChannels.arraySync());
            augmentedPolicies.push(sym.policy);
            augmentedValues.push(sample.value);
            tf.dispose([playerChannel, opponentChannel, colorChannel, stackedChannels]);
        }
    }
    console.log(`Created ${augmentedStates.length} augmented samples.`);
    const combined = augmentedStates.map((state, i) => ({
        state,
        policy: augmentedPolicies[i],
        value: augmentedValues[i]
    }));
    tf.util.shuffle(combined);
    const finalStates = [];
    const finalPolicies = [];
    const finalValues = [];
    combined.forEach(item => {
        finalStates.push(item.state);
        finalPolicies.push(item.policy);
        finalValues.push(item.value);
    });
    const xs = tf.tensor4d(finalStates);
    const policyTensor = tf.tensor2d(finalPolicies);
    const valueTensor = tf.tensor2d(finalValues, [finalValues.length, 1]);
    console.log('Tensors created successfully.');
    return { xs, ys: { 'policy': policyTensor, 'value': valueTensor } };
}
function getSymmetries(state, policy) {
    const symmetries = [];
    let currentBoard = state.map(row => [...row]);
    let currentPolicy = [...policy];
    for (let i = 0; i < 4; i++) {
        symmetries.push({ state: currentBoard, policy: currentPolicy });
        symmetries.push({ state: flipBoard(currentBoard), policy: flipPolicy(currentPolicy) });
        currentBoard = rotateBoard(currentBoard);
        currentPolicy = rotatePolicy(currentPolicy);
    }
    return symmetries;
}
function rotateBoard(board) {
    const newBoard = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(null));
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            newBoard[c][BOARD_SIZE - 1 - r] = board[r][c];
        }
    }
    return newBoard;
}
function flipBoard(board) {
    return board.map(row => row.slice().reverse());
}
function rotatePolicy(policy) {
    const newPolicy = Array(BOARD_SIZE * BOARD_SIZE).fill(0);
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            newPolicy[c * BOARD_SIZE + (BOARD_SIZE - 1 - r)] = policy[r * BOARD_SIZE + c];
        }
    }
    return newPolicy;
}
function flipPolicy(policy) {
    const newPolicy = Array(BOARD_SIZE * BOARD_SIZE).fill(0);
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            newPolicy[r * BOARD_SIZE + (BOARD_SIZE - 1 - c)] = policy[r * BOARD_SIZE + c];
        }
    }
    return newPolicy;
}
async function train() {
    const data = await augmentAndProcessData();
    if (!data) {
        return;
    }
    const { xs, ys } = data;
    let model;
    const modelPath = `file://${MODEL_SAVE_PATH}/model.json`;
    try {
        console.log(`Attempting to load existing model from ${MODEL_SAVE_PATH}...`);
        model = await tf.loadLayersModel(modelPath);
        model.compile({
            optimizer: tf.train.adam(),
            loss: { 'policy': 'categoricalCrossentropy', 'value': 'meanSquaredError' },
            metrics: { 'policy': 'accuracy', 'value': 'meanAbsoluteError' }
        });
        console.log('Existing model loaded and re-compiled successfully.');
    }
    catch (error) {
        console.log('No existing model found. Creating a new model...');
        model = (0, model_1.createDualResNetModel)();
    }
    console.log('Starting model training...');
    await model.fit(xs, ys, {
        epochs: EPOCHS,
        batchSize: BATCH_SIZE,
        shuffle: true,
        validationSplit: VALIDATION_SPLIT,
        callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 3 })
    });
    console.log('Model training finished.');
    console.log(`Saving model to ${MODEL_SAVE_PATH}...`);
    await model.save(`file://${MODEL_SAVE_PATH}`);
    console.log('Model saved successfully.');
    xs.dispose();
    ys.policy.dispose();
    ys.value.dispose();
}
train().catch(err => {
    console.error('An unexpected error occurred during training:', err);
});
