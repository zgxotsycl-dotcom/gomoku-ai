import * as tf from '@tensorflow/tfjs-node-gpu';
import { createDualResNetModel } from './model';
import * as fs from 'fs/promises';
import * as path from 'path';

const MODEL_SAVE_PATH = './gomoku_model';

async function createAndSaveInitialModel() {
    console.log('Creating a new, untrained "generation 0" model...');

    const model = createDualResNetModel();

    console.log(`Saving initial model to ${MODEL_SAVE_PATH}...`);
    try {
        // Ensure the directory exists, creating it if necessary.
        await fs.mkdir(MODEL_SAVE_PATH, { recursive: true });

        // tf.io.fileSystem is the standard way to save models in Node.js
        await model.save(`file://${path.resolve(MODEL_SAVE_PATH)}`);

        console.log('--- Initial Model Created Successfully! ---');
        console.log(`Model saved in: ${path.resolve(MODEL_SAVE_PATH)}`);

    } catch (error) {
        console.error("Failed to save the initial model:", error);
    }
}

createAndSaveInitialModel();
