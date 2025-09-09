/**
 * @file Manual Model Uploader
 * This script manually uploads the model from a specified directory to Supabase Storage.
 */

import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs/promises';
import * as path from 'path';

// --- Configuration ---
const MODEL_SOURCE_PATH = './model_main'; // Upload from the main model directory

// --- Supabase Configuration ---
const SUPABASE_URL = 'https://xkwgfidiposftwwasdqs.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhrd2dmaWRpcG9zZnR3d2FzZHFzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwODM3NzMsImV4cCI6MjA3MDY1OTc3M30.-9n_26ga07dXFiFOShP78_p9cEcIKBxHBEYJ1A1gaiE';
const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function uploadModel() {
    console.log(`Starting upload of model from ${MODEL_SOURCE_PATH} to Supabase Storage...`);
    try {
        const modelJsonPath = path.join(MODEL_SOURCE_PATH, 'model.json');
        const weightsBinPath = path.join(MODEL_SOURCE_PATH, 'weights.bin');

        console.log(`Reading ${modelJsonPath}...`);
        const modelJsonContent = await fs.readFile(modelJsonPath, 'utf-8');

        console.log(`Reading ${weightsBinPath}...`);
        const weightsBinContent = await fs.readFile(weightsBinPath);

        console.log('Uploading model.json...');
        const { error: jsonError } = await supabase.storage
            .from('models')
            .upload('gomoku_model/model.json', modelJsonContent, { upsert: true, contentType: 'application/json' });
        if (jsonError) throw new Error(`Failed to upload model.json: ${jsonError.message}`);

        console.log('Uploading weights.bin...');
        const { error: weightsError } = await supabase.storage
            .from('models')
            .upload('gomoku_model/weights.bin', weightsBinContent, { upsert: true, contentType: 'application/octet-stream' });
        if (weightsError) throw new Error(`Failed to upload weights.bin: ${weightsError.message}`);

        console.log('\n--- Upload Complete! ---');
        console.log('The model has been successfully uploaded to Supabase Storage.');

    } catch (e) {
        console.error('\n--- An error occurred during upload: ---', e);
    }
}

uploadModel();
