"use strict";
/**
 * @file Manual Model Uploader
 * This script manually uploads the model from a specified directory to Supabase Storage.
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
const supabase_js_1 = require("@supabase/supabase-js");
const fs = __importStar(require("fs/promises"));
const path = __importStar(require("path"));
// --- Configuration ---
const MODEL_SOURCE_PATH = './model_main'; // Upload from the main model directory
// --- Supabase Configuration ---
const SUPABASE_URL = 'https://xkwgfidiposftwwasdqs.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhrd2dmaWRpcG9zZnR3d2FzZHFzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwODM3NzMsImV4cCI6MjA3MDY1OTc3M30.-9n_26ga07dXFiFOShP78_p9cEcIKBxHBEYJ1A1gaiE';
const supabase = (0, supabase_js_1.createClient)(SUPABASE_URL, SUPABASE_ANON_KEY);
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
        if (jsonError)
            throw new Error(`Failed to upload model.json: ${jsonError.message}`);
        console.log('Uploading weights.bin...');
        const { error: weightsError } = await supabase.storage
            .from('models')
            .upload('gomoku_model/weights.bin', weightsBinContent, { upsert: true, contentType: 'application/octet-stream' });
        if (weightsError)
            throw new Error(`Failed to upload weights.bin: ${weightsError.message}`);
        console.log('\n--- Upload Complete! ---');
        console.log('The model has been successfully uploaded to Supabase Storage.');
    }
    catch (e) {
        console.error('\n--- An error occurred during upload: ---', e);
    }
}
uploadModel();
