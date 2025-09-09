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
const model_1 = require("./model");
const fs = __importStar(require("fs/promises"));
const path = __importStar(require("path"));
const MODEL_SAVE_PATH = './gomoku_model';
async function createAndSaveInitialModel() {
    console.log('Creating a new, untrained "generation 0" model...');
    const model = (0, model_1.createDualResNetModel)();
    console.log(`Saving initial model to ${MODEL_SAVE_PATH}...`);
    try {
        // Ensure the directory exists, creating it if necessary.
        await fs.mkdir(MODEL_SAVE_PATH, { recursive: true });
        // tf.io.fileSystem is the standard way to save models in Node.js
        await model.save(`file://${path.resolve(MODEL_SAVE_PATH)}`);
        console.log('--- Initial Model Created Successfully! ---');
        console.log(`Model saved in: ${path.resolve(MODEL_SAVE_PATH)}`);
    }
    catch (error) {
        console.error("Failed to save the initial model:", error);
    }
}
createAndSaveInitialModel();
