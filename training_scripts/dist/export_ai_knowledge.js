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
const supabase_js_1 = require("@supabase/supabase-js");
const fs = __importStar(require("fs/promises"));
// --- Configuration ---
const SUPABASE_URL = 'https://xkwgfidiposftwwasdqs.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhrd2dmaWRpcG9zZnR3d2FzZHFzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUwODM3NzMsImV4cCI6MjA3MDY1OTc3M30.-9n_26ga07dXFiFOShP78_p9cEcIKBxHBEYJ1A1gaiE';
const TABLE_NAME = 'ai_knowledge';
const OUTPUT_FILE_PATH = './ai_knowledge_export.csv';
const CHUNK_SIZE = 1000;
// --- Main Export Logic ---
const supabase = (0, supabase_js_1.createClient)(SUPABASE_URL, SUPABASE_ANON_KEY);
function jsonToCsv(data) {
    if (data.length === 0)
        return '';
    const csvRows = data.map(row => {
        return Object.values(row).map(value => {
            let stringValue = String(value);
            if (typeof value === 'object' && value !== null) {
                stringValue = JSON.stringify(value);
            }
            if (stringValue.includes('"') || stringValue.includes(',') || stringValue.includes('\n')) {
                return `"${stringValue.replace(/"/g, '""')}"`;
            }
            return stringValue;
        }).join(',');
    });
    return csvRows.join('\n') + '\n';
}
async function exportTable() {
    console.log(`Starting export of table "${TABLE_NAME}" to ${OUTPUT_FILE_PATH}...`);
    console.log(`Fetching data in chunks of ${CHUNK_SIZE} rows.`);
    let hasWrittenHeader = false;
    let currentIndex = 0;
    try {
        // Clear the file if it exists
        await fs.writeFile(OUTPUT_FILE_PATH, '');
        while (true) {
            const { data, error } = await supabase
                .from(TABLE_NAME)
                .select('*')
                .range(currentIndex, currentIndex + CHUNK_SIZE - 1);
            if (error) {
                console.error('Error fetching data:', error);
                break;
            }
            if (data && data.length > 0) {
                console.log(`Fetched ${data.length} rows from index ${currentIndex}...`);
                if (!hasWrittenHeader) {
                    const headers = Object.keys(data[0]).join(',') + '\n';
                    await fs.writeFile(OUTPUT_FILE_PATH, headers, { flag: 'a' });
                    hasWrittenHeader = true;
                }
                const csvString = jsonToCsv(data);
                await fs.writeFile(OUTPUT_FILE_PATH, csvString, { flag: 'a' });
                if (data.length < CHUNK_SIZE) {
                    console.log('All data has been fetched.');
                    break;
                }
                currentIndex += CHUNK_SIZE;
            }
            else {
                console.log('No more data to fetch.');
                break;
            }
        }
        console.log(`\nExport complete! Data saved to ${OUTPUT_FILE_PATH}`);
    }
    catch (e) {
        console.error("An error occurred during file writing:", e);
    }
}
exportTable();
