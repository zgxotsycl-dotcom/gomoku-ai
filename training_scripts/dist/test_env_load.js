"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const dotenv_1 = require("dotenv");
(0, dotenv_1.config)();
console.log("Attempting to load .env file...");
const supabaseUrl = process.env.SUPABASE_URL;
if (supabaseUrl) {
    console.log(".env file loaded successfully.");
    console.log("SUPABASE_URL is:", supabaseUrl);
}
else {
    console.error("Failed to load SUPABASE_URL from .env file.");
}
