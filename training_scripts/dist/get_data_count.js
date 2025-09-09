"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const dotenv_1 = require("dotenv");
const supabase_js_1 = require("@supabase/supabase-js");
(0, dotenv_1.config)();
async function getDataCount() {
    console.log("Connecting to database to count knowledge data...");
    const supabaseUrl = process.env.SUPABASE_URL;
    const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
    if (!supabaseUrl || !supabaseServiceKey) {
        console.error("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in your .env file.");
        return;
    }
    try {
        const supabase = (0, supabase_js_1.createClient)(supabaseUrl, supabaseServiceKey);
        const { count, error } = await supabase
            .from('ai_knowledge')
            .select('*', { count: 'exact', head: true });
        if (error) {
            throw error;
        }
        console.log("--- Data Count ---");
        console.log(`The AI has learned ${count} unique board patterns.`);
        console.log("------------------");
    }
    catch (e) {
        const errorMessage = e instanceof Error ? e.message : String(e);
        console.error("An error occurred while fetching data count:", errorMessage);
    }
}
getDataCount();
