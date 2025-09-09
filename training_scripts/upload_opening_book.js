const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

// This script reads your opening_book.json and uploads it to your Supabase database.

async function main() {
  // 1. Load Environment Variables
  // Make sure you have a .env file in this directory with your project URL and service role key.
  require('dotenv').config({ path: path.resolve(__dirname, './.env') });

  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!supabaseUrl || !supabaseKey) {
    console.error("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in a .env file in the training_scripts directory.");
    return;
  }

  // 2. Read the opening book data
  const filePath = path.resolve(__dirname, '../opening_book.json');
  if (!fs.existsSync(filePath)) {
    console.error(`Error: opening_book.json not found at ${filePath}`);
    return;
  }
  const openingBookData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  console.log(`Read ${openingBookData.length} entries from opening_book.json.`);

  // 3. Initialize Supabase client
  const supabase = createClient(supabaseUrl, supabaseKey);

  // 4. Invoke the import function
  console.log("Uploading data to Supabase...");
  const { data, error } = await supabase.functions.invoke('import-opening-book', {
    body: openingBookData
  });

  if (error) {
    console.error("Failed to import data:", error);
    return;
  }

  console.log("Successfully invoked the import function.");
  console.log("Response:", data);
}

main();
