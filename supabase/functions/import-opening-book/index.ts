import { createClient } from '@supabase/supabase-js'
import { corsHeaders } from '../_shared/cors.ts'

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const openingBookData = await req.json();

    if (!Array.isArray(openingBookData)) {
      throw new Error("Request body must be an array of opening book entries.");
    }

    const supabaseAdmin = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    // Upsert the data into the table. 
    // `board_hash` is the unique column, so it will update existing entries or insert new ones.
    const { error } = await supabaseAdmin
      .from('ai_opening_book')
      .upsert(openingBookData, { onConflict: 'board_hash' });

    if (error) {
      throw error;
    }

    console.log(`Successfully imported ${openingBookData.length} opening book entries.`);

    return new Response(JSON.stringify({ success: true, imported: openingBookData.length }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    });

  } catch (e) {
    console.error("Error importing opening book:", e);
    const errorMessage = e instanceof Error ? e.message : "An unknown error occurred.";
    return new Response(JSON.stringify({ error: errorMessage }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 400,
    });
  }
});
