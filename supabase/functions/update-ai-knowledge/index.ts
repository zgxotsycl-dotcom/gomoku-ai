import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.0.0'

// Define common headers for CORS
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

// Define the type for the incoming request body
interface RequestBody {
  patterns: string[];
  gameWinner: 'black' | 'white';
  aiPlayer: 'black' | 'white';
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { patterns, gameWinner, aiPlayer }: RequestBody = await req.json();

    // Improved validation
    if (!patterns || !Array.isArray(patterns) || !gameWinner || !aiPlayer) {
      return new Response(JSON.stringify({ error: 'Missing or invalid required fields' }), { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 400 
      });
    }

    // Create a Supabase admin client to bypass RLS
    const supabaseAdminClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );
    
    const didAiWin = gameWinner === aiPlayer;

    // Call the efficient RPC function once with the entire array
    const { error } = await supabaseAdminClient.rpc('update_multiple_patterns_knowledge', {
      p_pattern_hashes: patterns,
      p_wins_increment: didAiWin ? 1 : 0,
      p_losses_increment: didAiWin ? 0 : 1
    });

    if (error) {
      console.error(`Error updating multiple patterns:`, error);
      throw error; // Let the main catch block handle the response
    }

    return new Response(JSON.stringify({ success: true }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    });

  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
    console.error('General error in edge function:', errorMessage);
    return new Response(JSON.stringify({ error: errorMessage }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 500,
    });
  }
})