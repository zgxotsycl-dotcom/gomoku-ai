import { createClient } from 'https://esm.sh/@supabase/supabase-js@2.0.0'

// Add the CORS headers
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface RequestBody {
  winningPatterns: string[];
  losingPatterns: string[];
}

Deno.serve(async (req) => {
  // Handle CORS preflight request
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders });
  }

  if (req.method !== 'POST') {
    return new Response('Method Not Allowed', { status: 405, headers: corsHeaders });
  }

  try {
    const { winningPatterns, losingPatterns }: RequestBody = await req.json();

    if (!winningPatterns || !losingPatterns) {
      return new Response('Missing required fields', { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
    }

    const supabaseAdminClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    // Update winning patterns
    for (const hash of winningPatterns) {
      const { error } = await supabaseAdminClient.rpc('update_pattern_knowledge', {
        p_pattern_hash: hash,
        p_wins_increment: 1,
        p_losses_increment: 0
      });
      if (error) console.error(`Error updating winning pattern ${hash}:`, error.message);
    }

    // Update losing patterns
    for (const hash of losingPatterns) {
      const { error } = await supabaseAdminClient.rpc('update_pattern_knowledge', {
        p_pattern_hash: hash,
        p_wins_increment: 0,
        p_losses_increment: 1
      });
      if (error) console.error(`Error updating losing pattern ${hash}:`, error.message);
    }

    return new Response(JSON.stringify({ success: true }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    });

  } catch (error) {
  const errorMessage = error instanceof Error
    ? error.message : 'An unknown error occurred';
  console.error('General error in edge function:', errorMessage);
  return new Response(JSON.stringify({ error: errorMessage }), {
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    status: 500,
  });
}
})