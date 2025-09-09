import { createClient } from '@supabase/supabase-js';
import { corsHeaders } from '../_shared/cors.ts';
import { findBestMoveNN, Player } from '../_shared/ai.ts';
import * as tf from '@tensorflow/tfjs-node';

// --- Model Loading ---
const MODEL_URL = 'https://xkwgfidiposftwwasdqs.supabase.co/storage/v1/object/public/models/gomoku_model/model.json';
let model: tf.LayersModel | null = null;

async function loadModel() {
    if (model) return model;
    // Always load from the official URL. Local fallback is removed.
    console.log(`Loading model from ${MODEL_URL}...`);
    model = await tf.loadLayersModel(MODEL_URL);
    console.log("Model loaded successfully from URL.");
    return model;
}

// Pre-load the model when the function instance starts.
const modelLoadPromise = loadModel();

const boardToString = (board: (Player | null)[][]) => board.map(row => row.map(cell => cell ? cell[0] : '-').join('')).join('|');

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: corsHeaders });

  try {
    const { board, player, moves } = await req.json();
    if (!board || !player) throw new Error("Missing 'board' or 'player' in request body.");

    const loadedModel = await modelLoadPromise;
    if (!loadedModel) {
        throw new Error("AI model is not available. Cannot process the request.");
    }

    const supabaseAdmin = createClient(Deno.env.get('SUPABASE_URL') ?? '', Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '', { auth: { persistSession: false } });

    // Use opening book for the first 12 moves
    if (moves.length <= 12) {
      const boardHash = boardToString(board);
      const { data: bookMove } = await supabaseAdmin.from('ai_opening_book').select('best_move').eq('board_hash', boardHash).single();
      if (bookMove?.best_move && board[bookMove.best_move[0]][bookMove.best_move[1]] === null) {
          console.log("Found move in Opening Book:", bookMove.best_move);
          return new Response(JSON.stringify({ move: bookMove.best_move }), { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 });
      }
    }

    // Increased thinking time for stronger play
    const EARLY_GAME_MOVES = 6;
    const EARLY_GAME_THINK_TIME = 5000;  // 5 seconds
    const MID_GAME_THINK_TIME = 15000; // 15 seconds
    const LATE_GAME_THINK_TIME = 10000; // 10 seconds (to avoid timeout in complex endgames)
    
    let thinkTime;
    if (moves.length <= EARLY_GAME_MOVES) {
        thinkTime = EARLY_GAME_THINK_TIME;
    } else if (moves.length <= 30) { // Mid-game
        thinkTime = MID_GAME_THINK_TIME;
    } else { // Late-game
        thinkTime = LATE_GAME_THINK_TIME;
    }

    console.log(`Calculating move for ${player}. Using NN-MCTS with Time Limit: ${thinkTime}ms...`);
    
    const { bestMove } = await findBestMoveNN(loadedModel, board, player as Player, thinkTime);
    
    console.log(`NN-MCTS calculation complete. Best move:`, bestMove);

    return new Response(JSON.stringify({ move: bestMove }), { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error in get-ai-move.";
    console.error("Error in get-ai-move function:", errorMessage);
    return new Response(JSON.stringify({ error: errorMessage }), { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500 });
  }
});