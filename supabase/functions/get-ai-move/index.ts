import { createClient } from '@supabase/supabase-js';
import { corsHeaders } from '../_shared/cors.ts';
import type { Player } from '../_shared/ai.ts';

// This Edge Function proxies to external AI server (Node) for inference.
const AI_SERVER_URL = Deno.env.get('AI_SERVER_URL') ?? 'https://ai.omokk.com/get-move';

const boardToString = (board: (Player | null)[][]) => board.map(row => row.map(cell => cell ? cell[0] : '-').join('')).join('|');

// --- Symmetry helpers for opening book ---
type TransformId = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7;
const INV_T: Record<TransformId, TransformId> = { 0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7 };
function transformRC(r: number, c: number, size: number, t: TransformId): [number, number] {
  const n = size;
  switch (t) {
    case 0: return [r, c];
    case 1: return [c, n - 1 - r];
    case 2: return [n - 1 - r, n - 1 - c];
    case 3: return [n - 1 - c, r];
    case 4: return [r, n - 1 - c];
    case 5: return [n - 1 - r, c];
    case 6: return [c, r];
    case 7: return [n - 1 - c, n - 1 - r];
  }
}
function transformBoardHash(hash: string, t: TransformId): string {
  const rows = hash.split('|');
  const n = rows.length;
  const out: string[] = Array(n).fill('');
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const [rr, cc] = transformRC(r, c, n, t);
      const ch = rows[r][c];
      const row = out[rr] || ''.padEnd(n, ' ');
      out[rr] = (row.substring(0, cc) + ch + row.substring(cc + 1)).padEnd(n, ' ');
    }
  }
  return out.map(s => s.trim().padEnd(n, '-').substring(0, n)).join('|');
}
function getSymmetricHashes(board: (Player | null)[][]): Array<{ hash: string; t: TransformId }> {
  const base = boardToString(board);
  const trans: TransformId[] = [0,1,2,3,4,5,6,7];
  return trans.map(t => ({ hash: transformBoardHash(base, t), t }));
}

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: corsHeaders });

  try {
    const { board, player, moves = [], turnEndsAt, timeLeftMs, turnLimitMs } = await req.json();
    if (!board || !player) throw new Error("Missing 'board' or 'player' in request body.");

    const supabaseAdmin = createClient(Deno.env.get('SUPABASE_URL') ?? '', Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? '', { auth: { persistSession: false } });

    // Use opening book for the first 12 moves (symmetry-aware)
    if (moves.length <= 12) {
      const cands = getSymmetricHashes(board);
      const hashes = cands.map(c => c.hash);
      const { data: rows, error: bookErr } = await supabaseAdmin.from('ai_opening_book').select('board_hash,best_move').in('board_hash', hashes).limit(1);
      if (!bookErr && rows && rows.length > 0) {
        const row = rows[0] as { board_hash: string; best_move: [number, number] };
        const match = cands.find(c => c.hash === row.board_hash);
        let mv = row.best_move as [number, number];
        if (match) {
          const invT = INV_T[match.t];
          const [r, c] = transformRC(mv[0], mv[1], board.length, invT);
          mv = [r, c];
        }
        if (board[mv[0]]?.[mv[1]] === null) {
          console.log("Found move in Opening Book(sym):", mv);
          return new Response(JSON.stringify({ move: mv }), { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 });
        }
      }
    }

    // Proxy to external AI server, include timing hints if available
    const rsp = await fetch(AI_SERVER_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ board, player, moves, turnEndsAt, timeLeftMs, turnLimitMs }) });
    if (!rsp.ok) {
      const txt = await rsp.text();
      throw new Error(`AI server error: ${rsp.status} ${txt}`);
    }
    const data = await rsp.json();
    return new Response(JSON.stringify(data), { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : "Unknown error in get-ai-move.";
    console.error("Error in get-ai-move function:", errorMessage);
    return new Response(JSON.stringify({ error: errorMessage }), { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500 });
  }
});
