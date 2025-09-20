import { createClient } from '@supabase/supabase-js'
import { corsHeaders } from '../_shared/cors.ts'

type TransformId = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7;
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
function hashToBoard(hash: string): string[][] {
  const rows = hash.split('|');
  return rows.map(row => row.split(''));
}
function boardToHash(b: string[][]): string {
  return b.map(row => row.join('')).join('|');
}
function transformBoardHash(hash: string, t: TransformId): string {
  const rows = hashToBoard(hash);
  const n = rows.length;
  const out: string[][] = Array.from({ length: n }, () => Array(n).fill('-'));
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const [rr, cc] = transformRC(r, c, n, t);
    out[rr][cc] = rows[r][c];
  }
  return boardToHash(out);
}
function canonicalizeHash(hash: string): { hash: string; t: TransformId } {
  let best = '';
  let bestT: TransformId = 0;
  let first = true;
  for (const t of [0,1,2,3,4,5,6,7] as TransformId[]) {
    const h = transformBoardHash(hash, t);
    if (first || h < best) { best = h; bestT = t; first = false; }
  }
  return { hash: best, t: bestT };
}

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

    // Canonicalize by symmetry before upsert, so the table stores canonical hashes
    const canon = (openingBookData as Array<{ board_hash: string; best_move: [number, number] }>).map(e => {
      const { hash, t } = canonicalizeHash(e.board_hash);
      const size = Math.sqrt(hash.split('|').join('').length);
      const [r, c] = transformRC(e.best_move[0], e.best_move[1], size, t);
      return { board_hash: hash, best_move: [r, c] };
    });

    const { error } = await supabaseAdmin
      .from('ai_opening_book')
      .upsert(canon, { onConflict: 'board_hash' });

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
