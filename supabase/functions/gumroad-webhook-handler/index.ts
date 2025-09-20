import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

// Config via env
const SUPPORTER_PRODUCT_ID = Deno.env.get('GUMROAD_PRODUCT_ID_SUPPORTER') || '';
const WEBHOOK_SECRET = Deno.env.get('GUMROAD_WEBHOOK_SECRET') || '';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type, x-gumroad-signature',
}

async function verifySignature(req: Request, raw: Uint8Array): Promise<boolean> {
  // Optional HMAC SHA-256 verification when Gumroad sends JSON + signature header
  if (!WEBHOOK_SECRET) return true; // no secret configured → skip
  const sig = req.headers.get('X-Gumroad-Signature') || req.headers.get('x-gumroad-signature');
  if (!sig) return true; // accept if header missing to avoid breaking existing setups
  try {
    const key = await crypto.subtle.importKey(
      'raw',
      new TextEncoder().encode(WEBHOOK_SECRET),
      { name: 'HMAC', hash: 'SHA-256' },
      false,
      ['sign']
    );
    const mac = await crypto.subtle.sign('HMAC', key, raw);
    const hex = Array.from(new Uint8Array(mac)).map(b => b.toString(16).padStart(2, '0')).join('');
    // Gumroad provides lowercase hex
    return sig.trim().toLowerCase() === hex;
  } catch (_) {
    return false;
  }
}

function setNested(obj: Record<string, any>, path: string, value: any) {
  // Supports keys like purchase[product][id] and url_params[user_id]
  const parts: string[] = [];
  const re = /\[([^\]]+)\]/g;
  const base = path.split('[')[0];
  if (base) parts.push(base);
  let m: RegExpExecArray | null;
  while ((m = re.exec(path)) !== null) parts.push(m[1]);
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    const p = parts[i];
    if (typeof cur[p] !== 'object' || cur[p] === null) cur[p] = {};
    cur = cur[p];
  }
  cur[parts[parts.length - 1]] = value;
}

async function parseBody(req: Request): Promise<{ payload: any; raw: Uint8Array }> {
  // Support both JSON and form-url-encoded payloads
  const ct = (req.headers.get('content-type') || '').toLowerCase();
  const raw = new Uint8Array(await req.arrayBuffer());
  if (ct.includes('application/json')) {
    try {
      return { payload: JSON.parse(new TextDecoder().decode(raw)), raw };
    } catch (e) {
      throw new Error('Invalid JSON payload');
    }
  }
  if (ct.includes('application/x-www-form-urlencoded')) {
    const txt = new TextDecoder().decode(raw);
    const params = new URLSearchParams(txt);
    const obj: Record<string, any> = {};
    for (const [k, v] of params.entries()) {
      if (k.includes('[')) setNested(obj, k, v);
      else obj[k] = v;
    }
    return { payload: obj, raw };
  }
  // Fallback: try JSON
  return { payload: await req.json().catch(() => ({})), raw };
}

function extractProductIdentity(payload: any): { id?: string; permalink?: string } {
  // Try multiple locations for product id/permalink
  const cand: any[] = [payload, payload.product, payload.purchase, payload.purchase?.product, payload.sale, payload.sale?.product];
  let id: string | undefined;
  let permalink: string | undefined;
  for (const c of cand) {
    if (!c) continue;
    if (!id && typeof c.product_id === 'string' && c.product_id) id = c.product_id;
    if (!id && typeof c.id === 'string' && c.id) id = c.id;
    if (!permalink && typeof c.product_permalink === 'string' && c.product_permalink) permalink = c.product_permalink;
    if (!permalink && typeof c.permalink === 'string' && c.permalink) permalink = c.permalink;
  }
  return { id, permalink };
}

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Parse body as raw to support signature verification
    const { payload, raw } = await parseBody(req);
    if (!(await verifySignature(req, raw))) {
      return new Response('Invalid signature', { status: 401, headers: corsHeaders });
    }

    // Filter by product (id or permalink)
    const { id: productId, permalink } = extractProductIdentity(payload);
    const EXPECT_ID = SUPPORTER_PRODUCT_ID || '';
    const EXPECT_PERMALINK = Deno.env.get('GUMROAD_PRODUCT_PERMALINK') || '';
    const matchId = EXPECT_ID && productId ? productId === EXPECT_ID : true; // if EXPECT_ID unset, skip check
    const matchPermalink = EXPECT_PERMALINK && permalink ? permalink === EXPECT_PERMALINK : true;
    if (!(matchId && matchPermalink)) {
      return new Response('Irrelevant product', { status: 200, headers: corsHeaders });
    }

    // Identify user: prefer url_params.user_id, fallback to buyer email → auth.users via RPC
    let userId: string | null = payload.url_params?.user_id || null;
    const buyerEmail: string | null = payload.purchaser?.email || payload.email || null;

    const supabaseAdmin = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    );

    if (!userId && buyerEmail) {
      // map email to auth.users.id via RPC (function provided in migrations)
      const { data, error } = await supabaseAdmin.rpc('get_user_id_by_email', { user_email: buyerEmail });
      if (error) throw new Error(`Failed to map email to user id: ${error.message}`);
      if (data) userId = String(data);
    }

    if (!userId) {
      throw new Error('Could not resolve user id (missing url_params.user_id and email mapping failed)');
    }

    // Grant supporter flag (idempotent)
    const { error: upErr } = await supabaseAdmin
      .from('profiles')
      .update({ is_supporter: true })
      .eq('id', userId);
    if (upErr) throw new Error(`Failed to update profile for user ${userId}: ${upErr.message}`);

    return new Response(JSON.stringify({ ok: true, userId }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    });

  } catch (e) {
    console.error('Gumroad webhook handler error:', e);
    const msg = e instanceof Error ? e.message : 'Unknown error';
    return new Response(JSON.stringify({ ok: false, error: msg }), {
      status: 400,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
});
