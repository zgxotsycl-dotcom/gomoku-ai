import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import fastify from 'fastify';
import { getStatusPath, updateStatus } from './status';
import { updateMany as updateTuning, getNumber as getTuned, clamp as clampNum } from './tuning';

console.log('<<<<< START_PIPELINE (Self-Play + Optional Upload) >>>>>');

const DIST_DIR = path.resolve(__dirname);
const SELF_PLAY_SCRIPT = path.join(DIST_DIR, 'self_play_trainer.js');
const UPLOAD_SCRIPT = path.join(DIST_DIR, 'upload_model.js');
const DISTILL_SCRIPT = path.join(DIST_DIR, 'distill_student.js');
const BUILD_BOOK_SCRIPT = path.join(DIST_DIR, 'build_opening_book.js');
const IMPORT_BOOK_SCRIPT = path.join(DIST_DIR, 'import_opening_book.js');
const ARENA_SCRIPT = path.join(DIST_DIR, 'arena.js');

async function notify(stage: string, status: 'start' | 'success' | 'error', extra?: any) {
  const url = process.env.WEBHOOK_URL;
  if (!url) return;
  try {
    await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ stage, status, extra, ts: new Date().toISOString() }) });
  } catch {}
}

function runScript(scriptPath: string, name: string): Promise<void> {
  return new Promise((resolve, reject) => {
    console.log(`\n----- Starting: ${name} -----`);
    const nodeProcess = spawn('node', [scriptPath], { stdio: 'inherit' });
    nodeProcess.on('close', (code) => {
      if (code !== 0) reject(new Error(`${name} failed with code ${code}`));
      else resolve();
    });
    nodeProcess.on('error', (err) => reject(err));
  });
}

async function main() {
  // Start a small status server
  try {
    const app = fastify({ logger: false });
    app.get('/health', async () => ({ ok: true, ts: new Date().toISOString() }));
    app.get('/status', async () => {
      try {
        const raw = fs.readFileSync(getStatusPath(), 'utf-8');
        return JSON.parse(raw);
      } catch (e) {
        return { ok: false, error: String(e) };
      }
    });
    app.get('/', async (_req, reply) => {
      const html = `<!doctype html>
<html lang="ko"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Training Status</title>
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#0b0f17;color:#e8eef9;margin:0;padding:24px}
 .card{background:#111827;border:1px solid #1f2937;border-radius:8px;padding:16px;margin-bottom:16px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
 .kv{display:flex;justify-content:space-between;margin:4px 0}
 .phase{font-weight:700}
 code{background:#0f172a;padding:2px 6px;border-radius:4px}
 small{color:#9ca3af}
</style></head><body>
<h1>Training Status</h1>
<div id="summary" class="card">Loading...</div>
<div class="grid">
 <div class="card"><h3>Self-Play</h3><div id="sp"></div></div>
 <div class="card"><h3>Distillation</h3><div id="dist"></div></div>
 <div class="card"><h3>Arena</h3><div id="arena"></div></div>
 <div class="card"><h3>Upload/Book</h3><div id="ub"></div></div>
 </div>
<script>
 async function render(){
  try{
   const r = await fetch('/status');
   const j = await r.json();
   var summaryHtml = ''
     + '<div class="kv"><span>Phase</span><span class="phase">' + (j.phase || '-') + '</span></div>'
     + '<div class="kv"><span>Cycle</span><span>' + (j.cycle || '-') + '</span></div>'
     + '<div class="kv"><span>Updated</span><span><small>' + (j.ts || '') + '</small></span></div>';
   document.getElementById('summary').innerHTML = summaryHtml;

   var sp = j.selfPlay || {};
   var spHtml = ''
     + '<div class="kv"><span>Workers</span><span>' + (sp.workers != null ? sp.workers : '-') + '</span></div>'
     + '<div class="kv"><span>Samples</span><span>' + (sp.samplesTotal != null ? sp.samplesTotal : '-') + '</span></div>'
     + '<div class="kv"><span>Last Save</span><span><small>' + (sp.lastSave || '') + '</small></span></div>';
   document.getElementById('sp').innerHTML = spHtml;

   var ds = j.distill || {};
   var dsHtml = ''
     + '<div class="kv"><span>Epoch</span><span>' + (ds.epoch != null ? ds.epoch : '-') + ' / ' + (ds.epochs != null ? ds.epochs : '-') + '</span></div>'
     + '<div class="kv"><span>Last Saved</span><span><small>' + (ds.lastSaved || '') + '</small></span></div>';
   document.getElementById('dist').innerHTML = dsHtml;

   var ar = j.arena || {};
   var arHtml = ''
     + '<div class="kv"><span>Played</span><span>' + (ar.played != null ? ar.played : 0) + '/' + (ar.total != null ? ar.total : 0) + '</span></div>'
     + '<div class="kv"><span>Wins (Cand/Prod)</span><span>' + (ar.candWins != null ? ar.candWins : 0) + ' / ' + (ar.prodWins != null ? ar.prodWins : 0) + '</span></div>'
     + '<div class="kv"><span>Draws</span><span>' + (ar.draws != null ? ar.draws : 0) + '</span></div>'
     + '<div class="kv"><span>Promoted</span><span>' + ((ar.promoted ? 'Yes' : 'No')) + '</span></div>';
   document.getElementById('arena').innerHTML = arHtml;

   var ub = j.upload || {}; var bk = j.book || {};
   var urlHtml = ub.publicUrl ? '<a href="' + ub.publicUrl + '" target="_blank">link</a>' : '-';
   var ubHtml = ''
     + '<div class="kv"><span>Upload OK</span><span>' + (ub.ok ? 'Yes' : '-') + '</span></div>'
     + '<div class="kv"><span>Model URL</span><span>' + urlHtml + '</span></div>'
     + '<div class="kv"><span>Book Entries</span><span>' + (bk.entries != null ? bk.entries : '-') + '</span></div>'
     + '<div class="kv"><span>Imported</span><span>' + (bk.imported ? 'Yes' : '-') + '</span></div>';
   document.getElementById('ub').innerHTML = ubHtml;
  } catch (e) {
    document.getElementById('summary').innerText = 'Status unavailable';
  }
 }
 render(); setInterval(render, 2000);
</script>
</body></html>`;
      reply.header('content-type','text/html').send(html);
    });
    const port = Number(process.env.STATUS_PORT || 8090);
    app.listen({ port, host: '0.0.0.0' }).catch(() => {});
    console.log(`[Status] Server listening on :${port}`);
  } catch {}
  // Ensure Supabase env from gomoku-app-v2/.env.local if missing
  try {
    if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE_KEY) {
      const candidate = path.resolve(__dirname, '..', 'gomoku-app-v2', '.env.local');
      if (fs.existsSync(candidate)) {
        const txt = fs.readFileSync(candidate, 'utf-8');
        for (const line of txt.split(/\r?\n/)) {
          const m = line.match(/^([A-Z0-9_]+)=(.*)$/);
          if (m) {
            const k = m[1];
            const v = m[2];
            if (!process.env[k]) process.env[k] = v;
          }
        }
      }
    }
  } catch {}
  try {
    // Default: run forever unless PIPELINE_CYCLES > 0
    const cycles = Number(process.env.PIPELINE_CYCLES ?? 0);
    const intervalMs = Number(process.env.PIPELINE_INTERVAL_MS || 0);
    const runDistill = (process.env.RUN_DISTILLATION || 'true').toLowerCase() === 'true';
    const runUpload = (process.env.UPLOAD_MODEL_AFTER || 'true').toLowerCase() === 'true';
    const autoBook = (process.env.AUTO_GENERATE_OPENING_BOOK || 'true').toLowerCase() === 'true';
    const importBook = (process.env.IMPORT_OPENING_BOOK || 'true').toLowerCase() === 'true';
    const gatingEnabled = (process.env.GATING_ENABLED || 'true').toLowerCase() === 'true';
    const forever = (process.env.FOREVER || 'false').toLowerCase() === 'true' || cycles === 0;
    const onErrorDelayMs = Number(process.env.ON_ERROR_DELAY_MS || 60_000);

    let cycle = 1;
    while (forever || cycle <= cycles) {
      console.log(`\n================ CYCLE ${cycle}/${cycles} ================`);
      await updateStatus({ phase: 'self_play_start', cycle });
      await notify('self_play', 'start');
      let hadError = false;
      try {
        await runScript(SELF_PLAY_SCRIPT, 'Self-Play Data Generation');
        await notify('self_play', 'success');
        console.log('\n--- Data generation complete. ---');
        await updateStatus({ phase: 'self_play_done' });
      } catch (e) {
        hadError = true;
        await notify('self_play', 'error', { error: String(e) });
        console.error('Self-play failed; continuing to next stages. Reason:', e);
        await updateStatus({ error: `self_play: ${String(e)}` });
      }

      if (runDistill) {
        await updateStatus({ phase: 'distill_start' });
        await notify('distill', 'start');
        try {
          await runScript(DISTILL_SCRIPT, 'Distillation Training');
          await notify('distill', 'success');
          console.log('\n--- Distillation complete. ---');
          await updateStatus({ phase: 'distill_done' });
        } catch (e) {
          hadError = true;
          await notify('distill', 'error', { error: String(e) });
          console.error('Distillation failed; continuing pipeline. Reason:', e);
          await updateStatus({ error: `distill: ${String(e)}` });
        }
      } else {
        console.log('\n--- Distillation skipped (set RUN_DISTILLATION=true to enable) ---');
      }

      // Arena gating (if GATING_ENABLED)
      let promoted = false;
      if (gatingEnabled) {
        await updateStatus({ phase: 'arena_start' });
        await notify('arena', 'start');
        try {
          await runScript(ARENA_SCRIPT, 'Arena Gating');
          await notify('arena', 'success');
          await updateStatus({ phase: 'arena_done' });
          // Read arena result
          try {
            const resPath = process.env.ARENA_RESULT_PATH || require('path').resolve(__dirname, '..', 'arena_result.json');
            const text = require('fs').readFileSync(resPath, 'utf-8');
            const obj = JSON.parse(text);
            promoted = !!obj?.promoted;
            // Lightweight auto-tuning of tactical weights based on arena outcome
            try {
              const TUNE_ON_ARENA = (process.env.TUNE_PARAMS_ON_ARENA || 'true').toLowerCase() === 'true';
              if (TUNE_ON_ARENA && typeof obj?.winrate === 'number') {
                const winrate: number = obj.winrate;
                const thr: number = typeof obj.threshold === 'number' ? obj.threshold : getTuned('WINRATE_THRESHOLD', 0.55);
                const lr = Number(process.env.TUNE_LR || 0.03);
                const delta = (winrate - thr);
                // Move small steps; if above threshold, slightly increase key boosts; else, decrease a bit
                const f = 1 + lr * delta;
                const mul = clampNum(f, 0.9, 1.1);
                const mixDelta = clampNum(0.1 * delta, -0.05, 0.05);
                const CHILD_TT_PRIOR_MIX = clampNum(getTuned('CHILD_TT_PRIOR_MIX', 0.35) + mixDelta, 0, 0.6);
                updateTuning({
                  // Root boosts
                  BOOST_CREATE: getTuned('BOOST_CREATE', 1.5) * mul,
                  BOOST_BLOCK: getTuned('BOOST_BLOCK', 1.3) * mul,
                  BOOST_OPEN3_ROOT: getTuned('BOOST_OPEN3_ROOT', 1.08) * mul,
                  BOOST_OPEN3_ROOT_BLOCK: getTuned('BOOST_OPEN3_ROOT_BLOCK', 1.05) * mul,
                  BOOST_FOUR_ROOT: getTuned('BOOST_FOUR_ROOT', 1.15) * mul,
                  BOOST_FOUR_ROOT_BLOCK: getTuned('BOOST_FOUR_ROOT_BLOCK', 1.1) * mul,
                  BOOST_CONN3_ROOT: getTuned('BOOST_CONN3_ROOT', 1.05) * mul,
                  BOOST_CONN3_ROOT_BLOCK: getTuned('BOOST_CONN3_ROOT_BLOCK', 1.03) * mul,
                  BOOST_LINK_ROOT: getTuned('BOOST_LINK_ROOT', 1.03) * mul,
                  // Child boosts
                  BOOST_CREATE_CHILD: getTuned('BOOST_CREATE_CHILD', 1.3) * mul,
                  BOOST_BLOCK_CHILD: getTuned('BOOST_BLOCK_CHILD', 1.2) * mul,
                  BOOST_OPEN3_CHILD: getTuned('BOOST_OPEN3_CHILD', 1.1) * mul,
                  BOOST_OPEN3_BLOCK_CHILD: getTuned('BOOST_OPEN3_BLOCK_CHILD', 1.05) * mul,
                  BOOST_CONN3_CHILD: getTuned('BOOST_CONN3_CHILD', 1.05) * mul,
                  BOOST_CONN3_BLOCK_CHILD: getTuned('BOOST_CONN3_BLOCK_CHILD', 1.02) * mul,
                  BOOST_LINK_CHILD: getTuned('BOOST_LINK_CHILD', 1.02) * mul,
                  CHILD_TT_PRIOR_MIX,
                });
              }
            } catch {}
          } catch {}
        } catch (e) {
          hadError = true;
          await notify('arena', 'error', { error: String(e) });
          console.error('Arena failed; continuing pipeline. Reason:', e);
          await updateStatus({ error: `arena: ${String(e)}` });
        }
      }

      if (runUpload && (!gatingEnabled || promoted)) {
        await updateStatus({ phase: 'upload_start' });
        await notify('upload', 'start');
        try {
          await runScript(UPLOAD_SCRIPT, 'Upload Model to Supabase');
          await notify('upload', 'success');
          console.log('\n--- Upload complete. ---');
          await updateStatus({ phase: 'upload_done' });
        } catch (e) {
          hadError = true;
          await notify('upload', 'error', { error: String(e) });
          console.error('Upload failed; continuing. Reason:', e);
          await updateStatus({ error: `upload: ${String(e)}` });
        }
      } else {
        console.log('\n--- Upload skipped (either disabled or not promoted) ---');
      }

      if (autoBook) {
        await updateStatus({ phase: 'book_build_start' });
        await notify('build_book', 'start');
        try {
          await runScript(BUILD_BOOK_SCRIPT, 'Build Opening Book');
          await notify('build_book', 'success');
          console.log('\n--- Build opening book complete. ---');
        } catch (e) {
          hadError = true;
          await notify('build_book', 'error', { error: String(e) });
          console.warn('Build opening book failed; continuing. Reason:', e);
          await updateStatus({ error: `build_book: ${String(e)}` });
        }
        if (importBook) {
          await updateStatus({ phase: 'book_import_start' });
          await notify('import_book', 'start');
          const hasSupabase = !!(process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_ROLE_KEY);
          if (!hasSupabase) {
            console.log('\n--- Import opening book skipped: missing Supabase credentials ---');
            try { await updateStatus({ book: { imported: false } }); } catch {}
          } else {
            try {
              await runScript(IMPORT_BOOK_SCRIPT, 'Import Opening Book into Supabase');
              await notify('import_book', 'success');
              console.log('\n--- Import opening book complete. ---');
              await updateStatus({ phase: 'book_import_done' });
            } catch (e) {
              // Treat opening-book import as optional: log and mark status, but do not flag pipeline error
              await notify('import_book', 'error', { error: String(e) });
              console.warn('Import opening book failed (optional); continuing. Reason:', e);
              try { await updateStatus({ book: { imported: false } }); } catch {}
            }
          }
        }
      }

      cycle++;
      const delay = (hadError ? onErrorDelayMs : intervalMs);
      if ((forever || cycle <= cycles) && delay > 0) {
        console.log(`\n--- Waiting ${delay}ms before next cycle ---`);
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  } catch (error) {
    console.error('\n--- PIPELINE FAILED ---\n', error);
    await notify('pipeline', 'error', { error: String(error) });
    await updateStatus({ phase: 'error', error: String(error) });
    // Keep process alive (status server) for visibility; operator can fix and restart manually.
  }
}

main();
