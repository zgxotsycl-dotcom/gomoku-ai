import fastify from 'fastify';
import * as fs from 'fs';
import { getStatusPath } from './status';

async function main() {
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
     + '<div class="kv"><span>Promoted</span><span>' + (ar.promoted ? 'Yes' : 'No') + '</span></div>';
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
  const host = process.env.STATUS_HOST || '0.0.0.0';
  await app.listen({ port, host }).catch(() => {});
  console.log(`[Status] Server listening on ${host}:${port}`);
}

main();
