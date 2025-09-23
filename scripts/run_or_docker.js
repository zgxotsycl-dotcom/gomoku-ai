#!/usr/bin/env node
const { spawnSync, spawn } = require('child_process');
const path = require('path');
const dotenv = require('dotenv');

function run(cmd, args, opts = {}) {
  const res = spawnSync(cmd, args, { stdio: 'inherit', shell: false, ...opts });
  return res.status === 0;
}

// Load local env files if present (pipeline expects Supabase creds, etc.)
const envFiles = ['.env', '.env.supabase'];
for (const file of envFiles) {
  const resolved = path.resolve(__dirname, '..', file);
  dotenv.config({ path: resolved, override: false });
}

function main() {
  // 1) Check tfjs-node binding
  let hasTfjsNode = false;
  try {
    require('@tensorflow/tfjs-node');
    hasTfjsNode = true;
  } catch (e) {
    console.warn('[Precheck] @tensorflow/tfjs-node failed to load:', String(e && e.message || e));
  }

  if (hasTfjsNode) {
    // 2) Run pipeline locally
    const child = spawn(process.execPath, ['dist/start_pipeline.js'], { stdio: 'inherit' });
    child.on('close', (code) => process.exit(code || 0));
    child.on('error', (err) => { console.error('Failed to start pipeline:', err); process.exit(1); });
    return;
  }

  // 3) Fallback to Docker Compose if available
  const hasDocker = run('docker', ['--version']);
  if (!hasDocker) {
    console.error('\n[Error] tfjs-node native binding unavailable on this Node version.');
    console.error('Install Docker Desktop and run: npm run docker');
    console.error('Or use Node 20: nvm install 20 && nvm use 20, then re-run.');
    process.exit(1);
  }

  console.log('\n[Fallback] Using Docker Compose to run the pipeline in a Node 20 environment...');
  const ok = run('docker', ['compose', 'up', '-d', '--build']);
  if (!ok) {
    console.error('\n[Error] Docker Compose failed. Please run "npm run docker" and check Docker Desktop.');
    process.exit(1);
  }
  console.log('\n[OK] Trainer container started.');
  console.log('- Dashboard: http://localhost:8090');
  console.log('- Logs: docker compose logs -f trainer');
}

main();


