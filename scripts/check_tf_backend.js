#!/usr/bin/env node
// Simple local backend check. This runs outside Docker and only verifies
// which TFJS backend is active in the compiled build.

async function main() {
  try {
    const m = await import('../dist/tf.js');
    let tf = m;
    while (tf && typeof tf === 'object' && 'default' in tf) tf = tf.default;
    if (typeof tf?.ready === 'function') {
      try {
        await tf.ready();
      } catch {}
    }
    const backend = typeof tf?.getBackend === 'function' ? tf.getBackend() : '(unknown)';
    const versions = tf?.version ? JSON.stringify(tf.version) : '{}';
    const engineName = tf?.engine?.()?.backendName || '(unknown)';
    const ctor = tf?.engine?.()?.backend?.constructor?.name || '(unknown)';
    console.log('[TF Check] Node:', process.version);
    console.log('[TF Check] Backend(getBackend):', backend);
    console.log('[TF Check] Backend(engine):', engineName, '/', ctor);
    console.log('[TF Check] Versions:', versions);
    console.log('[TF Check] TF_USE_GPU:', process.env.TF_USE_GPU || '(unset)');
    if (backend !== 'tensorflow') {
      console.warn('[TF Check] Non-native backend detected. Heavy training will be skipped by distill script.');
    } else {
      console.log('[TF Check] Native TensorFlow backend detected. Training is enabled.');
    }
  } catch (e) {
    console.error('[TF Check] Failed to load compiled TF module:', e?.message || e);
    process.exitCode = 1;
  }
}

main();

