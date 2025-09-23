// Unified TensorFlow loader for Node. Prefer tfjs-node; fall back to pure tfjs
// and register minimal file:// IO handlers so models can be loaded/saved from disk.
import * as path from 'path';
import * as fs from 'fs/promises';
import * as fss from 'fs';

// Use types from tfjs for editor experience; runtime may be tfjs-node or tfjs.
import type * as TFNS from '@tensorflow/tfjs';
type TFModule = typeof import('@tensorflow/tfjs');

let tf: any;

function ensureTensorflowDll(pkgName: string) {
  if (process.platform !== 'win32') return;
  try {
    const pkgPath = require.resolve(`${pkgName}/package.json`);
    const baseDir = path.dirname(pkgPath);
    const libDir = path.join(baseDir, 'lib');
    const targetDir = path.join(libDir, 'napi-v8');
    if (!fss.existsSync(targetDir)) return;

    const currentPath = process.env.PATH || '';
    if (!currentPath.split(';').some((seg) => seg.toLowerCase() === targetDir.toLowerCase())) {
      process.env.PATH = `${targetDir};${currentPath}`;
    }

    const targetDll = path.join(targetDir, 'tensorflow.dll');
    if (!fss.existsSync(targetDll)) {
      const entries = fss.readdirSync(libDir, { withFileTypes: true });
      for (const entry of entries) {
        if (!entry.isDirectory()) continue;
        const candidate = path.join(libDir, entry.name, 'tensorflow.dll');
        if (fss.existsSync(candidate)) {
          try {
            fss.copyFileSync(candidate, targetDll);
            if (!process.env.SUPPRESS_TF_WARN) {
              console.log(`[TF] Patched tensorflow.dll into ${targetDir}`);
            }
            break;
          } catch {}
        }
      }
    }

    const zlibTarget = path.join(targetDir, 'zlibwapi.dll');
    if (!fss.existsSync(zlibTarget)) {
      const candidates: string[] = [
        path.join(libDir, 'napi-v10', 'zlibwapi.dll'),
        path.join(libDir, 'napi-v9', 'zlibwapi.dll'),
        path.join(libDir, 'napi-v8', 'zlibwapi.dll'),
        path.join(baseDir, 'deps', 'lib', 'zlibwapi.dll'),
        process.env.ZLIBWAPI_PATH ?? '',
        path.join(process.cwd(), 'zlibwapi.dll'),
      ];
      try {
        const cpuPkg = require.resolve('@tensorflow/tfjs-node/package.json');
        const cpuLibDir = path.join(path.dirname(cpuPkg), 'lib');
        for (const dirName of ['napi-v10', 'napi-v9', 'napi-v8']) {
          candidates.push(path.join(cpuLibDir, dirName, 'zlibwapi.dll'));
        }
      } catch {}
      for (const candidate of candidates) {
        if (!candidate || candidate === zlibTarget) continue;
        if (fss.existsSync(candidate)) {
          try {
            fss.copyFileSync(candidate, zlibTarget);
            if (!process.env.SUPPRESS_TF_WARN) {
              console.log(`[TF] Patched zlibwapi.dll into ${targetDir}`);
            }
            break;
          } catch {}
        }
      }
    }
  } catch {}
}


function loadTfjsNode(moduleName: string) {
  ensureTensorflowDll(moduleName);
  try {
    return require(moduleName);
  } catch (err) {
    const code = (err as NodeJS.ErrnoException)?.code;
    if (process.platform === 'win32' && code === 'ERR_DLOPEN_FAILED') {
      ensureTensorflowDll(moduleName);
      try {
        return require(moduleName);
      } catch (err2) {
        throw err2;
      }
    }
    throw err;
  }
}

function registerNodeFileIO(tflib: any) {
  // Register Node file:// IO handlers for pure @tensorflow/tfjs

  // Register load handler for file://model.json
  tflib.io.registerLoadRouter((url: string | TFNS.io.IOHandler) => {
    if (typeof url !== 'string' || !url.startsWith('file://')) return null;
    const href = url;
    return {
      load: async () => {
        const modelJsonPath = url.replace('file://', '');
        const dir = path.dirname(modelJsonPath);
        const raw = await fs.readFile(modelJsonPath, 'utf-8');
        const j = JSON.parse(raw);
        const weightSpecs = ([] as any[]).concat(...(j.weightsManifest || []).map((g: any) => g.weights || []));
        const binPaths = ([] as string[]).concat(...(j.weightsManifest || []).map((g: any) => g.paths || []));
        const buffers: Buffer[] = [];
        for (const p of binPaths) {
          const abs = path.join(dir, p);
          const buf = await fs.readFile(abs);
          buffers.push(buf);
        }
        const bufCombined = Buffer.concat(buffers);
        const weightData = bufCombined.buffer.slice(bufCombined.byteOffset, bufCombined.byteOffset + bufCombined.byteLength);
        return {
          modelTopology: j.modelTopology,
          format: j.format,
          generatedBy: j.generatedBy,
          convertedBy: j.convertedBy,
          trainingConfig: j.trainingConfig,
          weightSpecs,
          weightData,
        } as unknown as TFNS.io.ModelArtifacts;
      }
    } as TFNS.io.IOHandler;
  });

  // Register save handler for file://<dir> (writes model.json + weights.bin)
  tflib.io.registerSaveRouter((url: string | TFNS.io.IOHandler) => {
    if (typeof url !== 'string' || !url.startsWith('file://')) return null;
    const href = url;
    return {
      save: async (artifacts: TFNS.io.ModelArtifacts) => {
        const dir = url.replace('file://', '');
        await fs.mkdir(dir, { recursive: true });
        const modelJsonPath = path.join(dir, 'model.json');
        const weightsBinPath = path.join(dir, 'weights.bin');
        const weightSpecs = artifacts.weightSpecs || [];
        const weightData = artifacts.weightData || new ArrayBuffer(0);
        const modelJson = {
          modelTopology: artifacts.modelTopology,
          format: 'layers-model',
          generatedBy: 'tfjs-fallback',
          convertedBy: null,
          trainingConfig: artifacts.trainingConfig || null,
          weightsManifest: [
            { paths: ['weights.bin'], weights: weightSpecs }
          ]
        } as any;

        await fs.writeFile(modelJsonPath, JSON.stringify(modelJson));
        await fs.writeFile(weightsBinPath, Buffer.from(weightData as ArrayBuffer));

        const info: TFNS.io.ModelArtifactsInfo = {
          dateSaved: new Date(),
          modelTopologyType: 'JSON',
          modelTopologyBytes: Buffer.byteLength(JSON.stringify(artifacts.modelTopology || {})),
          weightSpecsBytes: Buffer.byteLength(JSON.stringify(weightSpecs)),
          weightDataBytes: (weightData as ArrayBuffer).byteLength,
        };
        return { modelArtifactsInfo: info };
      }
    } as TFNS.io.IOHandler;
  });
}

// Prefer GPU if available and not disabled
const WANT_GPU = (process.env.TF_USE_GPU || '1') !== '0';
try {
  if (WANT_GPU) {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    tf = loadTfjsNode('@tensorflow/tfjs-node-gpu');
    if (!process.env.SUPPRESS_TF_WARN) console.log('[TF] Using @tensorflow/tfjs-node-gpu backend.');
  } else {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    tf = loadTfjsNode('@tensorflow/tfjs-node');
    if (!process.env.SUPPRESS_TF_WARN) console.log('[TF] Using @tensorflow/tfjs-node (CPU) backend (GPU disabled by env).');
  }
} catch (e) {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    tf = require('@tensorflow/tfjs-node');
    if (!process.env.SUPPRESS_TF_WARN) console.log('[TF] Using @tensorflow/tfjs-node (CPU) backend (GPU binding not available).');
  } catch (e2) {
    // Fallback to pure JS runtime
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    tf = require('@tensorflow/tfjs');
    try { registerNodeFileIO(tf); } catch {}
    if (!process.env.SUPPRESS_TF_WARN) {
      console.warn('[TF] Using @tensorflow/tfjs (pure JS) fallback. Install tfjs-node or tfjs-node-gpu for better performance.');
    }
  }
}

// Optional: allow growth to avoid large preallocation
if (typeof process !== 'undefined' && process.env.TF_FORCE_GPU_ALLOW_GROWTH === '1') {
  // tfjs-node honors TF environment variables; explicit API not needed.
  if (!process.env.SUPPRESS_TF_WARN) console.log('[TF] GPU allow growth enabled (TF_FORCE_GPU_ALLOW_GROWTH=1).');
}

// Optional: enable mixed precision on supported backends (experimental)
try {
  const WANT_MP = (process.env.TF_MIXED_PRECISION || '0') === '1';
  if (WANT_MP && tf?.mixedPrecision?.setGlobalPolicy) {
    tf.mixedPrecision.setGlobalPolicy('mixed_float16');
    if (!process.env.SUPPRESS_TF_WARN) console.log('[TF] Mixed precision policy set to mixed_float16.');
  }
} catch {}

export default (tf as unknown as TFModule);
