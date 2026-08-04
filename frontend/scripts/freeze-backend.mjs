/**
 * Build the self-contained Python backend used by packaged Electron builds.
 *
 * The generated runtime is deliberately not committed. It contains:
 *   - CPython 3.11 embeddable runtime
 *   - NCIForge and its API dependencies
 *   - CPU-only PyTorch (the reliable fallback on every machine)
 *   - pip, used by the installer for the optional CUDA PyTorch layer
 */

import { spawnSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const backendDir = path.join(repoRoot, 'NCIForge');
const buildScript = path.join(__dirname, 'build_backend_runtime.py');
const outputDir = path.join(repoRoot, 'frontend', 'resources', 'backend');
const venvPython = path.join(
  backendDir,
  '.venv-nciforge',
  process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python',
);

if (process.platform !== 'win32') {
  console.error('[bundle-backend] Windows packaging must be built on Windows.');
  process.exit(1);
}

if (!fs.existsSync(venvPython)) {
  console.error(`[bundle-backend] Build environment not found: ${venvPython}`);
  console.error('[bundle-backend] Create NCIForge/.venv-nciforge and install the project first.');
  process.exit(1);
}

if (!fs.existsSync(buildScript)) {
  console.error(`[bundle-backend] Build script not found: ${buildScript}`);
  process.exit(1);
}

console.log('[bundle-backend] Building private CPU backend runtime...');
const result = spawnSync(
  venvPython,
  [
    buildScript,
    '--repo-root',
    repoRoot,
    '--output',
    outputDir,
  ],
  {
    cwd: repoRoot,
    stdio: 'inherit',
    env: {
      ...process.env,
      PYTHONUTF8: '1',
      PIP_DISABLE_PIP_VERSION_CHECK: '1',
    },
  },
);

if (result.status !== 0) {
  console.error(`[bundle-backend] Runtime build failed with exit code ${result.status}`);
  process.exit(result.status ?? 1);
}

const pythonExe = path.join(outputDir, 'runtime', 'python.exe');
const manifest = path.join(outputDir, 'runtime-manifest.json');
const cudaInstaller = path.join(outputDir, 'install_cuda_torch.py');

for (const required of [pythonExe, manifest, cudaInstaller]) {
  if (!fs.existsSync(required)) {
    console.error(`[bundle-backend] Required output is missing: ${required}`);
    process.exit(1);
  }
}

console.log('[bundle-backend] Private CPU runtime is ready.');
