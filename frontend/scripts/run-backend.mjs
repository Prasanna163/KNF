import { spawn, spawnSync } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const backendDir = path.join(repoRoot, 'NCIForge');

// Prefer the venv Python (has all backend packages installed)
const venvPython = process.platform === 'win32'
  ? path.join(backendDir, '.venv-nciforge', 'Scripts', 'python.exe')
  : path.join(backendDir, '.venv-nciforge', 'bin', 'python');

const candidates = [
  ...(fs.existsSync(venvPython) ? [{ command: venvPython, args: [] }] : []),
  { command: 'py', args: ['-3'] },
  { command: 'python', args: [] },
  { command: 'python3', args: [] },
];

function isAvailable(command, args) {
  const result = spawnSync(command, [...args, '--version'], { stdio: 'ignore' });
  return result.status === 0;
}

const python = candidates.find((candidate) => isAvailable(candidate.command, candidate.args));

if (!python) {
  console.error('Python was not found. Install Python 3 or add it to PATH.');
  process.exit(1);
}

const child = spawn(
  python.command,
  [...python.args, '-m', 'uvicorn', 'knf_core.api:app', '--host', '127.0.0.1', '--port', '8000'],
  { stdio: 'inherit', cwd: backendDir, shell: false },
);

child.on('exit', (code) => process.exit(code ?? 1));
