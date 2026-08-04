import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

const src = path.join(root, 'electron', 'preload.cjs');
const destDir = path.join(root, 'dist-electron');
const dest = path.join(destDir, 'preload.cjs');

if (!fs.existsSync(destDir)) {
  fs.mkdirSync(destDir, { recursive: true });
}

fs.copyFileSync(src, dest);
console.log('Copied preload.cjs to dist-electron');
