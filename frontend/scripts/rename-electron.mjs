import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dir = path.resolve(__dirname, '..', 'dist-electron');

for (const file of fs.readdirSync(dir)) {
  if (file.endsWith('.js')) {
    const oldPath = path.join(dir, file);
    const newPath = path.join(dir, file.replace(/\.js$/, '.cjs'));
    if (fs.existsSync(newPath)) fs.unlinkSync(newPath);
    fs.renameSync(oldPath, newPath);
  }
}

console.log('Electron files renamed to .cjs');
