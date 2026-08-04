import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const src = path.join(__dirname, 'install-dependencies-and-app.bat');
const destDir = path.join(__dirname, '..', 'dist-installer');
const dest = path.join(destDir, 'install-dependencies-and-app.bat');

if (!fs.existsSync(destDir)) {
  fs.mkdirSync(destDir, { recursive: true });
}
fs.copyFileSync(src, dest);
console.log('[copy-installer-helper] Copied install-dependencies-and-app.bat to dist-installer');
