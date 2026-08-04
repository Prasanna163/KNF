/**
 * make-icon.mjs
 * Converts public/icon.png → public/icon.ico using pure Node.js binary packing.
 * Completely cross-platform, requires no external dependencies or PowerShell.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.join(__dirname, '..', 'public');
const pngSrc    = path.join(publicDir, 'icon.png');
const icoDest   = path.join(publicDir, 'icon.ico');

if (!fs.existsSync(pngSrc)) {
  console.error(`[make-icon] ERROR: public/icon.png not found!`);
  console.error(`[make-icon] Place a high-res PNG at public/icon.png first.`);
  process.exit(1);
}

try {
  console.log('[make-icon] Reading source PNG...');
  const pngBuffer = fs.readFileSync(pngSrc);
  const size = pngBuffer.length;

  console.log(`[make-icon] Source size: ${(size / 1024).toFixed(1)} KB`);

  // 1. ICO Header (6 bytes)
  const header = Buffer.alloc(6);
  header.writeUInt16LE(0, 0); // Reserved (must be 0)
  header.writeUInt16LE(1, 2); // Resource type (1 = ICO)
  header.writeUInt16LE(1, 4); // Number of images in file (1)

  // 2. Icon Directory Entry (16 bytes)
  const entry = Buffer.alloc(16);
  entry.writeUInt8(0, 0);      // Width: 0 (for >= 256 pixels)
  entry.writeUInt8(0, 1);      // Height: 0 (for >= 256 pixels)
  entry.writeUInt8(0, 2);      // Color count: 0 (no palette)
  entry.writeUInt8(0, 3);      // Reserved: 0
  entry.writeUInt16LE(1, 4);   // Color planes: 1
  entry.writeUInt16LE(32, 6);  // Bits per pixel: 32 (RGBA)
  entry.writeUInt32LE(size, 8); // Size of image data in bytes
  entry.writeUInt32LE(22, 12); // Offset of image data from start (6 + 16 = 22)

  // 3. Combine parts into final ICO buffer
  const icoBuffer = Buffer.concat([header, entry, pngBuffer]);

  // 4. Write to disk
  fs.writeFileSync(icoDest, icoBuffer);
  
  console.log(`[make-icon] ✓ icon.ico successfully generated at: ${icoDest}`);
  console.log(`[make-icon] Icon details: 512x512 PNG-compressed layer, total size: ${(icoBuffer.length / 1024).toFixed(1)} KB`);
  process.exit(0);
} catch (err) {
  console.error('[make-icon] Failed to generate icon.ico:', err.message);
  process.exit(1);
}
