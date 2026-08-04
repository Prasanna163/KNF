import { createRequire } from 'module';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { startBackend, stopBackend, getBackendUrl, getBackendLogPath } from './backend.js';

const _require = createRequire(import.meta.url);
const { app, BrowserWindow, ipcMain, dialog } = _require('electron') as typeof import('electron');

// Disable GPU acceleration and sandboxing to avoid crashes in virtualized/RDP/sandboxed environments
app.commandLine.appendSwitch('disable-gpu');
app.commandLine.appendSwitch('disable-gpu-sandbox');
app.commandLine.appendSwitch('no-sandbox');

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const isDev = !app.isPackaged;

let mainWindow: InstanceType<typeof BrowserWindow> | null = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    title: 'KNF Studio',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    show: false,
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow?.show();
  });

  if (isDev) {
    mainWindow.loadURL('http://127.0.0.1:5179');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(async () => {
  ipcMain.handle('get-backend-url', () => getBackendUrl());
  ipcMain.handle('get-backend-status', () => 'running');
  ipcMain.handle('select-output-directory', async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openDirectory', 'createDirectory'],
      title: 'Select Output Directory',
    });
    return result.canceled ? null : result.filePaths[0] ?? null;
  });

  try {
    await startBackend();
  } catch (err: any) {
    console.error('Failed to start backend:', err.message);
    const logPath = getBackendLogPath();
    dialog.showErrorBox(
      'Backend Error',
      `Failed to start the computation server.\n\n` +
      `${err.message}\n\n` +
      `A detailed log has been written to:\n${logPath}\n\n` +
      `Please send this file when reporting the issue.`
    );
  }

  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  stopBackend();
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  stopBackend();
});
