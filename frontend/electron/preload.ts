import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
  getBackendUrl: () => ipcRenderer.invoke('get-backend-url'),
  getBackendStatus: () => ipcRenderer.invoke('get-backend-status'),
  selectOutputDirectory: () => ipcRenderer.invoke('select-output-directory'),
  onBackendStatus: (callback: (status: string) => void) => {
    ipcRenderer.on('backend-status', (_event, status) => callback(status));
  },
  onBackendError: (callback: (msg: string) => void) => {
    ipcRenderer.on('backend-error', (_event, msg) => callback(msg));
  },
});
