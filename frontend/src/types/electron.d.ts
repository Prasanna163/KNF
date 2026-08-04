export interface ElectronAPI {
  getBackendUrl: () => Promise<string>;
  getBackendStatus: () => Promise<string>;
  selectOutputDirectory: () => Promise<string | null>;
  onBackendStatus: (callback: (status: string) => void) => void;
  onBackendError: (callback: (msg: string) => void) => void;
}

declare global {
  interface Window {
    electronAPI?: ElectronAPI;
  }
}
