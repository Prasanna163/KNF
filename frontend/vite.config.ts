import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  // CRITICAL for Electron: assets must use relative paths (./assets/...)
  // not absolute paths (/assets/...) because file:// protocol has no host.
  base: "./",
  server: {
    host: "127.0.0.1",
    port: 5179,
    hmr: {
      overlay: false,
    },
  },
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
