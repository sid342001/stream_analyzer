import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    port: 5173,
    // Listen on all interfaces, not just localhost — lets another device on
    // the LAN (or a VM/container) load the console via this host's real IP.
    // Proxy targets below stay "localhost": that's the Vite dev *process*
    // talking to the backend on the same machine, unrelated to which
    // interface a remote browser used to reach Vite itself.
    host: true,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
});
