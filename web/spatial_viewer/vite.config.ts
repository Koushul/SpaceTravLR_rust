import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

const rootDir = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      child_process: path.join(rootDir, "vite-stubs/child_process.ts"),
    },
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8080",
    },
  },
  build: {
    outDir: "dist",
    emptyDir: true,
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (
            id.includes("node_modules/@deck.gl/") ||
            id.includes("node_modules/@luma.gl/") ||
            id.includes("node_modules/@math.gl/")
          ) {
            return "deck-stack";
          }
        },
      },
    },
  },
});
