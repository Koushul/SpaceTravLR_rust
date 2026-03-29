import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

const rootDir = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: rootDir,
  resolve: {
    alias: {
      child_process: path.join(rootDir, "vite-stubs/child_process.ts"),
    },
  },
  plugins: [viteSingleFile()],
  build: {
    outDir: "dist",
    emptyDir: false,
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      input: path.join(rootDir, "mcp-app.html"),
    },
  },
});
