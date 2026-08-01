import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

export default defineConfig({
  plugins: [wasm(), topLevelAwait()],
  server: {
    port: 5177,
    proxy: {
      "/api": "http://127.0.0.1:8787",
    },
  },
  build: {
    target: "esnext",
  },
});
