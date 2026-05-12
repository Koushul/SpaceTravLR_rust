# UMAP lab

Interactive UMAP over a Rust backend (`umap_lab` binary). Uses the same **umap-rs + HNSW** pipeline as `spacetravlr::rust_preprocess` (`run_umap_on_pca`). Loads **`obsm['X_pca']`** when present; otherwise runs normalize → HVG → PCA once, then only UMAP is re-run when you change parameters.

## Run (production-style)

From the SpaceTravLR repo root:

```bash
cd web/umap_lab && npm ci && npm run build
cargo build --features umap-lab --bin umap_lab
./target/debug/umap_lab --port 8765 --static-dir web/umap_lab/dist
```

Open `http://127.0.0.1:8765/` and enter the path to an `.h5ad` file.

## Run (dev: Vite HMR + API proxy)

Terminal A: `umap_lab` as above (serves API on 8765). Terminal B:

```bash
cd web/umap_lab && npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8765`.

## Scripts

- `npm run build` — typecheck + production bundle to `dist/`
- `npm run dev` — Vite dev server
- `npm run test` — Vitest
