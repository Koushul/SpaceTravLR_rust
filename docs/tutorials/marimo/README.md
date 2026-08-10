# Squidpy spatial + interactive UMAP (marimo)

Interactive intro tutorial:

- Load Squidpy Visium H&E (`sq.datasets.visium_hne_adata`)
- Spatial scatter colored by cluster
- Recompute UMAP with interactive `n_neighbors` / `min_dist`

## Run locally

```bash
pip install marimo scanpy squidpy umap-learn matplotlib
marimo run docs/tutorials/marimo/squidpy_spatial_umap.py
```

## Export WASM (for static hosting)

Use `/tmp` for uv caches (or disable them) so home-quota fills do not break export:

```bash
export UV_NO_CACHE=1
export UV_CACHE_DIR=/tmp/uv-cache
export XDG_CACHE_HOME=/tmp/xdg-cache

marimo export html-wasm docs/tutorials/marimo/squidpy_spatial_umap.py \
  -o docs/tutorials/marimo/site \
  --mode run \
  --show-code \
  --execute \
  --no-sandbox
```

Then serve `docs/tutorials/marimo/site/` over HTTP (required for WASM).

Live demo: https://graceful-hollow-a6tk.here.now/
