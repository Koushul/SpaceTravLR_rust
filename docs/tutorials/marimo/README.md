# Slide-seq + BANKSY + interactive UMAP (marimo)

Interactive tutorial:

- Load Squidpy **Slide-seq V2 mouse hippocampus** (`sq.datasets.slideseqv2`)
- Run **BANKSY** (`pybanksy`) spatial clustering for several `λ`
- Spatial + UMAP plots with interactive `λ`, color (BANKSY vs published), and UMAP params

## Prepare demo assets (once)

```bash
pip install scanpy squidpy umap-learn anndata numpy pybanksy matplotlib
python docs/tutorials/marimo/prepare_slideseq_banksy.py
```

Writes `docs/tutorials/marimo/public/slideseq_banksy/` (~7MB).

## Run locally

```bash
pip install marimo numpy matplotlib
marimo run docs/tutorials/marimo/squidpy_spatial_umap.py
```

## Export WASM

```bash
export UV_NO_CACHE=1
export UV_CACHE_DIR=/tmp/uv-cache
export XDG_CACHE_HOME=/tmp/xdg-cache

marimo export html-wasm docs/tutorials/marimo/squidpy_spatial_umap.py \
  -o docs/tutorials/marimo/site \
  --mode run \
  --show-code \
  --no-sandbox
```

Live demo: https://graceful-hollow-a6tk.here.now/
