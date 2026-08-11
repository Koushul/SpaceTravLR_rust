# Tonsil GC niche method benchmark (marimo)

Compare **SpaceTravLR β** niches against **BANKSY**, **COVET**, and **NicheCompass**
on human tonsil snRNA germinal-center B cells.

GC Light / Dark / Intermediate Zone labels (`cell_type_2`) are expression-derived
and treated as a **confounded** reference only. Primary read-outs are the
NicheCompass single-sample metrics (CAS, MLAMI, CLISIS, GCS, CNMI, NASW) and
pairwise ARI between methods — same suite as the hippocampus demo at
https://graceful-hollow-a6tk.here.now/.

Live report: https://plucky-meadow-perf.here.now/

## Prepare assets

```bash
source /tmp/nichebench-venv/bin/activate
export HOME=/tmp/fakehome
# optional: write elsewhere then sync into public/
# export TONSIL_NICHE_OUT=/tmp/tonsil_niche_benchmark
python docs/tutorials/marimo/prepare_tonsil_niche_benchmark.py
```

Writes `docs/tutorials/marimo/public/tonsil_niche_benchmark/`.

Requires SpaceTravLR β feathers under the tonsil run directory, plus
`beta_features_kept.csv` (copied into `public/tonsil_niche_benchmark/` for
reproducibility). Expression methods use `layers['raw_count']` because `.X` is
already scaled.

## Run locally

```bash
pip install marimo numpy matplotlib
marimo run docs/tutorials/marimo/tonsil_niche_benchmark.py
```

## Export WASM

```bash
export UV_NO_CACHE=1
export UV_CACHE_DIR=/tmp/uv-cache
export XDG_CACHE_HOME=/tmp/xdg-cache

marimo export html-wasm docs/tutorials/marimo/tonsil_niche_benchmark.py \
  -o docs/tutorials/marimo/site_tonsil \
  --mode run \
  --show-code \
  --no-sandbox
```
