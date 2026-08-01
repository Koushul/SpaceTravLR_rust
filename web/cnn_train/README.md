# SpaceTravLR WASM CNN trainer

Browser UI + Axum server that:

1. Prepares Lasso anchors and spatial CNN inputs natively from an `.h5ad`
2. Ships per-gene bincode packs to the browser
3. Runs `CellularNicheNetwork` Adam epochs in **WebAssembly** (Burn NdArray)

## Build

```bash
# WASM module
wasm-pack build crates/spacetravlr-cnn-wasm --target web --out-dir ../../web/cnn_train/pkg --release

# UI
cd web/cnn_train && npm ci && npm run build && cd ../..

# Server
cargo build --release --features cnn-web --bin spacetravlr-cnn-web
```

## Run (tonsil demo)

```bash
# Process once if needed
spacetravlr --plain --rust-process-h5ad \
  --h5ad data/h5ad/SlideTags_human_tonsil.h5ad \
  -o /tmp/tonsil_processed.h5ad

SPACETRAVLR_DATA_DIR=$PWD/data \
./target/release/spacetravlr-cnn-web \
  --h5ad /tmp/tonsil_processed.h5ad \
  --config spaceship_config.toml \
  --work-dir /tmp/spacetravlr_cnn_web \
  --static-dir web/cnn_train/dist \
  --default-genes AICDA,CD74,BCL6,PAX5 \
  --spatial-dim 8 \
  --max-ligands 64 \
  --wasm-epochs 4 \
  --bind 0.0.0.0 \
  --port 8787
```

Open `http://127.0.0.1:8787`, click **Train in WASM**.

## API

| Endpoint | Role |
| --- | --- |
| `GET /api/info` | Dataset / defaults |
| `POST /api/prepare` | `{ genes, epochs, native_cnn? }` — Lasso dump (+ optional native CNN) |
| `GET /api/pack?gene=AICDA` | Bincode `CnnGeneTrainPack` for WASM |
| `GET /api/status` | Last job status |
