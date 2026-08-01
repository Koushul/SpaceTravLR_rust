# SpaceTravLR WASM CNN trainer

Browser UI + Axum server that:

1. Prepares Lasso anchors and spatial CNN inputs natively from an `.h5ad`
2. Ships per-gene bincode packs to the browser
3. Runs `CellularNicheNetwork` Adam epochs in **WebAssembly** via Burn:
   - **WebGPU** when `navigator.gpu` is available (Chrome/Edge; Safari/Firefox as supported)
   - **NdArray** (CPU WASM) fallback otherwise

## Build

```bash
# Prefer rustup's toolchain for wasm32
export PATH="$HOME/.cargo/bin:$PATH"

# WASM module (includes wgpu → WebGPU + NdArray)
wasm-pack build crates/spacetravlr-cnn-wasm --target web --out-dir ../../web/cnn_train/pkg --release

# UI
cd web/cnn_train && npm ci && npm run build && cd ../..

# Server
cargo build --release --features cnn-web --bin spacetravlr-cnn-web
```

## Native GPU smoke (Metal / Vulkan / DX)

```bash
cargo run -p spacetravlr-cnn-wasm --features webgpu --release --bin webgpu_smoke
```

## Browser WebGPU smoke (Chrome)

```bash
cd web/cnn_train
# requires wasm pkg built; uses system Chrome + CDP
node scripts/webgpu_browser_smoke.mjs
```

Expect `backend: "webgpu"` and a finite `wall_ms`.
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

Open `http://127.0.0.1:8787`. The UI logs whether **webgpu** or **ndarray** was selected, then use **Train in WASM**.

## API

| Endpoint | Role |
| --- | --- |
| `GET /api/info` | Dataset / defaults |
| `POST /api/prepare` | `{ genes, epochs, native_cnn? }` — Lasso dump (+ optional native CNN) |
| `GET /api/pack?gene=AICDA` | Bincode `CnnGeneTrainPack` for WASM |
| `GET /api/status` | Last job status |

## Backend selection

| Host | Backend |
| --- | --- |
| Chrome / Edge with WebGPU | Burn `Wgpu` → browser WebGPU |
| Firefox / Safari without WebGPU, or init failure | Burn `NdArray` (CPU WASM) |
| Node.js (no `navigator.gpu`) | NdArray only — call `use_ndarray_backend()` |

WASM exports (async where noted): `init_webgpu()`, `use_ndarray_backend()`, `webgpu_available()`, `active_backend_name()`, `await train_pack()`, `await smoke_train_ms()`. Training uses `into_scalar_async` so WebGPU works in the browser.
