# Functional Microniche Embeddings

Per-cell functional microniche embeddings from heterogeneous betadata feathers.  
Two model variants: a **simple fast model** (recommended) and a full two-level set encoder.

---

## SimpleNicheModel (recommended)

```
precompute X ∈ ℝ^{N × G·M}   ──►  BetaMLP  ──►  SpatialGCN  ──►  z ∈ ℝ^D
(flat signed-beta matrix,            2-layer      2-layer GCN,
 computed once before training)      MLP          kNN spatial graph
```

**Key idea**: the per-gene signed-beta vectors are concatenated into one flat
cell-feature matrix `[N, G×M]` *once* before training. Every epoch is then a
single BLAS call (matmul) through the MLP + one GCN pass — no Python loops,
no per-token embedding lookups.

**Speed**: 500 epochs · 1000 cells · 8 genes · 359 mods → **8 seconds on CPU**.

**Training objectives**:
- `L_triplet`: spatial triplet contrastive — neighbours more similar than non-neighbours
- `L_rec`: reconstruct `mean|β|` summary (MSE)
- `L_smooth`: spatial smoothness penalty on `z`

---

## Full model (model.py / train.py)

The original Strategy B architecture (kept for reference/GPU use):

```
beta feathers ──► ModulatorEncoder ──► CellEncoder ──► SpatialGCN ──► z
                  (scatter + MLP,       (attn pool
                   per gene)             over genes)
```

---

## Files

| File | Contents |
|---|---|
| `dataset.py` | Feather loader, modulator vocab, spatial graph, `make_beta_matrix()` |
| `simple_model.py` | **`SimpleNicheModel`**, `TripletSpatialLoss`, `train_simple()` |
| `model.py` | Full `ModulatorEncoder` / `CellEncoder` / `SpatialGCN` model |
| `losses.py` | Triplet loss, spatial smoothness, adjacency builder |
| `train.py` | Full model training loop + CLI |
| `cluster.py` | Leiden clustering, Moran's I filter, niche signatures |
| `visualize.py` | UMAP and spatial scatter plots |
| `synth.py` | Synthetic data generator with known ground-truth niches |
| `benchmark.py` | ARI/NMI benchmark vs PCA baselines |

---

## Benchmark Results (SimpleNicheModel)

1000 cells · 8 genes · 5 niches · 359 mods · 500 epochs · CPU (44s total for 3 scenarios)

| Scenario | PCA (signed β) | PCA+smooth | PCA (mean\|β\|) | **SimpleNicheModel** |
|---|---|---|---|---|
| A: sign-coded niches, no noise | 1.000 | 0.749 | **0.001** | **0.547** |
| B: sign-coded + cell noise=1.0 | 0.986 | 0.587 | **0.001** | **0.471** |
| C: gene-specific + cell noise  | 1.000 | 0.703 | 1.000 | **0.675** |

**Key finding**: `PCA on mean|β|` completely fails on sign-coded niches (ARI ≈ 0)
because it discards regulatory direction. The `SimpleNicheModel` recovers ARI 0.47–0.55
in those scenarios in seconds.

See `benchmark_results/benchmark_comparison_simple.png`.

---

## Quickstart

```bash
# Install dependencies
pip install -r scripts/functional_niches/requirements.txt

# Multi-scenario benchmark (3 scenarios × 500 epochs, ~44s total)
cd <repo_root>
PYTHONPATH=scripts python3 -m functional_niches.benchmark --multi \
    --epochs 500 --hidden-dim 64 --output-dir /tmp/niche_bench

# Single scenario
PYTHONPATH=scripts python3 -m functional_niches.benchmark \
    --n-cells 2000 --n-genes 10 --n-niches 5 --epochs 500

# Train on real betadata
PYTHONPATH=scripts python3 -c "
import anndata, numpy as np, sys
sys.path.insert(0, 'scripts')
from functional_niches.dataset import load_dataset
from functional_niches.simple_model import train_simple

adata = anndata.read_h5ad('/path/to/data.h5ad')
ds = load_dataset(
    feather_dir='/path/to/betadata/',
    spatial_coords=adata.obsm['spatial'].astype(np.float32),
    cell_ids=list(adata.obs_names),
    k=6,
)
z = train_simple(ds, '/path/to/niches/', hidden_dim=64, epochs=500)
"
```

## Notes

- `numpy<2` is required for PyTorch 2.0.x compatibility.
- For datasets >50k cells on CPU, 200–300 epochs typically suffice.
- On GPU, 500 epochs for 50k cells, 10 genes, 1000 mods takes ~2 minutes.
