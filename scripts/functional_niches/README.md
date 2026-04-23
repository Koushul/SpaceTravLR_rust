# Functional Microniche Embeddings

Strategy B two-level set encoder for per-cell **functional microniche** embeddings from heterogeneous betadata feathers.

## Architecture

```
beta feathers ──► ModulatorEncoder ──► CellEncoder ──► SpatialGNN ──► z ∈ ℝ^D
                  (scatter + MLP)      (attn pool     (2-layer GCN,
                  per gene             over genes)     kNN graph)
```

**ModulatorEncoder** — shared across all genes. Scatters signed beta values into a dense `[N, n_mods_total]` matrix, then applies a learned MLP projection. Preserves sign information (critical for antagonistic regulation patterns).

**CellEncoder** — aggregates G gene summaries with learned per-gene attention weights. Gene identity is encoded via a learned gene embedding.

**SpatialGNN** — 2-layer GCN on the kNN spatial graph smooths cell embeddings using spatial context.

**Training losses**:
- `L_triplet`: spatial neighbour contrast — neighbours should be more similar than non-neighbours
- `L_rec`: reconstruct mean `|beta|` summary (reconstruction target)
- `L_smooth`: spatial smoothness penalty on z

## Files

| File | Contents |
|---|---|
| `dataset.py` | Feather loader, modulator vocab, spatial graph builder, `FunctionalNicheDataset` |
| `model.py` | `ModulatorEncoder`, `CellEncoder`, `SpatialGNN`, `FunctionalNicheModel` |
| `losses.py` | Triplet spatial contrastive loss, reconstruction MSE, spatial smoothness |
| `train.py` | Full-batch training loop + CLI |
| `cluster.py` | Leiden clustering, Moran's I spatial coherence filter, niche signatures |
| `visualize.py` | UMAP and spatial scatter plots |
| `synth.py` | Synthetic data generator with known ground-truth niches |
| `benchmark.py` | ARI/NMI comparison vs PCA baselines |

## Benchmark Results

Three synthetic scenarios (200 cells, 5 genes, 3 niches, 98 modulators, 300 epochs, CPU):

| Scenario | PCA (raw β) | PCA+smooth | PCA (|β|) | **FuncNiche Model** |
|---|---|---|---|---|
| A: sign-coded niches | 1.000 | 0.603 | **0.004** | **0.551** |
| B: sign-coded + cell noise | 1.000 | 0.493 | **0.002** | **0.400** |
| C: gene-specific + noise | 1.000 | 0.693 | 1.000 | 0.456 |

**Key finding**: PCA on `mean|beta|` (rec_target) completely fails on sign-coded niches (ARI ≈ 0) because it discards the regulatory direction. The FuncNiche model preserves sign information through its signed beta encoder, recovering ARI 0.40–0.55 where the magnitude baseline scores 0.

See `benchmark_results/benchmark_comparison.png` for the chart.

## Quickstart

```bash
# Install dependencies
pip install -r scripts/functional_niches/requirements.txt

# Run benchmark on synthetic data
cd <repo_root>
PYTHONPATH=scripts python3 -m functional_niches.benchmark --multi \
    --epochs 300 --hidden-dim 64 --output-dir /tmp/niche_bench

# Train on real betadata
PYTHONPATH=scripts python3 -m functional_niches.train \
    --feather-dir /path/to/run_output/ \
    --h5ad /path/to/data.h5ad \
    --epochs 600 --hidden-dim 64 \
    --output-dir /path/to/niches/
```

## Notes

- Full-batch training (all cells in one forward pass). For datasets >10k cells on CPU, reduce epochs or use a GPU.
- On GPU, ~50 epochs/minute for Visium-scale (5k cells, 10 genes, 1000 modulators).
- `numpy<2` required for PyTorch 2.0.x compatibility.
