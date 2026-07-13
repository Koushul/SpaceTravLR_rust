# Non-spatial tissue-structure ligands

SpaceTravLR normally computes **received ligands** with a Gaussian kernel over
spatial coordinates:

```text
received[i, l] = (1/N) Σ_j scale · exp(-d(i,j)² / 2r²) · expression[j, l]
```

For non-spatial scRNA-seq there are no coordinates, so CNN refinement is not
used (`seed_only` / `[training].mode = "seed"`). Instead we learn a **tissue
structure reference** from a matched spatial dataset and transfer neighborhood
composition.

## Idea

Group senders by cell type `t`:

```text
S[i, t] = (1/N) Σ_{j ∈ t} scale · exp(-d(i,j)² / 2r²)
received[i, l] ≈ Σ_t S[i, t] · μ[t, l]
```

Average `S` within each receiver type `c` on a spatial reference:

```text
Ŝ[c, t] = mean_{i : type(i)=c} S[i, t]
```

For a query scRNA-seq cell of type `c`:

```text
received̂[i, l] = Σ_t Ŝ[c, t] · μ_query[t, l]
```

where `μ_query` is the query dataset’s type-mean ligand expression.

The same reference also stores expected **soft** (Gaussian-weighted) and
**hard** (within-radius) neighbor counts by type — the “how many neighbors of
each type surround this cell” quantity.

## Config

```toml
[structure]
enabled = true
reference_adata = "/path/to/matched_spatial.h5ad"
# or:
# reference_path = "tissue_structure_ref.json"
# reference_cluster_annot = "cell_type"
# hard_radius = 200.0
```

Enabling structure mode forces seed-only training (no CNN).

## CLI

```bash
# Build a reusable reference
cargo run --release --bin structure_ligands -- build \
  --adata matched_spatial.h5ad --out tissue_structure_ref.json --radius 200

# Self-check: structure vs spatial Gaussian on one dataset
cargo run --release --bin structure_ligands -- validate-self \
  --adata matched_spatial.h5ad --radius 200 --n-ligands 30 --max-cells 3000
```

## Multi-dataset validation

```bash
python scripts/validate_structure_ligands.py \
  --data-dir /path/to/SpaceTravLR/data \
  --outdir results/structure_validation \
  --radius 200 --n-ligands 40 --max-cells 4000
```

The script reports Pearson / Spearman / MAE / relative MAE, neighbor-composition
cosine similarity, and an error decomposition:

| Method | Meaning |
|--------|---------|
| `type_mean_oracle` | True per-cell `S[i,t]` × type means (expression heterogeneity ceiling) |
| `structure_pooled` | Same-sample type-averaged `Ŝ` |
| `expression_matched` | Expression-kNN niche matching to transfer per-cell `S` |
| `abundance_baseline` | Type frequencies only (no architecture) |
| `structure_transfer` | Cross-sample / replicate / spatial-holdout reuse of `Ŝ` |

### Key empirical findings (r=200, 30 ligands, ≤2000 cells)

**Neighbor composition** (soft-count cosine vs spatial truth): **0.92–0.99** across
kidney, tonsil, melanoma, lymph node, and germinal center.

**Type-level received ligands** (`type_pearson_mean`, the natural non-spatial estimand):

| Dataset | structure_pooled | abundance_baseline | type_mean_oracle |
|---------|-----------------:|-------------------:|-----------------:|
| Germinal center | 0.993 | 0.714 | 0.993 |
| Tonsil | 0.964 | 0.692 | 0.964 |
| Melanoma | 0.934 | 0.812 | 0.934 |
| Kidney rep1 | 0.935 | 0.894 | 0.935 |
| Lymph node | 0.770 | 0.257 | 0.770 |

On organized lymphoid tissues, type-averaged structure recovers the oracle and
beats abundance-only baselines by a large margin. Kidney is more mixed, so
abundance nearly matches structure.

**Spatial holdout** (learn `Ŝ` on left half, apply to right half without coords):
lymph node type Pearson **0.40 vs −0.03** (abundance); tonsil **0.55 vs 0.34**;
germinal center **0.54 vs 0.38**. Structure matrices remain highly similar across
halves (cosine **0.90–0.99**).

**Negative controls**: transfers across mismatched cell-type vocabularies
(tonsil↔melanoma, kidney↔tonsil, mouse↔human lymphoid) correctly abort with
insufficient overlapping labels.

Cell-level Pearson for type-pooled predictions is intentionally lower: without
coordinates one cannot recover within-type niche variation, only the
type-conditional expectation (plus optional expression-matched niche transfer).

