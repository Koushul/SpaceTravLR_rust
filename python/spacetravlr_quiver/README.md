# spacetravlr-quiver

Pure-Rust UMAP transition / quiver fields (velocyto `colDeltaCor` + SpaceOracle Cartography), exposed to Python via PyO3.

## Layout

| Path | Role |
|------|------|
| `crates/spacetravlr-transition` | Core math (colDeltaCor, softmax P, null subtract, project, grid) |
| `crates/spacetravlr-py` | PyO3 extension `spacetravlr_quiver._lib` |
| `python/spacetravlr_quiver/` | Plotting, UMAP helpers, parameter sweeps |

`spacetravlr::transition_umap` re-exports the transition crate (spatial_viewer / tests unchanged).

## Install (editable)

```bash
cd crates/spacetravlr-py
maturin develop --uv
```

## API (Python)

```python
import numpy as np
import spacetravlr_quiver as sq

grid = sq.compute_transition_grid(
    expr,      # (n_cells, n_genes) float64
    delta,     # same shape; cartography uses round(pert - baseline, 3)
    umap,      # (n_cells, 2)
    n_neighbors=150,
    temperature=0.05,
    remove_null=True,
    unit_directions=False,          # cartography default
    grid_scale=1.0,
    vector_scale=0.85,
    null_subtract_mode="raw",       # SpaceOracle parity; or "clip_renorm"
)
# grid["grid_points_x/y"], grid["u/v"], grid["cell_u/v"], ...
```

UMAP: `ensure_umap_embedding(adata, prefer_rust=True)` uses `spacetravlr --umap` when available (rust-process), else umap-learn with the same knobs (`n_neighbors`, `min_dist`, …).

## Sweep plots

```bash
python -m spacetravlr_quiver.sweep --quick
# full grid of nn / grid_scale / unit_directions / null mode / UMAP settings:
python -m spacetravlr_quiver.sweep \
  --pert-dir /tmp/tonsil_full_seed_20260805/perturbations \
  --out-dir /tmp/tonsil_full_seed_20260805/perturbations/sweep_rust
```

## Tests

```bash
cargo test -p spacetravlr-transition
pytest python/spacetravlr_quiver/tests/test_parity.py
```
