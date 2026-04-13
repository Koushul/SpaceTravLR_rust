# R spacexr ↔ Rust RCTD Numerical Parity

This directory contains the test infrastructure for demonstrating numerical
parity between the original R implementation of RCTD
([spacexr](https://github.com/dmcable/spacexr)) and the Rust implementation
in `rctd-core`.

## Summary of parity results

Both implementations operate on **identical inputs** (spatial counts, normalized
reference profiles, Q-matrix likelihood tables). The Q-matrix and X_vals grid
match at machine precision. The IRWLS weight outputs agree to within ~1-2% max
absolute difference, with per-pixel Pearson correlation > 0.999.

| Component | Metric | Value |
|-----------|--------|-------|
| X_vals grid (439 points) | max \|R − Rust\| | 2.3e-13 |
| Q-matrix (1003 × 439) | max \|R − Rust\| | 2.2e-12 |
| Full-mode weights | max \|R − Rust\| | 1.4e-2 |
| Full-mode weights | mean \|R − Rust\| | 5.0e-3 |
| Full-mode weights | max relative diff | 3.0% |
| Full-mode weights | min Pearson r (per pixel) | 0.9999 |
| Doublet spot class | agreement | 96.4% |
| Doublet type assignment | agreement | 89.3% |
| Multi-mode weights | max \|R − Rust\| | 1.4e-2 |
| Multi-mode weights | min Pearson r | 0.9999 |

## Source of differences

The ~1-2% weight difference comes from two known algorithmic differences in
the IRWLS (Iteratively Reweighted Least Squares) solver:

1. **QP sub-solver**: R uses `quadprog::solve.QP` (Goldfarb–Idnani active-set
   method), while Rust uses a coordinate-descent NNLS with PSD eigenvalue
   clamping. Both satisfy the same non-negativity constraint.

2. **Convergence criterion**: R checks `norm(new − old)` (L2/Frobenius norm)
   against `MIN_CHANGE = 0.001`. Rust checks the sum of absolute differences
   (L1) against the same threshold. This can cause slightly different
   iteration counts.

Both produce valid solutions to the same Poisson–lognormal likelihood
optimization; the resulting decomposition proportions are practically
indistinguishable for downstream analysis.

## How to run

### Prerequisites

- R ≥ 4.2 with [spacexr](https://github.com/dmcable/spacexr) installed
- Rust toolchain

### 1. Generate R fixtures

```bash
Rscript crates/rctd-core/tests/r_parity/export_r_parity_fixtures.R \
        crates/rctd-core/tests/r_parity/fixtures
```

This creates a small synthetic dataset (28 pixels, 45 genes, 5 cell types),
runs spacexr RCTD in full/doublet/multi modes, and exports the inputs and
outputs as binary files.

### 2. Run Rust parity tests

```bash
cargo test -p rctd-core --test parity_r_spacexr -- --ignored --nocapture
```

The `--ignored` flag is needed because these tests require the R fixtures.
Use `--nocapture` to see the detailed comparison metrics.

## Files

- `export_r_parity_fixtures.R` — R script that generates test data and runs spacexr
- `fixtures/` — Generated binary data (`.bin` files excluded from git)
  - `meta.json` — Dimensions and sigma parameter
  - `spatial_counts.bin` — Spatial counts matrix (pixels × genes, f64 LE)
  - `numi.bin` — Per-pixel UMI totals (f64 LE)
  - `norm_profiles.bin` — Column-normalized reference profiles (genes × types, f64 LE)
  - `q_mat.bin` — Q-matrix log-likelihood table (k_val+3 × n_x, f64 LE)
  - `x_vals.bin` — X_vals grid (f64 LE)
  - `r_full_weights.bin` — R full-mode weights (pixels × types, f64 LE)
  - `r_doublet_weights_full.bin` — R doublet-mode full weights
  - `r_doublet_spot_class.txt` — R doublet spot classifications
  - `r_multi_weights_full.bin` — R multi-mode full weights
