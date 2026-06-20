# Rust vs Python scaling benchmarks for SpaceTravLR

This directory holds a self-contained, reproducible scaling study of the SpaceTravLR
pipeline that compares the **Rust port** (this crate) against the original
**Python + scanpy** stack at a range of cell counts taken from
`/ix/djishnu/shared/djishnu_kor11/training_data_revision/atera_human_cervix.h5ad`
(717,576 cells × 3,484 genes).

## Experiments

1. **Preprocess + imputation scaling** — for each `N ∈ {700, 2k, 7k, 20k, 70k, 200k,
   700k}` we time the Rust pipeline (`spacetravlr --rust-process-h5ad`: QC →
   normalize_total → log1p → HVG → PCA → HNSW kNN → UMAP → Leiden → MAGIC) against the
   Python pipeline (`spacetravlr --process-h5ad`: embedded scanpy + magic-impute via
   `uv run --isolated`). Scanpy is allowed to crash on large `N`; the JSON records the
   failure mode (timeout / OOM / non-zero exit) and the plots fall back gracefully.

2. **CNN training scaling** — for the same `N` we train SpaceTravLR's 32x32
   `CellularNicheNetwork` (3 conv blocks + MLP + sigmoid output) for a fixed number of
   epochs and minibatch size, once in Rust (`src/bin/scaling_bench.rs`, burn 0.16) and
   once in Python (`bench_cnn_python.py`, PyTorch on the available GPU).

3. **Dropout sensitivity** — at a fixed `N = 20,000` we sweep input spatial-map
   dropout `p ∈ {0, 0.2, 0.5, 0.8, 0.95}` and measure both training time and final MSE
   to see how each implementation degrades with input sparsity.

## Files

| File | Role |
|---|---|
| `config.json` | Sizes, hyperparameters, output paths |
| `subsample.py` | Stratified subsample of the big `.h5ad` (X = `layers/normalized_count`, no `uns['log1p']` so both pipelines run identical work) |
| `bench_preprocess.py` | Calls `spacetravlr --rust-process-h5ad` and `--process-h5ad`, captures wall time + peak RSS via `/usr/bin/time -v` |
| `bench_cnn_python.py` | PyTorch CNN training bench |
| `src/bin/scaling_bench.rs` (sibling crate file) | Rust burn CNN training bench |
| `run_all.py` | Orchestrator: subsample → preprocess → CNN → dropout, writes a single JSON |
| `plot_results.py` | Renders the comparison plots |
| `results/results.json` | Canonical results object |
| `results/plots/*.png` | Generated comparison figures |

## How to run

```bash
# 1. Build the Rust binaries (release, includes scaling_bench + spacetravlr).
cargo build --release --bin spacetravlr --bin scaling_bench

# 2. Run the full study (subsamples are cached on disk).
python benchmarks/run_all.py --config benchmarks/config.json \
    --results-json benchmarks/results/results.json

# 3. Plot.
python benchmarks/plot_results.py \
    --results benchmarks/results/results.json \
    --out-dir benchmarks/results/plots
```

Useful overrides:

```bash
python benchmarks/run_all.py --sizes 700,2000,7000           # smaller sweep
python benchmarks/run_all.py --skip-preprocess               # CNN + dropout only
python benchmarks/run_all.py --skip-cnn --skip-dropout       # preprocess only
python benchmarks/run_all.py --cnn-device-python cpu         # force CPU for Python CNN
```

## Notes

- The benchmark deliberately stresses the *slow path*: `X` is reset to the
  un-log-transformed `normalized_count` layer (no `uns['log1p']` flag) so both Rust
  and Python pipelines re-run `normalize_total → log1p → HVG → PCA → kNN → UMAP →
  Leiden → MAGIC` end-to-end.
- The Python CNN uses CUDA when available; the Rust CNN uses the
  `Autodiff<NdArray<f32, i32>>` CPU backend so it is fully deterministic and works on
  hosts without wgpu. This gives a conservative lower bound for the Rust side — the
  same training kernel can also run on wgpu/GPU in the main `spacetravlr` binary.
- Outputs of large preprocess runs are deleted after timing to keep disk usage bounded.
- `bench_preprocess.py` shells out to `/usr/bin/time -v` (GNU time) when present for
  accurate peak RSS. On hosts without GNU time, RSS columns will be blank but timings
  are still recorded.
