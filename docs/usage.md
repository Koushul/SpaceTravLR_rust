# Quick reference commands

Quick reference for **SpaceTravLR** CLIs. Full flags: **`spacetravlr --help`**, **`spacetravlr-perturb --help`**, etc.

## Run SpaceTravLR

Launching the SpaceTravLR UI will prompt you to specify an input h5ad

```bash
spacetravlr
```

--8<-- "docs/snippets/spacetravlr_tui.html"

Or you can specify one using `--h5ad`
```bash
spacetravlr --h5ad /storage/tissues/adata.h5ad \
  --output-dir /storage/outputs/spacetravlr_output
```

To launch multiple parallel jobs, use `--join-output-dir`

```bash
spacetravlr --join-output-dir  /storage/outputs/spacetravlr_output
```




## Inspect `.h5ad`

Blazing fast look inside any .h5ad or .h5 without loading the whole object into memory.

```bash
spacetravlr --peek adata.h5ad
```

--8<-- "docs/snippets/peek.html"

Load a single **`obs`** column and print ranked value counts (counts and %).

```bash
spacetravlr --peek path/to/data.h5ad --obs cell_type
```

## UMAP

Plot a UMAP scatter in the terminal.

```bash
spacetravlr --plot-umap path/to/data.h5ad
```

--8<-- "docs/snippets/plotumap.html"

```bash
spacetravlr --h5ad path/to/data.h5ad --plot-umap --obs cell_type
```

```bash
spacetravlr --plot-umap path/to/data.h5ad --leiden
```

## Preprocess
SpaceTravLR reimplements the core Scanpy preprocessing pipeline in pure rust to allow scalable processing of datasets beyond 1 million cells. Optionally, for imputation, a rust version of the MAGIC (Markov Affinity-based Graph Imputation of Cells) algorithm is also built-in.

QC → normalize → HVG → PCA → KNN → UMAP → Leiden / MAGIC.

```bash
spacetravlr --rust-process-h5ad --h5ad adata.h5ad 
```

## Auto-Annotation

Transfers labels from a reference `.h5ad` to a query `.h5ad`

```bash
spacetravlr --map-labels \
  --reference ref.h5ad \
  --query query.h5ad \
  --map-labels-outdir ./malt_out
```

## Auto-Annotation

Scan **`*_betadata.feather`** under a run and aggregate β into a multi–cell-type interaction table.

```bash
spacetravlr collect-interactions \
  --run-toml /path/run/spacetravlr_run_repro.toml
```


# Extras

## `umap_lab` binary (feature `umap-lab`)


```bash
cd web/umap_lab && npm ci && npm run build
cargo build --features umap-lab --bin umap_lab
./target/debug/umap_lab --port 8765 --static-dir web/umap_lab/dist
```


## RCTD (`--rctd`)

```bash
spacetravlr --rctd \
  --h5ad spatial.h5ad \
  --ref-adata reference.h5ad \
  --rctd-output ./out/deconv
```