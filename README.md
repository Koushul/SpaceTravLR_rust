# SpaceTravLR (Rust 🦀️🚀️)

<!-- <img alt="SpaceTravLR training dashboard UI" src="data/rust_chef.png" width="600"/> -->

[![Compilation Status](https://github.com/Koushul/SpaceTravLR_rust/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/Koushul/SpaceTravLR_rust/actions/workflows/release.yml)
[![SpaceShip CI](https://github.com/Koushul/SpaceTravLR_rust/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/Koushul/SpaceTravLR_rust/actions/workflows/rust.yml)

```bash
curl -fsSL https://tinyurl.com/spacetravlr/scripts/install.sh | sh
```

Rust implementation of [SpaceTravLR](https://github.com/jishnu-lab/SpaceTravLR)

![SpaceTravLR training dashboard UI](data/demo.gif)

Tech Stack:
  - [ratatui](https://github.com/ratatui/ratatui) for User Interface
  - [foyer](https://github.com/foyer-rs/foyer) for In-memory/Disk Cache
  - [burn](https://github.com/tracel-ai/burn) for Machine Learning
  - [tokio](https://github.com/tokio-rs/tokio) for Async pipelines
  - [axum](https://github.com/tokio-rs/axum) for HTTP Service
  - [polars](https://github.com/pola-rs/polars) for efficient Data Processing
  - [wgpu](https://github.com/gfx-rs/wgpu) for seemless GPU compute
  - [rayon](https://github.com/rayon-rs/rayon) for fearless Data-Parallelism


## RCTD spatial deconvolution (optional)

RCTD is integrated as an **opt-in** Cargo feature. The Rust version also implements GPU optimization. 
SpaceTravLR's RCTD version is about ~59x faster than the R version.

Build the `spacetravlr` binary with RCTD:

```bash
cargo build -p spacetravlr --features rctd
```


```bash
spacetravlr --rctd --h5ad spatial.h5ad --ref-adata reference.h5ad --rctd-output ./out/deconv
```

## DeepSpot virtual spatial transcriptomics from H&E (optional)

For an **H&E → virtual gene expression `.h5ad`** pipeline (DeepSpot → SpaceTravLR), see
[`tools/deepspot_visium_pipeline/README.md`](tools/deepspot_visium_pipeline/README.md) and
[`deepspot_spacetravlr_summary.md`](deepspot_spacetravlr_summary.md).




