# SpaceTravLR (Rust 🦀️🚀️)

<img alt="SpaceTravLR training dashboard UI" src="data/rust_chef.png" width="600"/>

[![Compilation Status](https://github.com/Koushul/SpaceTravLR_rust/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/Koushul/SpaceTravLR_rust/actions/workflows/release.yml)
[![SpaceShip CI](https://github.com/Koushul/SpaceTravLR_rust/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/Koushul/SpaceTravLR_rust/actions/workflows/rust.yml)

```bash
curl -fsSL https://tinyurl.com/spacetravlr/scripts/install.sh | sh
```

Rust implementation of [SpaceTravLR](https://github.com/jishnu-lab/SpaceTravLR)

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

RCTD is integrated as an **opt-in** Cargo feature. The vendored `rctd-core` and helper crate are licensed under **GPL-3.0-or-later**. If you build or distribute a binary that links this code, the **combined work** is subject to the GPL (see `NOTICE`). The rest of the repository remains MIT-licensed when RCTD features are not enabled.

Build the `spacetravlr` binary with RCTD:

```bash
cargo build -p spacetravlr --features rctd
```

For GPU (WebGPU / `wgpu`, **f32** inside RCTD; not bit-identical to the CPU **f64** path):

```bash
cargo build -p spacetravlr --features rctd,rctd-wgpu
```

Example (spatial and reference AnnData; same gene alignment and Q-matrix flow as upstream `rctd run`):

```bash
spacetravlr --rctd --h5ad spatial.h5ad --ref-adata reference.h5ad --rctd-output ./out/deconv
```

Use `spacetravlr --help` for flags under the **RCTD** section (`--rctd-mode`, `--sigma`, `--rctd-batch-size`, `--q-matrices`, `--gpu`, etc.). A `spaceship_config.toml` is not required for `--rctd`.


![SpaceTravLR training dashboard UI](data/demo.gif)

