# SpaceTravLR (Rust 🦀️🚀️)

<!-- <img alt="SpaceTravLR training dashboard UI" src="data/rust_chef.png" width="600"/> -->

[![Compilation Status](https://github.com/Koushul/SpaceTravLR_rust/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/Koushul/SpaceTravLR_rust/actions/workflows/release.yml)
[![SpaceShip CI](https://github.com/Koushul/SpaceTravLR_rust/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/Koushul/SpaceTravLR_rust/actions/workflows/rust.yml)

Rust implementation of [SpaceTravLR](https://github.com/jishnu-lab/SpaceTravLR) — spatial gene regulatory network inference and in-silico perturbation from Visium-style `.h5ad`.

```bash
curl -fsSL https://tinyurl.com/spacetravlr/scripts/install.sh | sh
```

For supported platforms, PATH setup, self-updates, and troubleshooting, see **[install.md](install.md)** (recommended over piping `curl` directly to `sh`: use `-o` then `sh` as shown there).

**Documentation:** [spacetravlr.readthedocs.io](https://spacetravlr.readthedocs.io/) via [Read the Docs](https://docs.readthedocs.com/platform/stable/intro/mkdocs.html) ([`.readthedocs.yaml`](.readthedocs.yaml)). Local preview: `pip install -r docs/requirements.txt && mkdocs serve`.

![SpaceTravLR training dashboard UI](data/demo.gif)

## Tech stack

- [ratatui](https://github.com/ratatui/ratatui) — training / perturbation TUIs
- [foyer](https://github.com/foyer-rs/foyer) — in-memory / disk cache (spatial viewer)
- [burn](https://github.com/tracel-ai/burn) — ML (CNN + backends)
- [tokio](https://github.com/tokio-rs/tokio) — async HTTP and pipelines
- [axum](https://github.com/tokio-rs/axum) — HTTP services (`spatial_viewer`, `umap_lab`)
- [polars](https://github.com/pola-rs/polars) — data processing
- [wgpu](https://github.com/gfx-rs/wgpu) — GPU compute where enabled
- [rayon](https://github.com/rayon-rs/rayon) — data parallelism
