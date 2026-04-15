Overview
========

SpaceTravLR models how spatial context and signaling (including ligand–receptor programs) relate to gene regulation in tissue. The Rust toolchain is built for real datasets: parallel gene-wise training, reproducible run metadata, Feather exports for downstream analysis, and optional integration with RCTD for cell-type deconvolution on spots.

Typical workflow
------------------

1. Prepare or process a spatial ``.h5ad`` (see :doc:`cli_utilities`).
2. Configure ``spaceship_config.toml`` (see :doc:`configuration`).
3. Run ``spacetravlr`` to train per-gene models and write ``*_betadata.feather`` artifacts.
4. Summarize or explore results (HTML run summary, :doc:`spatial_viewer`, or your own Polars / Scanpy pipeline).
5. Run :doc:`perturbation` to simulate knockdowns or expression shifts against a finished run.

Design notes
------------

* **AnnData on disk** via the Rust ``anndata`` stack; outputs use gzip-friendly HDF5 settings for interoperability with Python.
* **Compute**: WGPU-backed training by default; CLI flags tune Lasso, CNN head, and parallelism.
* **Reproducibility**: runs can emit ``spacetravlr_run_repro.toml`` for multi-host joins and condition splits.
