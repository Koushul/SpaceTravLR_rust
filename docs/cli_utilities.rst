CLI convenience utilities
==========================

The ``spacetravlr`` binary groups one-off **Utility** actions that exit before training. They share common inputs such as ``--h5ad``, ``--config``, and ``--process-output-dir`` where noted. Run ``spacetravlr --help`` for the full, versioned list.

Terminal plots
--------------

``--plot-h5ad``
   Prints a **terminal spatial scatter** using ``obsm['spatial']`` and colors spots by the cluster column from config (``[data].cluster_annot``) unless overridden. Requires ``--h5ad`` pointing to an existing file. Fails clearly if spatial coordinates are missing.

   Example:

   .. code-block:: bash

      spacetravlr --h5ad data/slide.h5ad --plot-h5ad

``--plot-umap``
   After optional auto-preprocessing, prints a **terminal UMAP** from ``obsm['X_umap']`` with the same cluster coloring semantics. Requires ``--h5ad``.

   .. code-block:: bash

      spacetravlr --h5ad data/slide.h5ad --plot-umap

   If ``X_umap`` is missing, the default is to run the **Rust** preprocess leg (same as ``--umap`` scope). ``--plot-umap-backend scanpy`` is **legacy** (embedded ``full_preprocess``); prefer ``--process-h5ad`` when you need the full Scanpy pipeline.

``--plot-umap-backend`` (``rust`` \| ``scanpy``)
   Only applies when ``--plot-umap`` must build UMAP coordinates. ``scanpy`` is deprecated for new workflows.

``--umap`` / ``--leiden`` / ``--rust-magic``
   Rust-only: write ``<stem>_prep_*.h5ad`` under ``--process-output-dir`` (default cwd). Combine as needed. ``--rust-magic`` is Rust clusterwise MAGIC into ``layers['imputed_count']`` (not the same as ``--impute``, which is Scanpy-only on existing ``normalized_count``). ``--rust-n-top-hvg`` and ``--rust-n-neighbors`` apply.

AnnData preprocessing and imputation
--------------------------------------

``--process-h5ad`` (alias ``--process_h5ad``)
   Full **Scanpy-oriented pipeline** driven via ``uv`` / Python: QC, graph, UMAP / Leiden where applicable, cluster-wise MAGIC imputation, and writes ``<stem>_processed.h5ad``. When configuration and GRN settings allow, also writes received-ligand structures (see CLI help text). Always requires ``--h5ad``.

``--impute``
   **Imputation only**: cluster-wise MAGIC on ``layers["normalized_count"]`` → ``<stem>_imputed.h5ad``. Expects ``cell_type`` or ``leiden`` (and ``--h5ad``).

``--process-output-dir``
   Directory for ``*_processed.h5ad`` or ``*_imputed.h5ad`` (default: current working directory).

``--magic-batch-obs``
   Run MAGIC per (cell type or Leiden) × this ``obs`` column. If omitted but ``--condition`` is set, the condition column becomes the batch axis for MAGIC.

``--skip-spatial-microns``
   With ``--process-h5ad``, skip heuristic rescaling of spatial coordinates to microns.

``--spatial-species`` / ``--spatial-microns-target-um``
   Species hint (``human`` / ``mouse``) and optional override for assumed median k-NN distance when converting spatial units.

Metadata and priors
---------------------

``--infer-species``
   Prints inferred **human vs. mouse** from gene symbols in ``var`` and exits (requires ``--h5ad``).

``--celloracle [PATH]``
   **CellOracle-style TF prior** inference only: reads AnnData, runs the Bayesian ridge GRN with SpaceTravLR priors, writes a Feather with ``(source, target, cell_type)``, then exits. Pass ``PATH`` to the ``.h5ad``, or omit it and use the same ``--h5ad`` you would use for training.

Related flags:

* ``--celloracle-output`` — explicit Feather path.
* ``--celloracle-output-dir`` — directory for default naming and auto preprocess output.
* ``--celloracle-layer`` (default ``imputed_count``).
* ``--celloracle-skip-preprocess`` — do not auto-impute or patch AnnData first.
* ``--celloracle-per-cluster`` / ``--celloracle-obs-key`` — per-group networks vs. global.
* ``--celloracle-species``, ``--celloracle-network-data-dir``, ``--celloracle-p-max``, ``--celloracle-threads``.

RCTD (optional; default feature set)
------------------------------------

When built with the ``rctd`` feature, **RCTD** spatial deconvolution is available and runs **before** training when requested:

* ``--rctd`` — enable RCTD mode (requires spatial ``--h5ad`` and ``--ref-adata``).
* Reference and parity knobs: ``--ref-rows-are-types``, ``--cell-type-col``, ``--rctd-mode``, ``--rctd-output``, ``--q-matrices``, ``--sigma``, TSV overrides for Q / X / nUMI, batch size, and more (see ``--help`` under **RCTD**).

Install and self-update
-----------------------

* ``--update`` / ``--update-version`` — opt-in binary refresh (requires ``self-update`` feature).

Other frequently used flags (not utility-only)
------------------------------------------------

These are not “exit before train” utilities but are often used alongside data prep:

* ``--skip-auto-adata-prep`` — do not launch automatic Scanpy / imputation when layers or ``cell_type`` are missing.
* ``run-summary`` subcommand — HTML report without fitting models (see :doc:`quickstart`).

For **perturbation** and batch simulation, use the separate ``spacetravlr-perturb`` binary (:doc:`perturbation`).
