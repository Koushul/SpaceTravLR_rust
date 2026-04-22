Training with ``spacetravlr``
==============================

The primary binary loads :doc:`configuration`, applies CLI overrides, and fits spatial GRN models gene by gene. Progress and diagnostics are shown in a **Ratatui** dashboard when the ``tui`` feature is enabled; use ``--plain`` for line-oriented logging.

Common flags
------------

``--h5ad``
   Override ``[data].adata_path``.

``--genes`` / ``--max-genes``
   Restrict which target genes are trained.

``--parallel``
   Worker threads (one active gene per worker).

``--training-mode``
   ``full`` or ``seed`` CNN modes (see CLI help).

``--output-dir`` / ``--join-output-dir``
   Fresh run directory vs. resume or multi-host shared storage.

``--condition``
   Train per biological condition under ``conditions/<value>/``.

``--skip-auto-adata-prep``
   Disable automatic Scanpy / imputation when metadata or layers are missing.

Subcommands
-----------

``run-summary``
   Writes ``spacetravlr_run_summary.html`` from AnnData + config + optional manifest (see :doc:`quickstart`).

Self-update (optional feature)
------------------------------

With the ``self-update`` feature, ``--update`` downloads the latest release and replaces adjacent binaries. ``--update-version <tag>`` pins a release tag.
