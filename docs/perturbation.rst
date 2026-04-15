In-silico perturbation
=======================

``spacetravlr-perturb`` loads a finished run (``spacetravlr_run_repro.toml`` plus betadata) and simulates expression changes for one or many genes. It shares the same runtime assumptions as the spatial viewer for reproducibility.

Modes
-------

* **TUI** (default when built with ``tui``): interactive gene choice, desired expression, propagation depth, optional cell subset from CSV.
* **Single-job batch**: ``--run-toml``, ``--gene``, ``--export`` / ``--out``, optional ``--cells-csv`` + ``--cells-csv-column``.
* **Multi-job batch**: ``--batch-toml`` and optional ``--batch-parallelism``.

Configuration can be passed as ``--config`` (or first positional argument): a small TOML that references ``run_toml`` and optional ``[data]``, ``[perturbation]``, and batch fields. See ``spacetravlr-perturb --help`` long help for concrete examples.

See also :doc:`ligand_perturbation` for how perturbation fits into ligand-centric analysis in the UI.
