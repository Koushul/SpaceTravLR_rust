Ligand-oriented perturbation
=============================

**In-silico perturbation** lets you change a gene’s baseline expression and propagate effects through the trained spatial model, producing simulated profiles for comparison with baseline betadata. The primary tool is ``spacetravlr-perturb`` (:doc:`perturbation`); the spatial viewer can orchestrate similar workflows from the browser when connected to a run.

Typical inputs are:

* ``spacetravlr_run_repro.toml`` from a completed training directory.
* Feather betadata artifacts for the same run.
* A target gene symbol and optional spot / cell subset.

For batch or HPC use, prefer ``--export`` / ``--batch-toml`` non-interactive modes. The spatial viewer links to this page for contextual help.
