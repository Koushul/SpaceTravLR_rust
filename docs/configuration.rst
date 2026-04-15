Configuration
===============

Training reads **``spaceship_config.toml``** (Spaceship config). The CLI can override many fields; see ``spacetravlr --help`` grouped sections (Input, Training, Output, and so on).

Important sections (conceptual)
---------------------------------

* **``[data]``** — path to spatial ``.h5ad``, layer names, cluster annotation column, optional condition column.
* **``[grn]``** — TF priors, extra modulators, ligand–receptor extras, ligand caps.
* **``[spatial]``** — spatial prior parameters used when building received-ligand features.
* **``[training]``**, **``[lasso]``**, **``[cnn]``** — optimization and model head settings.
* **``[execution]``** — parallelism, output directory, optional minimal repro export.

When ``--join-output-dir`` is used, hyperparameters are taken from the existing ``spacetravlr_run_repro.toml`` in that directory rather than re-applying ``--config`` overrides for locked fields—see the long help in ``spacetravlr --help`` for multi-host and condition-split behavior.
