Quickstart
==========

1. Place ``spaceship_config.toml`` next to your data (or pass ``--config`` / ``--h5ad``).
2. Ensure the AnnData has the fields your config expects (cluster labels, layers such as ``imputed_count``; the CLI can auto-prepare data when appropriate).
3. Start training:

.. code-block:: bash

   spacetravlr --h5ad path/to/spatial.h5ad

4. For logs without the full-screen dashboard (when built with ``tui``):

.. code-block:: bash

   spacetravlr --plain --h5ad path/to/spatial.h5ad

5. Generate the HTML run summary without training:

.. code-block:: bash

   spacetravlr run-summary --h5ad path/to/spatial.h5ad

Outputs default to a dated directory derived from the AnnData stem unless ``output_dir`` is set in config or ``--output-dir`` is passed on the CLI.
