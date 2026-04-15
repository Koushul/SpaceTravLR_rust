Spatial viewer
===============

The ``spatial_viewer`` binary (Cargo feature ``spatial-viewer``) serves an **Axum**-based HTTP API and static UI for exploring trained runs: spots, genes, perturbation hooks, and cached tiles. Build it with:

.. code-block:: bash

   cargo build --release -p spacetravlr --features spatial-viewer --bin spatial_viewer

Documentation for MCP-driven workflows may live in repository agent skills; this page is the stable entry point for Read the Docs users discovering the component from the user guide.
