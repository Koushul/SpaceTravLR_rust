SpaceTravLR
===========

**SpaceTravLR** infers spatial gene regulatory networks from Visium-style spatial single-cell data shipped as AnnData (``.h5ad``). This repository is the high-performance `Rust <https://www.rust-lang.org/>`_ implementation: GPU-accelerated training with `Burn <https://github.com/tracel-ai/burn>`_, optional RCTD deconvolution, terminal and web tooling, and in-silico perturbation workflows.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Install
      :link: installation
      :link-type: doc

      Prebuilt binaries and ``cargo`` builds.

   .. grid-item-card:: CLI utilities
      :link: cli_utilities
      :link-type: doc

      Plot, preprocess, impute, species inference, CellOracle priors, and more.

   .. grid-item-card:: Training
      :link: training
      :link-type: doc

      ``spaceship_config.toml``, dashboards, and distributed runs.

   .. grid-item-card:: Perturbation
      :link: perturbation
      :link-type: doc

      ``spacetravlr-perturb`` batch and TUI modes.

Where to look next
------------------

* **Rust API** on `docs.rs <https://docs.rs/spacetravlr>`_ for library types and modules.
* **Source** on `GitHub <https://github.com/Koushul/SpaceTravLR_rust>`_.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User guide

   overview
   installation
   quickstart
   training
   configuration
   cli_utilities
   perturbation
   rctd
   spatial_viewer

.. toctree::
   :hidden:
   :caption: Concepts

   ligand_receptor_interactions
   ligand_perturbation
