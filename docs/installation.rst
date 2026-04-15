Installation
============

Binary install
--------------

The README documents a one-line installer:

.. code-block:: bash

   curl -fsSL https://tinyurl.com/spacetravlr/scripts/install.sh | sh

This places ``spacetravlr``, ``spacetravlr-perturb``, and related binaries on your ``PATH`` when the script completes successfully.

Build from source
-----------------

Requires **Rust 1.86+** (see ``rust-version`` in ``Cargo.toml``).

Default build (TUI, self-update, RCTD):

.. code-block:: bash

   git clone https://github.com/Koushul/SpaceTravLR_rust.git
   cd SpaceTravLR_rust
   cargo build --release -p spacetravlr

Optional components
~~~~~~~~~~~~~~~~~~~

* **RCTD** is already included in default features; for GPU RCTD paths build with ``--features rctd,rctd-wgpu``.
* **Spatial viewer** HTTP service: ``cargo build --release -p spacetravlr --features spatial-viewer --bin spatial_viewer``.
* **Minimal / CI** builds can disable default features; see ``Cargo.toml`` feature table.

Verify
------

.. code-block:: bash

   spacetravlr --help
   spacetravlr-perturb --help
