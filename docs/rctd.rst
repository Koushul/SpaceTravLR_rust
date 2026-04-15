RCTD spatial deconvolution
===========================

RCTD is linked as an **optional Cargo feature** (enabled by default in this repository’s default feature set). It estimates cell-type mixtures per spatial spot from a reference ``.h5ad`` or matrix export.

Typical invocation:

.. code-block:: bash

   spacetravlr --rctd \
     --h5ad spatial.h5ad \
     --ref-adata reference.h5ad \
     --rctd-output ./out/deconv

GPU acceleration is available when building with ``rctd-wgpu`` and passing ``--gpu`` (see ``--help``; results may differ slightly from CPU).

For the full flag surface (Q matrices, sigma, subset files, modes), use ``spacetravlr --help`` and the **RCTD** section in the CLI.
