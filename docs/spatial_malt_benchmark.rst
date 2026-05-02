Spatial MALT real-data benchmark
================================

``scripts/spatial_malt_real_benchmark.py`` runs a reproducible train/test
benchmark on the public Open Problems seqFISH Mouse Organogenesis dataset
(``19402 x 351``). The script downloads the AnnData file, samples annotated
cell types, creates stratified reference/query splits, builds lightweight
seed-style betadata features from each cell's local spatial neighborhood, and
runs ``spacetravlr --map-labels --map-labels-spatial``.

Example command used for the PR benchmark:

.. code-block:: bash

   uv run --isolated \
     --with 'numpy<2' --with 'pandas>=2.2' --with 'anndata>=0.11' \
     --with scanpy --with scikit-learn --with pyarrow --with requests \
     python scripts/spatial_malt_real_benchmark.py \
       --out-dir /tmp/spacetravlr_seqfish_real_benchmark \
       --spacetravlr ./target/debug/spacetravlr \
       --cells-per-type 40 --n-types 8 --train-fraction 0.6 \
       --genes-per-type 4 --neighbor-k 8

Metrics from that run:

================  ========  =================  ========  ===============
method            accuracy  balanced_accuracy  ARI       dotplot_mean_r2
================  ========  =================  ========  ===============
beta_knn          0.765625  0.765625           0.541880  0.558157
knn               0.734375  0.734375           0.463668  0.545525
malt              0.460938  0.460938           0.159445  0.643714
spatial_malt      0.804688  0.804688           0.601333  0.562460
================  ========  =================  ========  ===============

The key comparison for label transfer is accuracy/ARI on held-out query cells:
SpaceTravLR betadata improves over expression KNN, and spatial MALT improves
over both.
