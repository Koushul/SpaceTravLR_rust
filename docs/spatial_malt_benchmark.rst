Spatial MALT real-data benchmark
================================

``scripts/spatial_malt_real_benchmark.py`` runs a reproducible train/test
benchmark on the public Open Problems seqFISH Mouse Organogenesis dataset
(``19402 x 351``). The script downloads the AnnData file, creates stratified
reference/query splits, builds lightweight seed-style betadata features from
each cell's local spatial neighborhood, and runs
``spacetravlr --map-labels --map-labels-spatial`` with
``--map-labels-no-leiden`` so the benchmark excludes adaptive Leiden cluster
enrichment/mapping.

Full-data 50/50 split command:

.. code-block:: bash

   uv run --isolated \
     --with 'numpy<2' --with 'pandas>=2.2' --with 'anndata>=0.11' \
     --with scanpy --with scikit-learn --with pyarrow --with requests \
     --with torch --with leidenalg --with igraph \
     python scripts/spatial_malt_real_benchmark.py \
       --out-dir /tmp/spacetravlr_seqfish_full_50_50_glorious2 \
       --spacetravlr python \
       --cells-per-type 0 --n-types 0 --train-fraction 0.5 \
       --genes-per-type 4 --neighbor-k 8

Metrics from that run:

================  ========  =================  ========  ===============
method            accuracy  balanced_accuracy  ARI       dotplot_mean_r2
================  ========  =================  ========  ===============
beta_knn          0.822550  0.643855           0.787832  0.895773
glorious          0.857432  0.683319           0.817335  0.920941
knn               0.832002  0.638101           0.790454  0.902739
ldl               0.854619  0.662892           0.815833  0.955021
malt              0.831552  0.681897           0.778450  0.950771
spatial_malt      0.852031  0.672629           0.812264  0.916087
================  ========  =================  ========  ===============

The full-data split shows expression KNN is a strong within-dataset baseline.
The per-cell anchor loss lets MALT retain most of KNN's held-out accuracy while
still improving the dotplot/profile objective. The label distribution learner
(``ldl``) uses an scLDL-style concentration/evidence model: it learns label
belief masses plus a background uncertainty term from expression PCA, UMAP,
spatial coordinates, spatial priors, and seed betadata features. In this run
the learned mean background mass was ``0.189``. LDL improves over KNN on
accuracy, ARI, and dotplot/profile R² in this 50/50 split. The ``glorious``
row is an adaptive probabilistic blend of KNN, anchored MALT, scLDL, beta KNN,
and spatial priors; it has the best held-out accuracy, balanced accuracy, and
ARI while keeping dotplot/profile R² above expression KNN.
