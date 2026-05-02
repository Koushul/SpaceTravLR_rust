Spatial MALT real-data benchmark
================================

``scripts/spatial_malt_real_benchmark.py`` runs reproducible train/test
benchmarks on annotated Open Problems spatial AnnData datasets. The script
downloads the AnnData file, creates stratified reference/query splits, builds
lightweight seed-style betadata features from each cell's local spatial
neighborhood, and runs ``spacetravlr --map-labels --map-labels-spatial`` with
``--map-labels-no-leiden`` so the benchmark excludes adaptive Leiden cluster
enrichment/mapping. Use ``--spacetravlr python`` to run
``scripts/malt_label_transfer.py`` directly when changing only Python code.

Full-data 50/50 seqFISH Mouse Organogenesis command:

.. code-block:: bash

   uv run --isolated \
     --with 'numpy<2' --with 'pandas>=2.2' --with 'anndata>=0.11' \
     --with scanpy --with scikit-learn --with pyarrow --with requests \
     --with torch --with leidenalg --with igraph \
     python scripts/spatial_malt_real_benchmark.py \
       --dataset seqfish_mouse_organogenesis \
       --out-dir /tmp/spacetravlr_seqfish_full_50_50_glorious2 \
       --spacetravlr python \
       --cells-per-type 0 --n-types 0 --train-fraction 0.5 \
       --genes-per-type 4 --neighbor-k 8

Metrics from that run (21 labels):

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
The full per-cell glorious distribution is written to
``glorious_probabilities.csv`` with columns ``obs_name``, one probability column
per cell type, and ``background``. Cell-type probabilities are scaled by
``1 - background`` so each row sums to one across labels plus background.

Full-data 50/50 MERFISH Mouse Cortex command:

.. code-block:: bash

   uv run --isolated \
     --with 'numpy<2' --with 'pandas>=2.2' --with 'anndata>=0.11' \
     --with scanpy --with scikit-learn --with pyarrow --with requests \
     --with torch --with leidenalg --with igraph \
     python scripts/spatial_malt_real_benchmark.py \
       --dataset merfish_mouse_cortex \
       --out-dir /tmp/spacetravlr_merfish_mouse_cortex_glorious \
       --spacetravlr python \
       --cells-per-type 0 --n-types 0 --min-cells-per-type 20 \
       --train-fraction 0.5 --genes-per-type 4 --neighbor-k 8

Metrics from that run (21 labels after filtering rare labels with fewer than
20 cells):

================  ========  =================  ========  ===============
method            accuracy  balanced_accuracy  ARI       dotplot_mean_r2
================  ========  =================  ========  ===============
beta_knn          0.883207  0.829209           0.802041  0.945467
glorious          0.926778  0.884284           0.865286  0.954609
knn               0.925870  0.877806           0.871644  0.968416
ldl               0.885477  0.786295           0.801938  0.971878
malt              0.924962  0.875765           0.866117  0.967638
spatial_malt      0.910136  0.858308           0.830697  0.945853
================  ========  =================  ========  ===============

On MERFISH mouse cortex, the glorious ensemble slightly improves held-out
accuracy and balanced accuracy over expression KNN, while KNN remains best on
ARI and dotplot/profile R².
