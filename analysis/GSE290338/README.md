# GSE290338 analysis

Spatial Visium HD (8 µm) MC38 tumors after 12 Gy irradiation ([GSE290338](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE290338)): 24 h and 7 d post-irradiation.

## Reference (map-labels)

[GSE179936](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE179936) MC38 tumor CD45+ scRNA-seq with `singler_label` annotations (strong **Macrophages** and **T cells** / **NKT** for CD8-lineage markers).

## Reproduce

```bash
cd analysis/GSE290338

# 1. Download GEO supplementary archives (see NCBI GEO pages), place under raw/

# 2. Build subsampled spot × gene AnnData (35k spots per timepoint)
python3 build_query_h5ad.py --samples 24h,7d

# 3. Rust QC → HVG → PCA → UMAP → Leiden → MAGIC
spacetravlr --plain --rust-process-h5ad --h5ad GSE290338_query.h5ad --process-output-dir .

# 4. UMAP + Leiden figure
python3 plot_umap_leiden.py GSE290338_query_rust_processed.h5ad --out figures/umap_leiden.png

# 5. Reference + MALT
python3 prepare_reference.py
python3 -c "import anndata as ad, scanpy as sc; q=ad.read_h5ad('GSE290338_query.h5ad'); sc.pp.subsample(q,n_obs=25000,random_state=0); q.write_h5ad('GSE290338_query_malt_input.h5ad')"
spacetravlr --plain --map-labels \
  --reference GSE179936_MC38_reference.h5ad \
  --query GSE290338_query_malt_input.h5ad \
  --map-labels-outdir malt_out \
  --map-labels-groupby cell_type \
  --map-labels-extra-markers Cd8a,Cd8b1,Cd3e,Lyz2,Spp1,Cxcl9,Cxcl10,C1qa,Mrc1

# 6. Label UMAPs (join malt CSV to processed object by obs_names)
python3 plot_malt_summary.py \
  --h5ad GSE290338_query_rust_processed.h5ad \
  --labels-csv malt_out/malt_labels.csv
```

Large artifacts (`.h5ad`, `raw/`, `malt_out/`) are gitignored; scripts and this README are tracked.
