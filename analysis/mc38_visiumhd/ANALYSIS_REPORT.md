# MC38 Visium HD MERCI Analysis Report (subQ-1)

Dataset: **subQ-1** — MC38 subcutaneous tumor, Visium HD, metastatic tumor-cell perturbation library ([SPAC-seq](https://spac.pku-genomics.org/#/download)).

## Methods summary

1. **Segmentation:** Used Space Ranger cell-level `filtered_feature_cell_matrix.h5` and `graphclust_annotated_cell_segmentations.geojson` from the SPAC segmentation bundle (~298k cells).
2. **Cell typing:** Marker scoring (tumor / immune / myeloid / fibroblast).
3. **MERCI:** Python port of `MERCI_LOO_MT_est` + `MERCI_ReceiverPre`. DNA rank uses donor-mt expression signature proxy (BAMs not in SPAC download).
4. **Spatial:** Squidpy neighborhood enrichment and immune proximity analysis on 2000 subsampled tumor cells.

## Key findings

### Mitochondrial transfer is detectable in the tumor population

- **Rcm > 1** at all rank cutoffs (10–80%), indicating non-random overlap between DNA-proxy and RNA ranks — consistent with true receivers among tumor cells (MERCI significance test).

### Receiver prevalence

- **789 / 1999 (39.5%)** subsampled tumor cells called receivers at top-50% DNA∩RNA rank cutoff.
- Analysis restricted to **tumor receiver candidates**; immune/myeloid cells served as donors (101,593 cells in full dataset).

### Microniche specificity

Spatial graph clusters enriched for receivers:

| Cluster | Receiver fraction |
|---------|-------------------|
| Cluster-7 | 71% |
| Cluster-10 | 67% |
| Cluster-5 | 60% |
| Cluster-1 | 50% |

Clusters 2–3 show lower receiver fractions (~29–39%), suggesting **spatially restricted microniches** where mitochondrial transfer is more frequent — potentially immune-invaded tumor edges vs core.

### Cell types as donors vs receivers

- **Donors:** immune + myeloid (by MERCI design).
- **Receivers:** exclusively tumor cells among candidates; no immune cells scored as receivers in this configuration (expected for MC38 tumor-dominant tissue).
- Fibroblasts were not in the receiver candidate set; future work could include stromal cells as candidate receivers.

### Impact on receiver tumor cells

| Metric | Receiver | non-Receiver | Notes |
|--------|----------|--------------|-------|
| Donor_MT_frac (median) | 0.473 | 0.476 | No increase; RNA deconvolution alone does not separate strongly |
| % mitochondrial UMIs | 0.97% | 0.98% | Similar |
| Stress score (Hspa1a, Atf4, …) | −0.097 | −0.094 | MW p = 0.27, not significant |

Receivers do **not** show elevated bulk mitochondrial content or acute stress vs non-receivers in this snapshot. Transfer may be subtle at transcriptome level, or captured mainly in the integrated MERCI rank rather than raw MT%.

### Spatial organization

- Neighborhood enrichment plots (`figures/nhood_enrichment_mt_receiver.png`) characterize receiver spatial clustering.
- Immune proximity analysis compares distance from receiver vs non-receiver tumor cells to nearest immune cell (`figures/immune_proximity_receiver.png`).

## Limitations

1. **No BAM files** in SPAC portal downloads — full MERCI-mtSNP not run; DNA rank is an expression-signature proxy.
2. **Subsampled MERCI** (2000 tumor cells) for LOO-SVR runtime.
3. **CRISPR perturbation guides** present in tissue; DE between receiver groups is confounded by guide heterogeneity (many top DE hits are guide features).

## Reproduce

```bash
python3 download_spac_data.py --name subQ-1 --out-dir subQ-1
python3 run_mc38_merci_analysis.py --data-dir subQ-1 --max-receivers 2000
```

## Outputs

- `subQ-1/processed/mc38_subq1_cells_annotated.h5ad`
- `subQ-1/results/merci_receiver_predictions.csv`
- `subQ-1/results/biological_summary.json`
- `subQ-1/figures/spatial_*.png`
