# MC38 Visium HD MERCI Analysis (SPAC-seq subQ-1)

Pipeline for the SPAC-seq **subQ-1** MC38 subcutaneous Visium HD dataset from [spac.pku-genomics.org](https://spac.pku-genomics.org/#/download).

## Data download

Datasets are listed via the SPAC API (`POST /spac/download/spatial/pageInfo`, `type=2`). For subQ-1:

| File | Description |
|------|-------------|
| `filtered_gene_bc_matrix.h5` | 2 µm bin-level expression |
| `filtered_guide_bc_matrix.h5` | CRISPR guide counts per bin |
| `segmentation.zip` | Cell/nucleus segmentations + `filtered_feature_cell_matrix.h5` |
| `raw_output.zip` | Space Ranger filtered bin matrix + tissue image/positions |

Download share IDs are resolved from the SPAC download page (Aliyun file shares).

## Segmentation → single cells

`segmentation.zip` already contains Space Ranger 4.x **cell-level** outputs:

- `filtered_feature_cell_matrix.h5` — segmented cell expression (~298k cells)
- `graphclust_annotated_cell_segmentations.geojson` — cell polygons + graph-based clusters

The pipeline maps numeric `cell_id` to barcodes (`cellid_000000001-1`) and attaches spatial centroids.

## MERCI

[MERCI](https://github.com/shyhihihi/MERCI) (Zhang et al., *Cancer Cell* 2023) normally requires:

1. **MERCI-mtSNP** on aligned BAM files (mtSNV calling)
2. **MERCI R package** for DNA + RNA rank integration

**Limitation:** SPAC raw downloads do not include `possorted_genome_bam.bam`, so full MERCI-mtSNP cannot be run from the portal files alone. This pipeline:

- Implements `MERCI_LOO_MT_est` and `MERCI_ReceiverPre` in Python (`merci_port.py`)
- Uses a **donor mitochondrial expression signature** as a DNA-rank proxy when BAMs are unavailable
- Subsamples receiver cells (default 2000) for LOO-SVR runtime

For publication-grade MERCI, obtain BAMs from the authors or re-run Space Ranger and use the official R package.

## Run

```bash
export PYTHONUSERBASE=/path/to/pyuser  # optional local package dir
export PYTHONPATH="$(pwd):$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONPATH"
export MPLBACKEND=Agg

python3 run_mc38_merci_analysis.py --data-dir subQ-1 --max-receivers 2000
```

## Outputs

| Path | Content |
|------|---------|
| `subQ-1/processed/mc38_subq1_cells_annotated.h5ad` | Cell-level AnnData with types + spatial |
| `subQ-1/results/merci_receiver_predictions.csv` | Receiver / non-Receiver calls |
| `subQ-1/results/biological_summary.json` | Summary statistics |
| `subQ-1/figures/spatial_*.png` | Spatial maps |
| `subQ-1/figures/nhood_enrichment_mt_receiver.png` | Microniche enrichment |

## Cell type strategy

Marker-based scoring (tumor / immune / myeloid / fibroblast) defines:

- **Donors:** immune + myeloid cells
- **Receiver candidates:** tumor cells

MERCI integrates DNA-proxy and RNA ranks at the top 50% cutoff (default).
