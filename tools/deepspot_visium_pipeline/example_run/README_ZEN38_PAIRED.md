# ZEN38 paired Visium evaluation (COAD)

We use sample **ZEN38** from the DeepSpot repository’s `example_data`: public colon cancer Visium with **paired** measured expression (`ZEN38.h5ad`) and the **tissue hires H&E** embedded in `adata.uns['spatial']['library_id']['images']['hires']`.  
This matches the **Colon_HEST1K** DeepSpot checkpoint (same disease context).

## Caveats

- Here we run **`--foundation-timm-imagenet`** (ImageNet ViT-L, not licensed UNI weights).  
  Expect **weak global** correlation vs measured expression; some epithelial markers can still align (see CSV).
- After you approve **MahmoodLab/UNI** on Hugging Face, rerun with `--foundation-weights path/to/pytorch_model.bin` for meaningful benchmarking.

## Outputs (example_run/)

| File | Description |
|------|-------------|
| `zen38_paired_measured_vs_deepspot.h5ad` | `layers['measured_log1p']`, `layers['imputed_count']`, `obsm['spatial']`, correlation summaries in `uns` |
| `zen38_pearson_all_genes.csv` | Per-gene Pearson *r* (387 spots) |
| `zen38_pearson_markers.csv` | Subset of CRC / immune markers |
| `zen38_paired_measured_vs_deepspot.counts.npz` | Cached prediction matrix (skip ViT re-run via `finalize_from_counts_npz.py`) |

## Commands

```bash
cd tools/deepspot_visium_pipeline
source .venv/bin/activate
export PYTHONPATH=/path/to/DeepSpot

# Copy ZEN38.h5ad from DeepSpot example_data into zen38_source/ (or rely on /tmp/DeepSpot path)

python eval_paired_zen38.py \
  --weights-dir example_run/DeepSpot_pretrained_model_weights/Colon_HEST1K \
  --foundation-timm-imagenet \
  --out-h5ad example_run/zen38_paired_measured_vs_deepspot.h5ad \
  --out-corr-csv example_run/zen38_pearson_all_genes.csv \
  --out-markers-csv example_run/zen38_pearson_markers.csv
```

Recompute `.h5ad` + CSV from cache only:

```bash
python finalize_from_counts_npz.py \
  --counts-npz example_run/zen38_paired_measured_vs_deepspot.counts.npz \
  --measured-h5ad zen38_source/ZEN38.h5ad \
  --weights-dir example_run/DeepSpot_pretrained_model_weights/Colon_HEST1K \
  --out-h5ad example_run/zen38_paired_from_npz.h5ad \
  --out-corr-csv example_run/zen38_pearson_all_genes.csv
```

Dataset citation: see DeepSpot `example_data/data/meta/ZEN38.json` (`study_link`, Zenodo ID).
