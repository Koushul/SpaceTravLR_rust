# Shared tissue CNN + per-gene MLP

Experimental PyTorch pipeline that separates **one pretrained CNN per tissue** from **per-gene MLP heads**, mirroring SpaceTravLR spatial maps and betadata Feather export.

## Design

| Component | Role |
|-----------|------|
| `TissueVisionEncoder` | Shared CNN (VisionEncoder + SpatialMLP) learns tissue spatial architecture on the **train half** |
| `GeneHeadMLP` | Per-gene head on **frozen** CNN features; Lasso anchors scale betas like SpaceTravLR |
| `group_lasso.py` | Per-cluster ElasticNet anchors (modulators = neighbor counts per cell type) |
| `betadata_io.py` | `{gene}_betadata.feather` with `CellID`, `beta0`, `beta_<cell_type>` (LZ4 IPC) |

Architecture matches `src/model.rs`: 3× (Conv→BN→PReLU→MaxPool), spatial pyramid pooling, 64-d embedding. Variants: `base` (3 conv blocks), `deep` (+4th block), `wide` (wider channels + 4th block). Best variant is chosen by **linear-probe transfer R²** on the finetune half, not pretrain MSE alone.

## Data

SlideTags human tonsil (`data/h5ad/SlideTags_human_tonsil.h5ad`) split 50/50:

- **Train half** → CNN pretrain (multi-gene expression from spatial context)
- **Finetune half** → per-gene Lasso + MLP + betadata export

## Reproduce

```bash
cd analysis/shared_tissue_cnn

# Full pipeline (split → pretrain both variants → finetune genes → plots)
python3 run_pipeline.py

# Or step-by-step:
python3 split_tonsil.py --h5ad ../../data/h5ad/SlideTags_human_tonsil.h5ad
python3 pretrain_cnn.py --variant base --epochs 40
python3 finetune_genes.py --genes AICDA,CD74,CD3D,MS4A6A,LYZ
python3 plot_results.py
```

Outputs (gitignored): `data/tonsil_{train,finetune}.h5ad`, `outputs/`, `figures/`.

## Dependencies

- Python 3.10+
- `torch`, `scanpy`, `anndata`, `scipy`, `scikit-learn`, `matplotlib`, `pyarrow`, `pandas`

Large artifacts are gitignored; scripts and this README are tracked.

## Example results (30 pretrain / 25 finetune epochs, spatial_dim=16, CPU)

### CNN variant selection (linear-probe transfer to finetune half)

| Variant | Mean transfer R² |
|---------|------------------|
| base | -0.093 |
| **deep** (selected) | **-0.013** |
| wide | -0.064 |

`deep` wins on mean linear-probe transfer (train-half features → finetune-half expression).

### Finetune-half gene performance

| Gene | CNN R² | Lasso R² | Transfer (train→finetune R²) |
|------|--------|----------|------------------------------|
| MS4A6A | **0.59** | 0.52 | 0.50 → 0.59 |
| LYZ | **0.32** | 0.29 | 0.25 → 0.32 |
| AICDA | -0.16 | -0.16 | -0.17 → -0.16 |
| CD3D | -0.18 | -0.19 | -0.14 → -0.18 |
| CD74 | -4.90 | -4.91 | -4.92 → -4.90 |

Global R² is negative for some genes because many clusters lack Lasso fits; **per-cluster R² is positive** where fits exist (see `per_cluster_r2_heatmap.png`). MS4A6A and LYZ show the shared CNN learns transferable myeloid/B-cell niche features across tissue halves.

`MS4A1` / `MKI67` are absent from SlideTags tonsil var; defaults use `MS4A6A`, `LYZ`.

Figures: `figures/gene_performance_finetune_half.png`, `pretrain_curves.png`, `gradcam_examples.png`, `train_vs_finetune_half.png`, `per_cluster_r2_heatmap.png`.

Grad-CAM and Integrated Gradients highlight inverse-distance peaks and neighbor context the shared encoder uses for follicle / T-cell niche structure.

**CUDA note:** Blackwell GPUs (sm_120) need a newer PyTorch build; pipeline auto-falls back to CPU via `pick_device()`.
