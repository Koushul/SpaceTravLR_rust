# Shared tissue CNN + per-gene MLP

Experimental PyTorch pipeline that separates **one pretrained CNN per tissue** from **per-gene MLP heads**, mirroring SpaceTravLR spatial maps and betadata Feather export.

## Design

| Component | Role |
|-----------|------|
| `TissueVisionEncoder` | Shared CNN (VisionEncoder + SpatialMLP) learns tissue spatial architecture on the **train half** |
| `GeneHeadMLP` | Per-gene head on **frozen** CNN features; Lasso anchors scale betas like SpaceTravLR |
| `group_lasso.py` | Per-cluster ElasticNet anchors (modulators = neighbor counts per cell type) |
| `betadata_io.py` | `{gene}_betadata.feather` with `CellID`, `beta0`, `beta_<cell_type>` (LZ4 IPC) |

Architecture matches `src/model.rs`: 3× (Conv→BN→PReLU→MaxPool), spatial pyramid pooling, 64-d embedding. Variants: `base` (3 conv blocks) and `deep` (+4th block).

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

## Example results (25 pretrain / 20 finetune epochs, spatial_dim=16, CPU)

| Metric | base CNN | deep CNN (selected) |
|--------|----------|---------------------|
| Pretrain MSE (64 genes, train half) | 0.423 | **0.415** |

Finetune-half in-sample R² (global; per-cluster R² is positive where Lasso passed):

| Gene | CNN R² | Lasso R² | Notes |
|------|--------|----------|-------|
| AICDA | -0.16 | -0.16 | per-cluster R² 0.15–0.42 |
| CD74 | -4.90 | -4.91 | per-cluster R² 0.16–0.69 |
| CD3D | -0.18 | -0.19 | per-cluster R² 0.13–0.70 |

`MS4A1` / `MKI67` are absent from SlideTags tonsil var; defaults use `MS4A6A`, `LYZ`.

Figures: `figures/gene_performance_finetune_half.png`, `pretrain_curves.png`, `gradcam_examples.png`, `train_vs_finetune_half.png`.

**CUDA note:** Blackwell GPUs (sm_120) need a newer PyTorch build; pipeline auto-falls back to CPU via `pick_device()`.
