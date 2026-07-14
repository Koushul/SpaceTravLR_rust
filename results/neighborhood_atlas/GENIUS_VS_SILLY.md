# Genius vs Silly — lasso/ridge coefficients from real vs inferred received ligands

```text
Genius X  = spatial Gaussian received ligands (ground truth)
Silly  X  = structure-inferred received ligands (S_hat @ mu)
Y         = target gene expression (same)
```

## Headline result (type-level — the snRNA-seq estimand)

Type-pooled `S_hat` makes Silly **constant within each cell type**. The fair
coefficient comparison is therefore at **type level**:

```text
X_type[c, l] = mean_{i: type(i)=c} received[i, l]     # Genius
X_type[c, l] = sum_t S_hat[c, t] * mu[t, l]            # Silly
Y_type[c]    = mean_{i: type(i)=c} expr[i, target]
```

Fit Ridge on the `n_types` rows. Median coefficient Pearson (Genius vs Silly):

| dataset | tech | organ | type feat r | **ridge coef r** | ridge cosine |
|---|---|---|---:|---:|---:|
| `mouse_brain_slideseqv2` | SlideSeqV2 | brain | 0.987 | **0.991** | 0.948 |
| `human_tonsil_slidetags` | SlideTags | tonsil | 0.978 | **0.953** | 0.976 |
| `human_melanoma_slidetags` | SlideTags | melanoma | 0.983 | **0.906** | 0.968 |
| `human_tonsil_slidetags_fine` | SlideTags | tonsil | 0.986 | **0.866** | 0.987 |
| `mouse_brain_slideseqv2_regions` | SlideSeqV2 | brain | 0.692 | **0.855** | 0.810 |
| `mouse_hippocampus_slideseqv2` | SlideSeqV2 | hippocampus | 0.775 | **0.854** | 0.843 |
| `mouse_kidney_visiumhd` | VisiumHD | kidney | 0.922 | **0.677** | 0.865 |
| `mouse_ln_slideseqv2` | SlideSeqV2 | lymph_node | 0.956 | **0.660** | 0.810 |

**Inferred (Silly) coefficients are close to Genius at type level** — typically
coef Pearson ~0.66–0.99, cosine ~0.81–0.99 — matching how close type-mean
received ligands themselves are (`type feat r`).

Files: `genius_vs_silly_type_level_summary.csv`, `genius_vs_silly_type_level_detail.csv`.

## Cell-pooled Lasso (secondary — expected to look worse)

Pooling all cells and fitting Lasso/Ridge, Silly X has only `n_types` unique
rows. Genius still has within-type niche variation, so the design matrices
differ geometrically and **sparse coefficients do not match closely**
(median Lasso coef r ~0.0–0.14), even though Silly still beats abundance on
most lymphoid/brain datasets.

| dataset | Lasso coef r (Silly) | abund Lasso r | feat r (cell) |
|---|---:|---:|---:|
| tonsil fine | 0.138 | -0.284 | 0.432 |
| tonsil | 0.113 | -0.096 | 0.346 |
| hippocampus | 0.113 | -0.087 | 0.173 |
| brain | 0.130 | -0.071 | 0.247 |
| kidney | 0.000 | 0.021 | 0.428 |
| melanoma | 0.025 | -0.004 | 0.140 |
| LN | -0.025 | -0.004 | 0.308 |

This is why snRNA-seq structure mode should be interpreted at **cell-type**
resolution (or use expression-matched cell-specific niches), not as a
drop-in replacement for within-type spatial ligand variation.

## How to reproduce

```bash
python scripts/eval_genius_vs_silly_lasso.py --max-cells 8000 --n-ligands 30 \
  --n-targets 20 --use-atlas-radius
# type-level table: results/neighborhood_atlas/genius_vs_silly_type_level_*.csv
```
