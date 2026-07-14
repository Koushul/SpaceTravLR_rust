# Genius vs Silly — lasso coefficients from real vs inferred received ligands

```text
Genius X  = spatial Gaussian received ligands
Silly  X  = structure-inferred received ligands (S_hat @ mu_query)
Y         = target gene expression (same)
Fit       = pooled-cell Lasso, column std scaling (no mean subtraction)
Alpha     = LassoCV on Genius, reused for Silly and abundance
```

Fits are pooled across cell types because type-pooled `S_hat` is constant
within a type (within-type Silly Lasso is degenerate).

## Summary (median over target genes with non-trivial Genius fits)

### Lasso (sparse; non-trivial Genius fits only)

| dataset | tech | organ | n_ok | coef r | cosine | Jaccard | abund r | lift | feat r |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `mouse_brain_slideseqv2` | SlideSeqV2 | brain | 17/20 | 0.130 | 0.122 | 0.154 | -0.071 | +0.201 | 0.247 |
| `mouse_brain_slideseqv2_regions` | SlideSeqV2 | brain | 4/20 | 0.094 | 0.095 | 0.313 | -0.143 | +0.237 | 0.117 |
| `mouse_hippocampus_slideseqv2` | SlideSeqV2 | hippocampus | 17/20 | 0.113 | 0.115 | 0.552 | -0.087 | +0.200 | 0.173 |
| `mouse_ln_slideseqv2` | SlideSeqV2 | lymph_node | 3/20 | -0.025 | 0.000 | 0.000 | -0.004 | -0.020 | 0.308 |
| `human_tonsil_slidetags` | SlideTags | tonsil | 18/20 | 0.113 | 0.110 | 0.397 | -0.096 | +0.209 | 0.346 |
| `human_tonsil_slidetags_fine` | SlideTags | tonsil | 17/20 | 0.138 | 0.139 | 0.500 | -0.284 | +0.423 | 0.432 |
| `human_melanoma_slidetags` | SlideTags | tumor_melanoma | 6/20 | 0.025 | 0.027 | 0.278 | -0.004 | +0.029 | 0.140 |
| `mouse_kidney_visiumhd` | VisiumHD | kidney | 20/20 | 0.000 | -0.002 | 0.158 | 0.021 | -0.021 | 0.428 |

### Ridge (stable linear coefficients; all targets)

| dataset | tech | organ | ridge coef r | ridge cosine | abund ridge r | lift | ridge R2 genius | ridge R2 silly |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `mouse_brain_slideseqv2` | SlideSeqV2 | brain | 0.067 | 0.051 | nan | +nan | 0.079 | 0.166 |
| `mouse_brain_slideseqv2_regions` | SlideSeqV2 | brain | 0.136 | 0.136 | nan | +nan | 0.011 | 0.011 |
| `mouse_hippocampus_slideseqv2` | SlideSeqV2 | hippocampus | 0.069 | 0.069 | nan | +nan | 0.040 | 0.037 |
| `mouse_ln_slideseqv2` | SlideSeqV2 | lymph_node | 0.052 | 0.060 | nan | +nan | 0.005 | 0.001 |
| `human_tonsil_slidetags` | SlideTags | tonsil | -0.025 | -0.028 | nan | +nan | 0.067 | 0.216 |
| `human_tonsil_slidetags_fine` | SlideTags | tonsil | -0.152 | -0.153 | nan | +nan | 0.042 | 0.147 |
| `human_melanoma_slidetags` | SlideTags | tumor_melanoma | 0.295 | 0.296 | nan | +nan | 0.012 | 0.306 |
| `mouse_kidney_visiumhd` | VisiumHD | kidney | 0.124 | 0.132 | nan | +nan | 0.115 | 0.087 |

Lasso betas are brittle when Silly X is piecewise-constant by type (low rank).
Ridge answers the same question with a stable linear map.
Abundance is the composition-only negative control.

Files: `genius_vs_silly_summary.csv`, `genius_vs_silly_detail.csv`.

