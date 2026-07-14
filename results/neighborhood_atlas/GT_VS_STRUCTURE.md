# Ground-truth vs structure-inferred received ligands

For each atlas dataset, spatial Gaussian received ligands (ground truth)
are compared to predictions from the type-pooled neighborhood grammar `Ŝ`.

```
truth[i,l]     = (1/N) Σ_j exp(-d²/2r²) · expr[j,l]
structure[i,l] = Σ_t Ŝ[type(i), t] · μ[t, l]
```

Primary metric: **type-level Pearson** (receiver-type means of pred vs truth).
Cell-level Pearson is expected to be lower because `Ŝ` cannot recover
within-type niche heterogeneity without coordinates.

## Summary (structure_pooled vs GT)

| dataset | tech | organ | type r | type MAE | cell r | cell MAE | soft cos | vs abund Δr |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `mouse_brain_slideseqv2` | SlideSeqV2 | brain | 0.859 | 0.005302 | 0.251 | 0.03152 | 0.898 | +0.700 |
| `mouse_brain_slideseqv2_regions` | SlideSeqV2 | brain | 0.629 | 0.004132 | 0.111 | 0.01381 | 0.994 | +0.380 |
| `mouse_hippocampus_slideseqv2` | SlideSeqV2 | hippocampus | 0.787 | 0.001019 | 0.164 | 0.006824 | 0.946 | +0.189 |
| `mouse_ln_slideseqv2` | SlideSeqV2 | lymph_node | 0.925 | 0.0008539 | 0.314 | 0.004802 | 0.966 | +0.286 |
| `human_tonsil_slidetags` | SlideTags | tonsil | 0.971 | 0.0284 | 0.336 | 0.3564 | 0.924 | +0.345 |
| `human_tonsil_slidetags_fine` | SlideTags | tonsil | 0.988 | 0.03392 | 0.430 | 0.3111 | 0.922 | +0.167 |
| `human_melanoma_slidetags` | SlideTags | tumor_melanoma | 0.969 | 0.01613 | 0.138 | 0.3712 | 0.985 | +0.370 |
| `mouse_kidney_visiumhd` | VisiumHD | kidney | 0.950 | 0.00313 | 0.432 | 0.01496 | 0.952 | +1.048 |

Files: `gt_vs_structure_summary.csv`, `gt_vs_structure_detail.csv`.

