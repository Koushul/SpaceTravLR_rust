---
license: mit
tags:
  - spatial-transcriptomics
  - gene-regulatory-network
  - xenium
  - spacetravlr
  - mouse
  - skin
pretty_name: SpaceTravLR outputs
size_categories:
  - 100B<n<1T
---

# SpaceTravLR dataset hub

Precomputed [SpaceTravLR](https://github.com/Koushul/SpaceTravLR_rust) outputs: per-gene beta matrices (`*_betadata.feather`), run metadata, and optional per-sample `.h5ad` exports.

## Layout

```
spacetravlr/
├── tonsil/                         # placeholder / demo gene outputs
└── xenium_skin_mixed/
    ├── run.toml                    # shared training config for this cohort
    ├── manifest.json               # sample index and upload metadata
    ├── sample12/
    ├── sample13/
    ├── sample14/
    ├── sample15/
    └── sample16/
```

Each `xenium_skin_mixed/sample*/` directory contains:

| File / folder | Description |
|---|---|
| `*_betadata.feather` | Per-modulator beta matrices from SpaceTravLR training |
| `sample*.h5ad` | Condition-filtered AnnData export for the sample |
| `sample*.csv` | Summary table for the run |
| `condition_label.txt` | Condition identifier |
| `spacetravlr_run_repro.toml` | Reproducibility config (paths rewritten on download) |
| `spacetravlr_run_summary.html` | HTML training summary |
| `log/` | Training logs |

**Not uploaded:** `perturbations/` folders (in-silico perturbation grids).

## Cohort: `xenium_skin_mixed`

Mouse skin Xenium data with mixed samples. Training used `xenium_skin_mixed.h5ad` with `sample_id` as the condition column and `cell_type_int` as the cluster annotation.

| Sample | Approx. size | Beta files |
|--------|-------------:|-----------:|
| sample12 | 59 GB | 2,809 |
| sample13 | 103 GB | 2,807 |
| sample14 | 66 GB | 2,810 |
| sample15 | 100 GB | 2,801 |
| sample16 | 89 GB | 2,801 |

## Download

```bash
# Entire xenium_skin_mixed cohort
hf download Koushul/spacetravlr xenium_skin_mixed --repo-type dataset --local-dir ./spacetravlr

# Single sample
hf download Koushul/spacetravlr xenium_skin_mixed/sample12 --repo-type dataset --local-dir ./sample12
```

## Usage with SpaceTravLR

Point `spatial_viewer` or `spacetravlr-perturb` at a downloaded sample directory containing `*_betadata.feather` and the matching `spacetravlr_run_repro.toml`.

## Citation

If you use these outputs, please cite SpaceTravLR and the underlying Xenium skin dataset.
