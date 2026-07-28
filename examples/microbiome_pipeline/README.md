# Microbiome BR pipeline example

Host–microbe bacterial→receptor (BR) training and post-hoc analysis for SpaceTravLR.

## Contents

- `configs/` — example TOML + bacterial–host interaction CSVs (1× and 2× radii)
- `scripts/` — data prep, Slurm train/eval wrappers, and post-hoc analysis:
  - `01`–`06` — QC, lasso pilot, sender table, imputation
  - `07_br_beta_microniches.py` — cluster host cells on BR |β|; compare to bacterial niches
  - `08_stacked_bars_genus_ligands.py` — BR ranking stacked by genus ligand expression
  - `09_functional_microniches.py` — secretion-driven functional microniches (modules + BR + NMF)
  - `slurm_*.sh` — cluster job templates used for full / 2×-radius runs

## Model changes (this PR)

Rust microbial BR support lives in `src/microbial.rs` and related config / spatial estimator hooks. See `docs/microbial.md`.

## Paths

Analysis scripts read:

```bash
export SPACETRAVLR_MICROBIOME_ROOT=/path/to/spacetravlr_microbiome
```

Default (if unset) assumes a sibling checkout:

`../spacetravlr_microbiome` next to this `SpaceTravLR_rust` repo.

Example TOML and Slurm scripts still contain absolute path placeholders from the Stereo-seq tumor run — edit `adata_path`, `sender_table`, `output_dir`, and binary paths before submitting jobs.

## Minimal analysis flow (after a BR train/collect)

```bash
export SPACETRAVLR_MICROBIOME_ROOT=/path/to/data_workdir
python scripts/08_stacked_bars_genus_ligands.py
python scripts/07_br_beta_microniches.py
python scripts/09_functional_microniches.py
```

Expected layout under `$SPACETRAVLR_MICROBIOME_ROOT`:

- `processed/` — host h5ad + bacterial sender parquet
- `runs/tumor_br_r2x/` — `*_betadata.feather`, `top_br_terms.csv`
- `site_br_report/assets/r2x/` — figure output (created/updated by scripts)
