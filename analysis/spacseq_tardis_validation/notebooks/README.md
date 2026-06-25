# SPAC-seq TARDIS validation notebooks

Jupyter notebooks that drive the existing Python scripts under `../scripts/` and display
result tables and figures. Each notebook uses `nb_common.run_script()` so analysis logic
stays in the `.py` files.

## Setup

From `analysis/spacseq_tardis_validation/`:

```bash
pip install jupyter nbconvert ipython  # if needed
jupyter lab notebooks/
```

Notebooks expect pooled subQ data, tuned predictions (`results/predictions_tuned/`), and
(optionally) CNN runs under `runs/baseline_pooled_cnn/`.

## Notebooks

| Notebook | Scripts | Outputs |
| --- | --- | --- |
| `01_core_multislice_validation.ipynb` | `08`, `10` | `results/multislice/`, `figures/scorecard/` |
| `02_spatial_graphclust_validation.ipynb` | `09` | `results/spatial/`, `figures/spatial/` |
| `03_beta_leiden_microniches.ipynb` | `11`, `12` | `results/beta_leiden/`, `figures/beta_leiden/` |
| `04_niche_deg_ccc_spp1.ipynb` | `13`, `18` | `results/niche_deg/`, `figures/niche_spp1/` |
| `05_paper_findings.ipynb` | `19`, `20` | `results/paper_findings/`, `figures/extended_paper/` |
| `06_cnn_guide_enrichment.ipynb` | `23` | `results/cnn_enrichment/`, `figures/cnn_enrichment/` |
| `07_validation_dashboard.ipynb` | `21` | `results/validation_dashboard/` |

Regenerate notebooks after editing templates:

```bash
python3 scripts/create_notebooks.py
```

Headless smoke test (from repo validation root):

```bash
python3 scripts/execute_notebooks.py --quick      # ~5 min
python3 scripts/execute_notebooks.py            # all notebooks (~20–40 min)
```
