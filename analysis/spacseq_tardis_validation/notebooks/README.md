# SPAC-seq TARDIS validation notebooks

Interactive Jupyter notebooks that **load cached CSV/JSON** from `cache/{tag}/manifest.json`
and render **editable matplotlib figures** via `nb_viz.py`. Heavy analysis runs once through
the cache script—not on every notebook open.

## Workflow

```bash
cd analysis/spacseq_tardis_validation

# 1. Index existing results (fast) or re-run analyses (slow)
python3 scripts/cache_validation_results.py --manifest-only
python3 scripts/cache_validation_results.py              # full re-run

# 2. Regenerate notebooks after editing create_notebooks.py
python3 scripts/create_notebooks.py

# 3. Open in Jupyter
jupyter lab notebooks/
```

Use `notebooks/00_refresh_cache.ipynb` to refresh the cache from the UI.

## Cache layout

| Path | Purpose |
| --- | --- |
| `cache/tuned/manifest.json` | Index of all artifact paths + config |
| `results/*/` | Tables and summary JSON (written by `.py` scripts) |
| `scripts/nb_cache.py` | `load_cache()` → `ValidationBundle` |
| `scripts/nb_viz.py` | Plot functions returning `Figure` objects |

## Notebooks

| Notebook | Cached sections | Plots (editable kwargs) |
| --- | --- | --- |
| `00_refresh_cache.ipynb` | all | — (runs cache script) |
| `01_core_multislice_validation.ipynb` | multislice, scorecard | meta bar, slice heatmap, cell-type boxplot, scorecard |
| `02_spatial_graphclust_validation.ipynb` | spatial | niche concordance bars |
| `03_beta_leiden_microniches.ipynb` | beta_leiden | β-Leiden niche concordance |
| `04_niche_deg_ccc_spp1.ipynb` | niche_deg, niche_spp1 | direct DEG + spatial kNN bars |
| `05_paper_findings.ipynb` | paper, extended_paper | scorecard, heatmap, lung module bars |
| `06_cnn_guide_enrichment.ipynb` | cnn | scatter grid, correlation heatmap |
| `07_validation_dashboard.ipynb` | dashboard, niche_spp1 | multi-panel dashboard |

## Smoke test

```bash
python3 scripts/execute_notebooks.py --quick   # load cache + plot, no re-analysis
python3 scripts/execute_notebooks.py           # all notebooks
```

## Tweaking plots

Each plot cell exposes parameters you can change and re-run, e.g.:

```python
TOP_N = 12
SUPPORT_THRESHOLD = 0.6
fig, ax = nb_viz.plot_meta_analysis(meta, top_n=TOP_N)
plt.show()
```

To add a new plot, implement a function in `scripts/nb_viz.py` and wire it in
`scripts/create_notebooks.py`, then regenerate notebooks.
