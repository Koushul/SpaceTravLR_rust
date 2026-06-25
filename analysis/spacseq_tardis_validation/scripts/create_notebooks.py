#!/usr/bin/env python3
"""Generate validation Jupyter notebooks under ../notebooks/."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"


def cell(typ: str, source: str) -> dict:
    return {
        "cell_type": typ,
        "metadata": {},
        "source": source if isinstance(source, list) else source.splitlines(keepends=True),
    }


def nb(cells: list[dict], title: str) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (SpaceTravLR validation)",
                "language": "python",
                "name": "spacetravlr-validation",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
            "title": title,
        },
        "cells": cells,
    }


BOOTSTRAP = """\
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path.cwd()
if (ROOT / "scripts" / "nb_common.py").exists():
    pass
elif (ROOT.parent / "scripts" / "nb_common.py").exists():
    ROOT = ROOT.parent
else:
    raise RuntimeError(
        "Start Jupyter from analysis/spacseq_tardis_validation/ "
        "(or open notebooks from that directory)."
    )

sys.path.insert(0, str(ROOT / "scripts"))
from nb_common import bootstrap, default_config, run_script, load_cache
import nb_viz

bootstrap()
CFG = default_config()
CACHE_TAG = CFG["tag"]
print("ROOT:", ROOT)
print("Config:", CFG)
"""

CACHE_REFRESH = """\
# Optional: refresh cached results (runs heavy .py scripts once; skip in normal notebook use)
REFRESH_CACHE = False
if REFRESH_CACHE:
    proc = run_script(
        "cache_validation_results.py",
        "--manifest-only" if False else "--manifest-only",
        "--tag", CACHE_TAG,
    )
    print(proc.stdout[-2000:] if proc.stdout else proc.stderr)
"""

LOAD_CACHE = """\
bundle = load_cache(CACHE_TAG)
print("Cached sections:", bundle.sections())
if bundle.missing():
    print(f"Warning: {len(bundle.missing())} artifacts missing from cache manifest")
bundle.missing()[:5]
"""

NOTEBOOKS: list[tuple[str, str, list[dict]]] = [
    (
        "01_core_multislice_validation.ipynb",
        "Core multislice validation & scorecard",
        [
            cell("markdown", "# Core multislice validation\n\nLoads cached multislice + scorecard tables and renders editable figures in-notebook.\n\nRun `python3 scripts/cache_validation_results.py --sections multislice,scorecard` once to refresh data."),
            cell("code", BOOTSTRAP),
            cell("code", LOAD_CACHE),
            cell("code", """\
combined = bundle.table("multislice", "combined")
meta = bundle.table("multislice", "meta")
scorecard = bundle.table("scorecard", "table")
overall = bundle.json("multislice", "overall")
overall
"""),
            cell("code", """\
# Tweak plot parameters below
TOP_N = 12
fig, ax = nb_viz.plot_meta_analysis(meta, top_n=TOP_N)
plt.show()
"""),
            cell("code", """\
fig, ax = nb_viz.plot_slice_heatmap(combined, slices=["subQ-1", "subQ-2", "subQ-3", "subQ-4"])
plt.show()
"""),
            cell("code", """\
fig, ax = nb_viz.plot_celltype_boxplot(combined)
plt.show()
"""),
            cell("code", """\
fig, ax = nb_viz.plot_prediction_scorecard(scorecard, levels=["cell_type", "graphclust"])
plt.show()
scorecard.head(10)
"""),
        ],
    ),
    (
        "02_spatial_graphclust_validation.ipynb",
        "Spatial graphclust microniche validation",
        [
            cell("markdown", "# Spatial validation (graphclust niches)\n\nInteractive plots from cached spatial niche concordance tables."),
            cell("code", BOOTSTRAP),
            cell("code", LOAD_CACHE),
            cell("code", """\
overall = bundle.json("spatial", "overall")
niche_corr = bundle.table("spatial", "niche_corr")
summary = bundle.table("spatial", "summary")
overall
"""),
            cell("code", """\
NICHE_TYPE = "graphclust"
CELL_TYPES = ["immune", "myeloid", "fibroblast"]
fig, ax = nb_viz.plot_spatial_niche_corr(niche_corr, niche_type=NICHE_TYPE, cell_types=CELL_TYPES)
plt.show()
"""),
            cell("code", """\
summary.head(12) if not summary.empty else niche_corr.head(12)
"""),
        ],
    ),
    (
        "03_beta_leiden_microniches.ipynb",
        "Beta-Leiden functional microniches",
        [
            cell("markdown", "# β-Leiden functional microniches\n\nExplore cached β-Leiden niche concordance and summary metrics."),
            cell("code", BOOTSTRAP),
            cell("code", LOAD_CACHE),
            cell("code", """\
summary = bundle.json("beta_leiden", "overall")
niche_corr = bundle.table("beta_leiden", "niche_corr")
bl_summary = bundle.table("beta_leiden", "summary")
summary
"""),
            cell("code", """\
fig, ax = nb_viz.plot_spatial_niche_corr(
    niche_corr, niche_type="beta_leiden", cell_types=["immune", "myeloid", "fibroblast", "tumor"]
)
plt.show()
"""),
            cell("code", """\
bl_summary.sort_values("median_pearson_r", ascending=False).head(12)
"""),
        ],
    ),
    (
        "04_niche_deg_ccc_spp1.ipynb",
        "Niche DEG, CCC, and Spp1 recovery",
        [
            cell("markdown", "# Niche DEG + CCC + Spp1\n\nDirect sgP-cell and spatial kNN concordance from cache."),
            cell("code", BOOTSTRAP),
            cell("code", LOAD_CACHE),
            cell("code", """\
niche_overall = bundle.json("niche_deg", "overall")
spp1_overall = bundle.json("niche_spp1", "overall")
direct = bundle.table("niche_spp1", "direct_deg")
spatial = bundle.table("niche_deg", "spatial_neighbor")
ccc = bundle.table("niche_deg", "ccc")
niche_overall, spp1_overall
"""),
            cell("code", """\
fig, ax = nb_viz.plot_direct_deg_bars(direct)
plt.show()
"""),
            cell("code", """\
fig, ax = nb_viz.plot_direct_deg_bars(
    spatial,
    title="Spatial kNN niche DEG concordance",
)
plt.show()
"""),
            cell("code", """\
spatial.sort_values("pearson_r", ascending=False).head(15)
"""),
        ],
    ),
    (
        "05_paper_findings.ipynb",
        "Paper headline biology validation",
        [
            cell("markdown", "# Paper findings (Zhang et al. Cell 2026)\n\nModule-level hypothesis tests — adjust thresholds and labels in the cells below."),
            cell("code", BOOTSTRAP),
            cell("code", LOAD_CACHE),
            cell("code", """\
paper = bundle.json("paper", "overall")
modules = bundle.table("paper", "modules")
lung_icam1 = bundle.table("extended_paper", "lung_icam1")
lung_bcam = bundle.table("extended_paper", "lung_bcam")
paper
"""),
            cell("code", """\
SUPPORT_THRESHOLD = 0.6
fig, axes = nb_viz.plot_paper_scorecard(modules, support_threshold=SUPPORT_THRESHOLD)
plt.show()
"""),
            cell("code", """\
fig, axes = nb_viz.plot_paper_module_heatmap(modules)
plt.show()
"""),
            cell("code", """\
fig, ax = nb_viz.plot_lung_module_bars(
    lung_icam1,
    title="Lung M001 sgIcam1 — paper immune-escape modules (observed)",
    support_threshold=SUPPORT_THRESHOLD,
)
plt.show()
"""),
            cell("code", """\
fig, ax = nb_viz.plot_lung_module_bars(
    lung_bcam,
    title="Lung M001 sgBcam — Cd44/Spp1 axis modules (observed)",
    support_threshold=SUPPORT_THRESHOLD,
)
plt.show()
"""),
        ],
    ),
    (
        "06_cnn_guide_enrichment.ipynb",
        "CNN beta microniches & guide enrichment",
        [
            cell("markdown", """\
# CNN β-microniches → guide enrichment

## Link to Zhang et al. Cell 2026 (SPAC-seq)

The paper shows that CRISPR KOs reshape **where** cells survive in tissue, not just **what** they express:

| Paper theme | Perturbation | Spatial/composition phenotype | Our proxy |
| --- | --- | --- | --- |
| **Immune exclusion** | sgIcam1 (lung M001) | sgIcam1+ tumor accumulates in immune-cold niches; IFN/LFA-1↓, T cells↓, M2/Spp1↑ | Predicted exclusion index + Icam1 CNN β; validate log₂ OR across CNN microniches |
| **Cd44–Spp1 crosstalk** | sgCd44 / sgSpp1 (lung); sgBcam in subQ | Macrophage Spp1 couples to T-cell Cd44; exhaustion/ECM programs | sgBcam in subQ/lung; Spp1/ECM escape module in predicted score |
| **Antigen presentation** | sgIl4ra, sgCd83, sgCd74 (subQ expanded) | MHC-II / costimulation down in immune niches | Immune-infiltration vs exclusion balance in niche score |

**This notebook:** each scatter point is a **tumor microniche** (CNN β-Leiden cluster). We ask whether niches SpaceTravLR scores as guide-favorable match niches where sgP cells are actually over-represented vs NTC.

Refresh data (higher niche resolution + spatial maps): set `REFRESH_CNN=True` below or run:
`python3 scripts/23_cnn_microniche_enrichment.py --leiden-resolution 0.9`
"""),
            cell("code", BOOTSTRAP + "\nfrom nb_common import run_script\n"),
            cell("code", """\
REFRESH_CNN = False
LEIDEN_RESOLUTION = 0.9  # higher → more microniches (default was 0.55)
if REFRESH_CNN:
    proc = run_script(
        "23_cnn_microniche_enrichment.py",
        "--tag", CFG["cnn_tag"],
        "--leiden-resolution", str(LEIDEN_RESOLUTION),
        "--min-ntc", "2", "--min-pert", "2",
    )
    print(proc.stdout[-3000:] if proc.stdout else proc.stderr)
"""),
            cell("code", LOAD_CACHE),
            cell("code", """\
summary = bundle.json("cnn", "overall")
enrich = bundle.table("cnn", "enrichment")
corr = bundle.table("cnn", "corr")
print("Leiden resolution:", summary.get("leiden_resolution", "unknown"))
print("Median n niches:", corr["n_niches"].median())
summary, corr.sort_values("pearson_r", ascending=False).head(8)
"""),
            cell("code", """\
TOP_N = 6
POINT_COLOR = "#2563eb"
fig, axes = nb_viz.plot_cnn_enrichment_scatter(enrich, corr, top_n=TOP_N, tag=CFG["cnn_tag"], point_color=POINT_COLOR)
plt.show()
"""),
            cell("code", """\
fig, ax = nb_viz.plot_cnn_enrichment_heatmap(corr, tag=CFG["cnn_tag"], cmap="RdBu_r")
plt.show()
"""),
            cell("code", """\
# Spatial microniche map on tissue (cached parquet from script 23)
SPATIAL_SLICE = "Lung_Metastasis_M001"
SPATIAL_PERT = "Icam1"
spatial = bundle.spatial_tumor(SPATIAL_SLICE)
print("Available slices:", bundle.spatial_slices())
if spatial.empty:
    print("Run with REFRESH_CNN=True to generate spatial_tumor_*.parquet")
else:
    fig, axes = nb_viz.plot_microniche_spatial(
        spatial,
        slice_id=SPATIAL_SLICE,
        perturb=SPATIAL_PERT,
        panel="triple",
        point_size=3.5,
        title=f"{SPATIAL_SLICE} — paper Icam1 immune-exclusion niches (sg{SPATIAL_PERT})",
    )
    plt.show()
"""),
            cell("code", """\
# All tumor microniches on tissue (single panel)
if not spatial.empty:
    fig, ax = nb_viz.plot_microniche_spatial(
        spatial, slice_id=SPATIAL_SLICE, panel="all", point_size=4,
        title=f"{SPATIAL_SLICE} CNN β-microniches ({spatial['cnn_leiden'].nunique()} niches)",
    )
    plt.show()
"""),
        ],
    ),
    (
        "07_validation_dashboard.ipynb",
        "Consolidated validation dashboard",
        [
            cell("markdown", "# Validation dashboard\n\nAll-in-one metrics panel from cached dashboard + direct DEG tables."),
            cell("code", BOOTSTRAP),
            cell("code", LOAD_CACHE),
            cell("code", """\
metrics = bundle.table("dashboard", "metrics")
direct = bundle.table("niche_spp1", "direct_deg")
dash_overall = bundle.json("dashboard", "overall")
metrics
"""),
            cell("code", """\
SUPPORT_THRESHOLD = 0.6
fig, axes = nb_viz.plot_validation_dashboard(
    metrics, direct, tag=CFG["spatial_tag"], support_threshold=SUPPORT_THRESHOLD,
)
plt.show()
"""),
        ],
    ),
    (
        "00_refresh_cache.ipynb",
        "Refresh validation cache",
        [
            cell("markdown", "# Refresh validation cache\n\nRun this notebook (or the CLI) once to populate `cache/{tag}/manifest.json` and all result CSVs/JSONs.\n\n```bash\npython3 scripts/cache_validation_results.py --manifest-only  # index existing\npython3 scripts/cache_validation_results.py                  # full re-run\n```"),
            cell("code", BOOTSTRAP + "\nfrom nb_common import run_script\n"),
            cell("code", """\
MANIFEST_ONLY = True  # set False to re-run all analysis scripts
sections = "multislice,spatial,scorecard,beta_leiden,niche_deg,niche_spp1,paper,extended_paper,dashboard,cnn"
args = ["cache_validation_results.py", "--tag", CACHE_TAG, "--sections", sections]
if MANIFEST_ONLY:
    args.append("--manifest-only")
proc = run_script(*args)
print(proc.stdout[-4000:] if proc.stdout else proc.stderr)
"""),
            cell("code", LOAD_CACHE),
        ],
    ),
]


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for fname, title, cells in NOTEBOOKS:
        path = NB_DIR / fname
        path.write_text(json.dumps(nb(cells, title), indent=1) + "\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
