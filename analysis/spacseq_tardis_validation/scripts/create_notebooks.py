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
from nb_common import bootstrap, default_config, run_script, load_json, load_csv, show_figures, artifact_status

bootstrap()
CFG = default_config()
print("ROOT:", ROOT)
print("Config:", CFG)
"""

NOTEBOOKS: list[tuple[str, str, list[dict]]] = [
    (
        "01_core_multislice_validation.ipynb",
        "Core multislice validation & scorecard",
        [
            cell("markdown", "# Core multislice validation\n\nRuns pooled 4-slice cell-type validation and prediction scorecard using existing `.py` scripts."),
            cell("code", BOOTSTRAP),
            cell("code", """\
proc = run_script(
    "08_multislice_validation.py",
    "--baseline-h5ad", CFG["baseline_h5ad"],
    "--pred-dir", CFG["pred_dir"],
    "--tag", CFG["tag"],
)
print(proc.stdout[-4000:] if len(proc.stdout) > 4000 else proc.stdout)
"""),
            cell("code", """\
proc = run_script("10_sharpened_scorecard.py", "--models", "pooled", "seed")
print(proc.stdout)
scorecard = load_csv("results/scorecard/prediction_scorecard.csv")
scorecard.head(10)
"""),
            cell("code", """\
show_figures("figures/multislice/fig2_meta_analysis_multislice.png")
show_figures("figures/scorecard/fig_scorecard.png")
"""),
        ],
    ),
    (
        "02_spatial_graphclust_validation.ipynb",
        "Spatial graphclust microniche validation",
        [
            cell("markdown", "# Spatial validation (graphclust niches)\n\nCompares predicted vs observed KO effects within Space Ranger graphclust clusters."),
            cell("code", BOOTSTRAP),
            cell("code", """\
proc = run_script(
    "09_spatial_validation.py",
    "--baseline-h5ad", CFG["baseline_h5ad"],
    "--pred-dir", CFG["pred_dir"],
    "--tag", CFG["tag"],
)
print(proc.stdout[-3000:] if proc.stdout else proc.stderr)
"""),
            cell("code", """\
overall = load_json(f"results/spatial/overall_{CFG['tag']}.json")
overall
"""),
            cell("code", 'show_figures(f"figures/spatial/spatial_map_*_{CFG[\'tag\']}.png", max_images=6)'),
        ],
    ),
    (
        "03_beta_leiden_microniches.ipynb",
        "Beta-Leiden functional microniches",
        [
            cell("markdown", "# β-Leiden functional microniches\n\nCNN/seed beta scores + spatial Leiden vs SPAC-seq concordance."),
            cell("code", BOOTSTRAP),
            cell("code", """\
proc = run_script(
    "11_beta_leiden_microniches.py",
    "--baseline-h5ad", CFG["baseline_h5ad"],
    "--betadata-dir", CFG["betadata_dir"],
    "--pred-dir", CFG["pred_dir"],
    "--tag", CFG["tag"],
)
print(proc.stdout[-3000:] if proc.stdout else proc.stderr)
"""),
            cell("code", """\
proc = run_script(
    "12_beta_leiden_report_figures.py",
    "--baseline-h5ad", CFG["baseline_h5ad"],
    "--betadata-dir", CFG["betadata_dir"],
    "--pred-dir", CFG["pred_dir"],
    "--tag", CFG["tag"],
)
print(proc.stdout)
"""),
            cell("code", """\
summary = load_json(f"results/beta_leiden/overall_{CFG['tag']}.json")
summary
"""),
            cell("code", 'show_figures(f"figures/beta_leiden/fig1_main_overview_{CFG[\'tag\']}.png")'),
            cell("code", 'show_figures(f"figures/beta_leiden/fig2_spotlight_Il4ra_immune_{CFG[\'tag\']}.png")'),
        ],
    ),
    (
        "04_niche_deg_ccc_spp1.ipynb",
        "Niche DEG, CCC, and Spp1 recovery",
        [
            cell("markdown", "# Niche DEG + CCC + Spp1\n\nSpatial kNN niche DEGs and direct sgP-cell concordance. β-Leiden section skipped by default (slow on CPU)."),
            cell("code", BOOTSTRAP),
            cell("code", """\
proc = run_script(
    "13_niche_deg_ccc_analysis.py",
    "--baseline-h5ad", CFG["baseline_h5ad"],
    "--betadata-dir", CFG["betadata_dir"],
    "--pred-dir", CFG["pred_dir"],
    "--tag", CFG["spatial_tag"],
    "--skip-beta-leiden",
)
print(proc.stdout[-4000:] if proc.stdout else proc.stderr)
"""),
            cell("code", """\
proc = run_script(
    "18_perturbation_niche_spp1.py",
    "--baseline-h5ad", CFG["baseline_h5ad"],
    "--pred-dir", CFG["pred_dir"],
    "--tag", CFG["tag"],
    "--skip-spp1-perturb",
)
print(proc.stdout[-3000:] if proc.stdout else proc.stderr)
"""),
            cell("code", """\
overall = load_json(f"results/niche_deg/overall_{CFG['spatial_tag']}.json")
spp1 = load_json(f"results/niche_spp1/overall_{CFG['tag']}.json")
overall, spp1
"""),
            cell("code", 'show_figures("figures/niche_deg/fig6_spatial_neighbor_grid_{}.png".format(CFG["spatial_tag"]), max_images=4)'),
            cell("code", 'show_figures("figures/niche_spp1/fig12_spp1_recovery_{}.png".format(CFG["tag"]))'),
        ],
    ),
    (
        "05_paper_findings.ipynb",
        "Paper headline biology validation",
        [
            cell("markdown", "# Paper findings (Zhang et al. Cell 2026)\n\nModule-level hypothesis tests for Icam1, Cd44–Spp1, Il4ra, Cd83, Cd74."),
            cell("code", BOOTSTRAP),
            cell("code", """\
proc = run_script(
    "19_paper_findings_validation.py",
    "--baseline-h5ad", CFG["baseline_h5ad"],
    "--pred-dir", CFG["pred_dir"],
    "--tag", CFG["tag"],
)
print(proc.stdout[-3000:] if proc.stdout else proc.stderr)
"""),
            cell("code", """\
proc = run_script(
    "20_extended_paper_validation.py",
    "--baseline-h5ad", CFG["baseline_h5ad"],
    "--pred-dir", CFG["pred_dir"],
    "--tag", CFG["tag"],
)
print(proc.stdout[-3000:] if proc.stdout else proc.stderr)
"""),
            cell("code", """\
paper = load_json(f"results/paper_findings/overall_{CFG['tag']}.json")
lung = load_json(f"results/extended_paper/overall_{CFG['tag']}.json")
paper, lung
"""),
            cell("code", 'show_figures("figures/paper_findings/fig13_paper_findings_scorecard_{}.png".format(CFG["tag"]))'),
            cell("code", 'show_figures("figures/extended_paper/fig15_lung_icam1_observed_{}.png".format(CFG["tag"]))'),
        ],
    ),
    (
        "06_cnn_guide_enrichment.ipynb",
        "CNN beta microniches & guide enrichment",
        [
            cell("markdown", "# CNN β-microniches → guide enrichment\n\nPer-cell CNN betas define tumor niches; predicted enrichment vs observed sgP/NTC fractions."),
            cell("code", BOOTSTRAP),
            cell("code", """\
status = artifact_status([
    CFG["betadata_dir_cnn"],
    CFG["pred_dir"],
    CFG["pred_dir_cnn"],
])
status
"""),
            cell("code", """\
args = [
    "23_cnn_microniche_enrichment.py",
    "--tag", "cnn",
    "--betadata-dir", CFG["betadata_dir_cnn"],
    "--pred-dir", CFG["pred_dir_cnn"],
    "--seed-pred-dir", CFG["pred_dir"],
    "--baseline-h5ad", CFG["baseline_h5ad"],
]
proc = run_script(*args)
print(proc.stdout)
"""),
            cell("code", """\
summary = load_json("results/cnn_enrichment/overall_cnn.json")
corr = load_csv("results/cnn_enrichment/enrichment_corr_cnn.csv")
summary, corr.sort_values("pearson_r", ascending=False).head(8)
"""),
            cell("code", 'show_figures("figures/cnn_enrichment/fig20_enrichment_scatter_cnn.png")'),
            cell("code", 'show_figures("figures/cnn_enrichment/fig21_enrichment_heatmap_cnn.png")'),
            cell("code", 'show_figures("figures/cnn_enrichment/fig22_cnn_niche_map_Lung_Metastasis_M001_Icam1_cnn.png")'),
        ],
    ),
    (
        "07_validation_dashboard.ipynb",
        "Consolidated validation dashboard",
        [
            cell("markdown", "# Validation dashboard\n\nAggregates metrics from niche DEG, paper findings, lung cohort, and β-Leiden pipelines."),
            cell("code", BOOTSTRAP),
            cell("code", """\
proc = run_script("21_validation_dashboard.py", "--tag", CFG["spatial_tag"])
print(proc.stdout)
"""),
            cell("code", """\
metrics = load_csv(f"results/validation_dashboard/metrics_{CFG['spatial_tag']}.csv")
metrics
"""),
            cell("code", 'show_figures("figures/validation_dashboard/fig20_validation_dashboard_{}.png".format(CFG["spatial_tag"]))'),
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
