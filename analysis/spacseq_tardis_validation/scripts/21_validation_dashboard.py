#!/usr/bin/env python3
"""Consolidated validation dashboard across all analysis pipelines.

Aggregates tuned-model results from niche DEG, Spp1, paper findings, and
extended lung cohort into summary tables and a multi-panel figure.

Outputs:
  results/validation_dashboard/overall_{tag}.json
  results/validation_dashboard/metrics_{tag}.csv
  figures/validation_dashboard/fig20_validation_dashboard_{tag}.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _py_boot import ensure_boot

ensure_boot()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def collect_metrics(tag: str) -> pd.DataFrame:
    rows = []

    niche_spp1 = read_json(ROOT / "results" / "niche_spp1" / f"overall_{tag}.json")
    if niche_spp1:
        rows.append({"category": "Direct sgP DEG", "metric": "median_pearson_r", "value": niche_spp1.get("direct_median_pearson")})
        rows.append({"category": "Spatial niche DEG", "metric": "median_pearson_r", "value": niche_spp1.get("spatial_median_pearson")})

    paper = read_json(ROOT / "results" / "paper_findings" / f"overall_{tag}.json")
    if paper:
        rows.append({"category": "Paper modules (obs)", "metric": "frac_support", "value": paper.get("frac_obs_support")})
        rows.append({"category": "Paper modules (pred)", "metric": "frac_support", "value": paper.get("frac_pred_support")})

    extended = read_json(ROOT / "results" / "extended_paper" / f"overall_{tag}.json")
    if extended:
        n_icam = extended.get("lung_icam1_n_modules") or 1
        n_bcam = extended.get("lung_bcam_n_modules") or 1
        rows.append({
            "category": "Lung Icam1 (obs)",
            "metric": "frac_support",
            "value": (extended.get("lung_icam1_modules_obs_support") or 0) / n_icam,
        })
        rows.append({
            "category": "Lung Bcam (obs)",
            "metric": "frac_support",
            "value": (extended.get("lung_bcam_modules_obs_support") or 0) / n_bcam,
        })

    scorecard = read_csv(ROOT / "results" / "scorecard" / "prediction_scorecard.csv")
    if not scorecard.empty and "model" in scorecard.columns:
        tuned = scorecard[(scorecard.model == tag) & (scorecard.level == "cell_type") & (scorecard.compartment == "immune")]
        if not tuned.empty:
            rows.append({"category": "Multislice immune r", "metric": "median_pearson_r", "value": float(tuned.median_r.iloc[0])})

    beta = read_json(ROOT / "results" / "beta_leiden" / f"overall_{tag}.json")
    if beta and "by_niche_type" in beta:
        rows.append({
            "category": "β-Leiden niche",
            "metric": "median_pearson_r",
            "value": beta["by_niche_type"].get("beta_leiden"),
        })

    niche_deg = read_csv(ROOT / "results" / "niche_deg" / f"spatial_neighbor_stats_{tag}.csv")
    if not niche_deg.empty and "pearson_r" in niche_deg.columns:
        rows.append({"category": "Spatial kNN (script 13)", "metric": "median_pearson_r", "value": float(niche_deg.pearson_r.median())})

    return pd.DataFrame([r for r in rows if r.get("value") is not None and np.isfinite(r["value"])])


def plot_dashboard(metrics: pd.DataFrame, tag: str, out_dir: Path, fig_dir: Path) -> None:
    if metrics.empty:
        return
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    m = metrics[metrics.metric == "median_pearson_r"].copy()
    if not m.empty:
        colors = ["#2166ac" if v >= 0 else "#b2182b" for v in m.value]
        ax0.barh(m.category, m.value, color=colors, edgecolor="k", linewidth=0.3)
        ax0.axvline(0, color="k", lw=0.8)
        ax0.set_xlabel("Pearson r")
        ax0.set_title("Concordance metrics (β-tuned)", fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 1])
    m2 = metrics[metrics.metric == "frac_support"].copy()
    if not m2.empty:
        colors = ["#4daf4a" if v >= 0.6 else "#984ea3" for v in m2.value]
        ax1.barh(m2.category, m2.value, color=colors, edgecolor="k", linewidth=0.3)
        ax1.axvline(0.6, color="#b2182b", ls="--", lw=1)
        ax1.set_xlim(0, 1)
        ax1.set_xlabel("Fraction modules ≥60% sign match")
        ax1.set_title("Paper biology recapitulation", fontweight="bold")

    ax2 = fig.add_subplot(gs[1, :])
    direct = read_csv(ROOT / "results" / "niche_spp1" / f"direct_cell_deg_stats_{tag}.csv")
    if not direct.empty and "pearson_r" in direct.columns:
        show = direct.sort_values("pearson_r", ascending=True)
        ax2.barh(
            [f"{r.perturbation}|{r.cell_type}" for _, r in show.iterrows()],
            show.pearson_r,
            color=["#2166ac" if v >= 0 else "#b2182b" for v in show.pearson_r],
            edgecolor="k", linewidth=0.3,
        )
        ax2.axvline(0, color="k", lw=0.8)
        ax2.set_xlabel("Pearson r (obs vs pred Δ on sgP cells)")
        ax2.set_title("Direct perturbed-cell DEG concordance by perturbation × cell type", fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "Direct DEG stats not found", ha="center", va="center", transform=ax2.transAxes)
        ax2.axis("off")

    fig.suptitle(
        f"SpaceTravLR × SPAC-seq validation dashboard ({tag})\n"
        "Zhang et al. Cell 2026 — subQ pooled + lung M001 observed",
        fontsize=13, fontweight="bold",
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"fig20_validation_dashboard_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "tag": tag,
        "n_metrics": int(len(metrics)),
        "metrics": metrics.to_dict("records"),
    }
    (out_dir / f"overall_{tag}.json").write_text(json.dumps(summary, indent=2, default=str))
    metrics.to_csv(out_dir / f"metrics_{tag}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tuned")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "validation_dashboard")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures" / "validation_dashboard")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    metrics = collect_metrics(args.tag)
    plot_dashboard(metrics, args.tag, args.out_dir, args.fig_dir)
    print(json.dumps(read_json(args.out_dir / f"overall_{args.tag}.json"), indent=2))


if __name__ == "__main__":
    main()
