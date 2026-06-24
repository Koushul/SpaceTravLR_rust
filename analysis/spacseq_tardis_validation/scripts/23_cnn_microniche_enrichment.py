#!/usr/bin/env python3
"""CNN β-microniches → guide enrichment prediction in tumor niches."""

from __future__ import annotations

import argparse
import importlib.util
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
import scanpy as sc
import seaborn as sns

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

import cnn_microniche_utils as cmu

_spec09 = importlib.util.spec_from_file_location("sp09", HERE / "09_spatial_validation.py")
_sp09 = importlib.util.module_from_spec(_spec09)
_spec09.loader.exec_module(_sp09)
load_baseline = _sp09.load_baseline
load_pool = _sp09.load_pool

SUBQ_SLICES = ["subQ-1", "subQ-2", "subQ-3", "subQ-4"]
LUNG_SLICE = "Lung_Metastasis_M001"
SUBQ_PERTS = ["Il4ra", "Cd83", "Cd74", "Bcam", "Cks1b", "Ptk6"]
LUNG_PERTS = ["Icam1", "Bcam"]


def resolve_paths(args, slice_id: str) -> tuple[Path, Path, Path]:
    if slice_id == LUNG_SLICE:
        bd = ROOT / "runs" / "lung_m001_cnn"
        if not (bd / "Icam1_betadata.feather").exists():
            bd = args.betadata_dir if cmu.betadata_ready(args.betadata_dir, min_genes=5) else args.seed_betadata_dir
        bl = args.data_root / "slices" / slice_id / "baseline_ntc.h5ad"
        pd_dir = args.pred_dir if (args.pred_dir / "predicted_KO_Icam1.feather").exists() else args.seed_pred_dir
    else:
        bd = args.betadata_dir if cmu.betadata_ready(args.betadata_dir) else args.seed_betadata_dir
        bl = args.baseline_h5ad
        pd_dir = args.pred_dir if any(args.pred_dir.glob("predicted_KO_*.feather")) else args.seed_pred_dir
    return bd, bl, pd_dir


def assign_pool_niches(
    prep: sc.AnnData,
    pool: sc.AnnData,
    beta_matrix: np.ndarray,
    slice_id: str,
) -> tuple[sc.AnnData, sc.AnnData]:
    cmu.ensure_cluster_id(prep)
    labels = cmu.assign_slice_microniches(prep, beta_matrix, slice_id, "tumor")
    prep = prep.copy()
    prep.obs["cnn_leiden"] = labels

    pool = pool.copy()
    pool_labels = pd.Series("unassigned", index=pool.obs_names, dtype=str)
    prep_names = prep.obs_names
    for bc in pool.obs_names:
        key = cmu.map_pool_to_prep(slice_id, bc, prep_names)
        if key is not None:
            pool_labels[bc] = prep.obs.loc[key, "cnn_leiden"]
    pool.obs["cnn_leiden"] = cmu.knn_assign_perturbed(pool, pool_labels, "tumor")
    return prep, pool


def resolve_pred_path(pred_dir: Path, fallback_dir: Path, perturb: str) -> Path | None:
    for d in (pred_dir, fallback_dir):
        p = d / f"predicted_KO_{perturb}.feather"
        if p.exists():
            return p
    return None


def run_slice_enrichment(
    slice_id: str,
    perturbations: list[str],
    data_root: Path,
    baseline: sc.AnnData,
    betadata_dir: Path,
    pred_dir: Path,
    tag: str,
    global_baseline: sc.AnnData | None = None,
    fallback_pred_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, sc.AnnData, sc.AnnData]:
    pool = load_pool(slice_id, data_root)
    pool.obs["slice_id"] = slice_id

    prep = baseline.copy()
    if slice_id.startswith("subQ"):
        prep = prep[prep.obs["slice_id"].astype(str) == slice_id].copy()
    cmu.ensure_cluster_id(prep)

    beta_matrix, score_genes = cmu.build_beta_score_matrix(prep, betadata_dir)
    prep, pool = assign_pool_niches(prep, pool, beta_matrix, slice_id)

    all_enrich: list[pd.DataFrame] = []
    all_corr: list[dict] = []

    for pert in perturbations:
        n_pert = int((pool.obs["target_gene"].astype(str) == pert).sum())
        if n_pert < 5:
            continue
        pred_path = resolve_pred_path(pred_dir, fallback_pred_dir or pred_dir, pert)
        if pred_path is None:
            continue
        pred = pd.read_feather(pred_path)
        if "CellID" in pred.columns:
            pred = pred.set_index("CellID")

        profile = cmu.PERT_ENRICHMENT_PROFILE.get(
            pert, {"exclusion_sign": 0.0, "escape_up": [], "escape_dn": []}
        )
        target_idx = score_genes.index(pert) if pert in score_genes else None
        cnn_by_cell = None
        if target_idx is not None:
            tumor_mask = prep.obs["cell_type"].astype(str) == "tumor"
            beta_tumor = beta_matrix[np.where(tumor_mask.values)[0]]
            cnn_by_cell = pd.Series(beta_tumor[:, target_idx], index=prep.obs_names[tumor_mask])
        obs_df = cmu.observed_log_enrichment(pool, pert, "tumor", "cnn_leiden")
        pred_df = cmu.predicted_niche_scores(
            prep, pool, pred, pert, "tumor", "cnn_leiden", score_genes, profile, cnn_by_cell,
            global_baseline=global_baseline,
        )
        merged = cmu.merge_obs_pred_enrichment(obs_df, pred_df)
        corr = cmu.enrichment_correlation(merged)
        corr.update({"slice": slice_id, "perturbation": pert, "n_pert_cells": n_pert, "tag": tag})
        all_corr.append(corr)
        if not merged.empty:
            merged["slice"] = slice_id
            merged["perturbation"] = pert
            merged["tag"] = tag
            all_enrich.append(merged)

    return (
        pd.concat(all_enrich, ignore_index=True) if all_enrich else pd.DataFrame(),
        pd.DataFrame(all_corr),
        prep,
        pool,
    )


def plot_scatter(enrich_df: pd.DataFrame, corr_df: pd.DataFrame, fig_dir: Path, tag: str) -> None:
    if enrich_df.empty or corr_df.empty:
        return
    pairs = corr_df.dropna(subset=["pearson_r"]).sort_values("pearson_r", ascending=False).head(6)
    if pairs.empty:
        pairs = corr_df.head(6)
    n = len(pairs)
    cols = min(3, max(n, 1))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), squeeze=False)
    for ax, (_, row) in zip(axes.ravel(), pairs.iterrows()):
        sub = enrich_df[
            (enrich_df["slice"] == row["slice"]) & (enrich_df["perturbation"] == row["perturbation"])
        ]
        ax.scatter(sub["pred_enrichment_score"], sub["obs_log2_enrichment"], s=35, alpha=0.85, c="#2563eb")
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.axvline(0, color="k", lw=0.4, alpha=0.4)
        r = row.get("pearson_r", float("nan"))
        ax.set_title(f"sg{row['perturbation']} | {row['slice']}\nr={r:+.2f}, n={int(row['n_niches'])}", fontsize=9)
        ax.set_xlabel("Predicted enrichment score")
        ax.set_ylabel("Observed log2 OR")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"CNN β-microniche guide enrichment ({tag})", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig20_enrichment_scatter_{tag}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(corr_df: pd.DataFrame, fig_dir: Path, tag: str) -> None:
    if corr_df.empty:
        return
    pivot = corr_df.pivot_table(index="perturbation", columns="slice", values="pearson_r", aggfunc="first")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(0.7 * len(pivot.columns) + 2, 0.5 * len(pivot.index) + 1.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title(f"Obs vs pred niche enrichment correlation ({tag})")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig21_enrichment_heatmap_{tag}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_niche_maps(slice_id: str, pool: sc.AnnData, perturb: str, fig_dir: Path, tag: str) -> None:
    ct = pool[pool.obs["cell_type"].astype(str) == "tumor"].copy()
    if "cnn_leiden" not in ct.obs.columns or ct.n_obs < 20:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    ntc = ct[ct.obs["target_gene"].astype(str) == "non-targeting"]
    pert = ct[ct.obs["target_gene"].astype(str) == perturb]
    cmap = plt.colormaps["tab10"]
    for ax, (sub, title) in zip(axes, [(ntc, "NTC tumor"), (pert, f"sg{perturb} tumor"), (ct, "All tumor")]):
        if sub.n_obs == 0:
            ax.axis("off")
            continue
        labs = sub.obs["cnn_leiden"].astype(str)
        for i, lab in enumerate(sorted(labs.unique())):
            m = labs == lab
            c = "#dddddd" if lab in ("unassigned", "nan") else cmap(i % 10)
            ax.scatter(sub.obsm["spatial"][m, 0], sub.obsm["spatial"][m, 1], c=[c], s=2, alpha=0.7, rasterized=True)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle(f"{slice_id} CNN β-Leiden tumor niches — sg{perturb} ({tag})", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig22_cnn_niche_map_{slice_id}_{perturb}_{tag}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cnn")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--betadata-dir", type=Path, default=ROOT / "runs/baseline_pooled_cnn")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "results/predictions_cnn")
    ap.add_argument("--baseline-h5ad", type=Path, default=ROOT / "data/pooled/baseline_ntc.h5ad")
    ap.add_argument("--slices", nargs="+", default=SUBQ_SLICES + [LUNG_SLICE])
    ap.add_argument("--seed-betadata-dir", type=Path, default=ROOT / "runs/baseline_pooled_seed")
    ap.add_argument("--seed-pred-dir", type=Path, default=ROOT / "results/predictions_tuned")
    args = ap.parse_args()

    out_dir = ROOT / "results" / "cnn_enrichment"
    fig_dir = ROOT / "figures" / "cnn_enrichment"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_cell = cmu.betadata_is_per_cell(args.betadata_dir) if cmu.betadata_ready(args.betadata_dir, min_genes=5) else False
    print(f"betadata={args.betadata_dir} per_cell={per_cell}")

    pooled_baseline = load_baseline(args.baseline_h5ad)
    if "slice_id" not in pooled_baseline.obs.columns:
        pooled_baseline.obs["slice_id"] = pooled_baseline.obs_names.str.split("@").str[-1]

    all_enrich: list[pd.DataFrame] = []
    all_corr: list[pd.DataFrame] = []

    for sl in args.slices:
        bd, bl_path, pd_dir = resolve_paths(args, sl)
        if sl.startswith("subQ"):
            baseline = pooled_baseline[pooled_baseline.obs["slice_id"].astype(str) == sl].copy()
            perts = SUBQ_PERTS
            gb = None
        else:
            baseline = load_baseline(bl_path)
            baseline.obs["slice_id"] = sl
            perts = LUNG_PERTS
            gb = pooled_baseline

        enrich, corr, prep, pool = run_slice_enrichment(
            sl, perts, args.data_root, baseline, bd, pd_dir, args.tag,
            global_baseline=gb, fallback_pred_dir=args.seed_pred_dir,
        )
        if not enrich.empty:
            all_enrich.append(enrich)
        if not corr.empty:
            all_corr.append(corr)
        for pert in perts:
            if (pool.obs["target_gene"].astype(str) == pert).sum() >= 20:
                plot_niche_maps(sl, pool, pert, fig_dir, args.tag)

    enrich_df = pd.concat(all_enrich, ignore_index=True) if all_enrich else pd.DataFrame()
    corr_df = pd.concat(all_corr, ignore_index=True) if all_corr else pd.DataFrame()

    enrich_df.to_csv(out_dir / f"niche_enrichment_{args.tag}.csv", index=False)
    corr_df.to_csv(out_dir / f"enrichment_corr_{args.tag}.csv", index=False)

    summary = {
        "tag": args.tag,
        "per_cell_betas": per_cell,
        "n_enrichment_tests": int(len(corr_df)),
        "median_pearson_r": float(corr_df["pearson_r"].median()) if not corr_df.empty else None,
        "mean_pearson_r": float(corr_df["pearson_r"].mean()) if not corr_df.empty else None,
        "best_cases": (
            corr_df.nlargest(5, "pearson_r")[["slice", "perturbation", "pearson_r", "n_niches"]].to_dict("records")
            if not corr_df.empty else []
        ),
    }
    (out_dir / f"overall_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    plot_scatter(enrich_df, corr_df, fig_dir, args.tag)
    plot_heatmap(corr_df, fig_dir, args.tag)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
