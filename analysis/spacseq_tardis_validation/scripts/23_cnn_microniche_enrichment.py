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
    v2 = ROOT / "runs" / "baseline_pooled_cnn_v2"
    pooled_cnn = v2 if cmu.betadata_ready(v2, min_genes=30) else args.betadata_dir
    if slice_id == LUNG_SLICE:
        bd = ROOT / "runs" / "lung_m001_cnn"
        if not (bd / "Icam1_betadata.feather").exists():
            bd = pooled_cnn if cmu.betadata_ready(pooled_cnn, min_genes=5) else args.seed_betadata_dir
        bl = args.data_root / "slices" / slice_id / "baseline_ntc.h5ad"
        pd_dir = args.pred_dir if (args.pred_dir / "predicted_KO_Icam1.feather").exists() else args.seed_pred_dir
    else:
        bd = pooled_cnn if cmu.betadata_ready(pooled_cnn) else args.seed_betadata_dir
        bl = args.baseline_h5ad
        pd_dir = args.pred_dir if any(args.pred_dir.glob("predicted_KO_*.feather")) else args.seed_pred_dir
    return bd, bl, pd_dir


def propagate_niche_labels(
    prep: sc.AnnData,
    pool: sc.AnnData,
    slice_id: str,
    labels: pd.Series,
    niche_key: str,
) -> tuple[sc.AnnData, sc.AnnData]:
    prep = prep.copy()
    prep.obs[niche_key] = labels
    pool = pool.copy()
    pool_labels = pd.Series("unassigned", index=pool.obs_names, dtype=str)
    prep_names = prep.obs_names
    for bc in pool.obs_names:
        key = cmu.map_pool_to_prep(slice_id, bc, prep_names)
        if key is not None and key in prep.obs_names:
            pool_labels[bc] = prep.obs.loc[key, niche_key]
    pool.obs[niche_key] = cmu.knn_assign_perturbed(pool, pool_labels, "tumor")
    return prep, pool


def assign_pool_niches(
    prep: sc.AnnData,
    pool: sc.AnnData,
    beta_matrix: np.ndarray,
    slice_id: str,
    leiden_kw: dict | None = None,
    niche_key: str = "cnn_leiden",
) -> tuple[sc.AnnData, sc.AnnData]:
    cmu.ensure_cluster_id(prep)
    labels = cmu.assign_slice_microniches(prep, beta_matrix, slice_id, "tumor", **(leiden_kw or {}))
    return propagate_niche_labels(prep, pool, slice_id, labels, niche_key)


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
    leiden_kw: dict | None = None,
    min_ntc: int = 2,
    min_pert: int = 2,
    niche_key: str = "cnn_leiden",
    prep: sc.AnnData | None = None,
    pool: sc.AnnData | None = None,
    score_genes: list[str] | None = None,
    beta_matrix: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, sc.AnnData, sc.AnnData]:
    if pool is None:
        pool = load_pool(slice_id, data_root)
        pool.obs["slice_id"] = slice_id

    if prep is None:
        prep = baseline.copy()
        if slice_id.startswith("subQ"):
            prep = prep[prep.obs["slice_id"].astype(str) == slice_id].copy()
        cmu.ensure_cluster_id(prep)

    if beta_matrix is None or score_genes is None:
        beta_matrix, score_genes = cmu.build_beta_score_matrix(
            prep, betadata_dir, gene_filter=cmu.MICRONICHE_CLUSTER_GENES,
        )
    if niche_key not in pool.obs.columns:
        prep, pool = assign_pool_niches(prep, pool, beta_matrix, slice_id, leiden_kw=leiden_kw, niche_key=niche_key)

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
        obs_df = cmu.observed_log_enrichment(pool, pert, "tumor", niche_key, min_ntc=min_ntc, min_pert=min_pert)
        pred_df = cmu.predicted_niche_scores(
            prep, pool, pred, pert, "tumor", niche_key, score_genes, profile, cnn_by_cell,
            global_baseline=global_baseline,
            min_ntc_per_niche=min_ntc,
            min_pert_per_niche=min_pert,
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
    import nb_viz
    fig, _ = nb_viz.plot_cnn_enrichment_scatter(
        enrich_df, corr_df, top_n=6, tag=tag,
        label_niches=True, color_by_niche=True, show_regression=True,
    )
    fig.savefig(fig_dir / f"fig20_enrichment_scatter_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_histology(
    enrich_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    pool_by_slice: dict[str, sc.AnnData],
    fig_dir: Path,
    tag: str,
    *,
    mc38_dir: Path,
    top_n: int = 6,
    slice_filter: str | None = None,
    out_stem: str | None = None,
) -> None:
    if enrich_df.empty or corr_df.empty or not pool_by_slice:
        return
    import nb_viz
    import spatial_histology as sh

    sh.apply_publication_style()
    fig, _ = nb_viz.plot_cnn_enrichment_scatter_with_histology(
        enrich_df,
        corr_df,
        pool_by_slice,
        top_n=top_n,
        tag=tag,
        mc38_dir=mc38_dir,
        slice_filter=slice_filter,
    )
    stem = out_stem or (f"fig20_enrichment_scatter_histology_{tag}" if not slice_filter else f"fig25_lung_enrichment_scatter_histology_{tag}")
    fig.savefig(fig_dir / f"{stem}.svg", **sh.FIGURE_PARAMS)
    fig.savefig(fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight", transparent=True)
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


def export_spatial_tumor(
    pool: sc.AnnData,
    slice_id: str,
    out_dir: Path,
    tag: str,
    genes: list[str] | None = None,
) -> Path | None:
    ct = pool[pool.obs["cell_type"].astype(str) == "tumor"].copy()
    if ct.n_obs == 0 or "cnn_leiden" not in ct.obs.columns:
        return None
    df = pd.DataFrame({
        "x": ct.obsm["spatial"][:, 0],
        "y": ct.obsm["spatial"][:, 1],
        "cnn_leiden": ct.obs["cnn_leiden"].astype(str).values,
        "target_gene": ct.obs["target_gene"].astype(str).values,
        "slice": slice_id,
        "tag": tag,
    })
    genes = genes or cmu.PAPER_LUNG_GENES
    for g in genes:
        if g in ct.var_names:
            x = ct[:, g].X
            arr = x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()
            df[f"expr_{g}"] = arr
    path = out_dir / f"spatial_tumor_{slice_id}_{tag}.parquet"
    df.to_parquet(path, index=False)
    return path


def _niche_colors(labels: pd.Series) -> dict[str, tuple]:
    uniq = sorted(l for l in labels.unique() if l not in ("unassigned", "nan", "0") or labels.nunique() <= 2)
    if not uniq:
        uniq = sorted(labels.unique())
    cmap = plt.colormaps.get_cmap("tab20")
    return {lab: cmap(i % 20) for i, lab in enumerate(uniq)}


def plot_lung_composite(
    enrich: pd.DataFrame,
    corr: pd.DataFrame,
    pool: sc.AnnData,
    slice_id: str,
    perturb: str,
    out_dir: Path,
    fig_dir: Path,
    tag: str,
) -> None:
    import nb_viz

    spatial_path = out_dir / f"spatial_tumor_{slice_id}_{tag}.parquet"
    if not spatial_path.exists():
        export_spatial_tumor(pool, slice_id, out_dir, tag, genes=cmu.PAPER_LUNG_GENES)
    if not spatial_path.exists():
        return
    spatial_df = pd.read_parquet(spatial_path)
    fig, _ = nb_viz.plot_lung_m001_composite(
        enrich, corr, spatial_df, perturb=perturb, slice_id=slice_id, tag=tag,
    )
    fig.savefig(fig_dir / f"fig24_lung_composite_{perturb}_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_niche_maps(slice_id: str, pool: sc.AnnData, perturb: str, fig_dir: Path, tag: str) -> None:
    ct = pool[pool.obs["cell_type"].astype(str) == "tumor"].copy()
    if "cnn_leiden" not in ct.obs.columns or ct.n_obs < 20:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ntc = ct[ct.obs["target_gene"].astype(str) == "non-targeting"]
    pert = ct[ct.obs["target_gene"].astype(str) == perturb]
    colors = _niche_colors(ct.obs["cnn_leiden"].astype(str))
    for ax, (sub, title) in zip(axes, [(ntc, "NTC tumor"), (pert, f"sg{perturb} tumor"), (ct, "All tumor")]):
        if sub.n_obs == 0:
            ax.axis("off")
            continue
        labs = sub.obs["cnn_leiden"].astype(str)
        for lab in sorted(labs.unique()):
            m = labs == lab
            c = "#dddddd" if lab in ("unassigned", "nan") else colors.get(lab, "#888888")
            ax.scatter(sub.obsm["spatial"][m, 0], sub.obsm["spatial"][m, 1], c=[c], s=3, alpha=0.75, rasterized=True)
        ax.set_title(f"{title} (n={sub.n_obs})", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    n_niches = ct.obs["cnn_leiden"].astype(str).nunique()
    fig.suptitle(
        f"{slice_id} CNN β-Leiden tumor microniches (n={n_niches}) — sg{perturb} ({tag})",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig22_cnn_niche_map_{slice_id}_{perturb}_{tag}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_overview(pool: sc.AnnData, slice_id: str, fig_dir: Path, tag: str) -> None:
    """Single-panel spatial map of all tumor microniches (no perturbation split)."""
    ct = pool[pool.obs["cell_type"].astype(str) == "tumor"].copy()
    if "cnn_leiden" not in ct.obs.columns or ct.n_obs < 20:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    labs = ct.obs["cnn_leiden"].astype(str)
    colors = _niche_colors(labs)
    for lab in sorted(labs.unique()):
        m = labs == lab
        c = "#dddddd" if lab in ("unassigned", "nan") else colors.get(lab, "#888888")
        ax.scatter(ct.obsm["spatial"][m, 0], ct.obsm["spatial"][m, 1], c=[c], s=4, alpha=0.8, rasterized=True, label=lab.split("|")[-1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{slice_id} tumor CNN β-microniches on tissue (n={labs.nunique()} niches)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig23_spatial_microniches_{slice_id}_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cnn_v2")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--betadata-dir", type=Path, default=ROOT / "runs/baseline_pooled_cnn_v2")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "results/predictions_tuned")
    ap.add_argument("--baseline-h5ad", type=Path, default=ROOT / "data/pooled/baseline_ntc.h5ad")
    ap.add_argument("--slices", nargs="+", default=SUBQ_SLICES + [LUNG_SLICE])
    ap.add_argument("--seed-betadata-dir", type=Path, default=ROOT / "runs/baseline_pooled_seed")
    ap.add_argument("--seed-pred-dir", type=Path, default=ROOT / "results/predictions_tuned")
    ap.add_argument("--mc38-dir", type=Path, default=(ROOT.parent / "mc38_visiumhd").resolve())
    ap.add_argument("--leiden-resolution", type=float, default=cmu.DEFAULT_LEIDEN_KW["resolution"])
    ap.add_argument("--spatial-weight", type=float, default=cmu.DEFAULT_LEIDEN_KW["spatial_weight"])
    ap.add_argument("--min-ntc", type=int, default=2, help="Min NTC tumor cells per niche (obs + pred)")
    ap.add_argument("--min-pert", type=int, default=2, help="Min sgP tumor cells per niche (observed)")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures" / "cnn_enrichment")
    ap.add_argument("--figures-only", action="store_true", help="Skip CSV writes; use cached enrich/corr if present")
    args = ap.parse_args()

    leiden_kw = {
        "resolution": args.leiden_resolution,
        "spatial_weight": args.spatial_weight,
        "n_pcs": cmu.DEFAULT_LEIDEN_KW["n_pcs"],
        "min_cells": cmu.DEFAULT_LEIDEN_KW["min_cells"],
    }

    out_dir = ROOT / "results" / "cnn_enrichment"
    fig_dir = args.fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_cell = cmu.betadata_is_per_cell(args.betadata_dir) if cmu.betadata_ready(args.betadata_dir, min_genes=5) else False
    print(f"betadata={args.betadata_dir} per_cell={per_cell}")

    pooled_baseline = load_baseline(args.baseline_h5ad)
    if "slice_id" not in pooled_baseline.obs.columns:
        pooled_baseline.obs["slice_id"] = pooled_baseline.obs_names.str.split("@").str[-1]

    all_enrich: list[pd.DataFrame] = []
    all_corr: list[pd.DataFrame] = []
    pool_by_slice: dict[str, sc.AnnData] = {}

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
            leiden_kw=leiden_kw, min_ntc=args.min_ntc, min_pert=args.min_pert,
        )
        export_spatial_tumor(pool, sl, out_dir, args.tag, genes=cmu.PAPER_LUNG_GENES)
        plot_spatial_overview(pool, sl, fig_dir, args.tag)
        pool_by_slice[sl] = pool
        if sl == LUNG_SLICE and not enrich.empty:
            for pert in perts:
                if (pool.obs["target_gene"].astype(str) == pert).sum() >= 20:
                    plot_lung_composite(enrich, corr, pool, sl, pert, out_dir, fig_dir, args.tag)
        if not enrich.empty:
            all_enrich.append(enrich)
        if not corr.empty:
            all_corr.append(corr)
        for pert in perts:
            if (pool.obs["target_gene"].astype(str) == pert).sum() >= 20:
                plot_niche_maps(sl, pool, pert, fig_dir, args.tag)

    enrich_df = pd.concat(all_enrich, ignore_index=True) if all_enrich else pd.DataFrame()
    corr_df = pd.concat(all_corr, ignore_index=True) if all_corr else pd.DataFrame()

    if args.figures_only:
        enrich_path = out_dir / f"niche_enrichment_{args.tag}.csv"
        corr_path = out_dir / f"enrichment_corr_{args.tag}.csv"
        if enrich_df.empty and enrich_path.exists():
            enrich_df = pd.read_csv(enrich_path)
        if corr_df.empty and corr_path.exists():
            corr_df = pd.read_csv(corr_path)
    else:
        enrich_df.to_csv(out_dir / f"niche_enrichment_{args.tag}.csv", index=False)
        corr_df.to_csv(out_dir / f"enrichment_corr_{args.tag}.csv", index=False)

    summary = {
        "tag": args.tag,
        "per_cell_betas": per_cell,
        "leiden_resolution": args.leiden_resolution,
        "spatial_weight": args.spatial_weight,
        "min_ntc": args.min_ntc,
        "min_pert": args.min_pert,
        "n_enrichment_tests": int(len(corr_df)),
        "median_pearson_r": float(corr_df["pearson_r"].median()) if not corr_df.empty else None,
        "mean_pearson_r": float(corr_df["pearson_r"].mean()) if not corr_df.empty else None,
        "best_cases": (
            corr_df.nlargest(5, "pearson_r")[["slice", "perturbation", "pearson_r", "n_niches"]].to_dict("records")
            if not corr_df.empty else []
        ),
    }
    if not args.figures_only:
        (out_dir / f"overall_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    plot_scatter(enrich_df, corr_df, fig_dir, args.tag)
    plot_scatter_histology(enrich_df, corr_df, pool_by_slice, fig_dir, args.tag, mc38_dir=args.mc38_dir)
    plot_scatter_histology(
        enrich_df, corr_df, pool_by_slice, fig_dir, args.tag,
        mc38_dir=args.mc38_dir, top_n=2, slice_filter=LUNG_SLICE,
        out_stem=f"fig25_lung_enrichment_scatter_histology_{args.tag}",
    )
    plot_heatmap(corr_df, fig_dir, args.tag)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
