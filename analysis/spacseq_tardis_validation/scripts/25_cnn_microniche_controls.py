#!/usr/bin/env python3
"""Negative controls for CNN microniche enrichment.

1. Random niche labels — shuffle CNN β-Leiden labels (should not enrich).
2. Expression Leiden — cluster tumor cells on gene expression + spatial coords.
3. BANKSY — spatial clusters from neighborhood-augmented expression (pybanksy).
"""

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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

import cnn_microniche_utils as cmu

_spec23 = importlib.util.spec_from_file_location("sp23", HERE / "23_cnn_microniche_enrichment.py")
_sp23 = importlib.util.module_from_spec(_spec23)
_spec23.loader.exec_module(_sp23)

resolve_paths = _sp23.resolve_paths
run_slice_enrichment = _sp23.run_slice_enrichment
propagate_niche_labels = _sp23.propagate_niche_labels
assign_pool_niches = _sp23.assign_pool_niches
load_baseline = _sp23.load_baseline

SUBQ_SLICES = _sp23.SUBQ_SLICES
LUNG_SLICE = _sp23.LUNG_SLICE
SUBQ_PERTS = _sp23.SUBQ_PERTS
LUNG_PERTS = _sp23.LUNG_PERTS


def build_slice_context(
    slice_id: str,
    args,
    pooled_baseline: sc.AnnData,
) -> tuple[sc.AnnData, sc.AnnData, np.ndarray, list[str], Path, Path, list[str], sc.AnnData | None]:
    bd, bl_path, pd_dir = resolve_paths(args, slice_id)
    if slice_id.startswith("subQ"):
        baseline = pooled_baseline[pooled_baseline.obs["slice_id"].astype(str) == slice_id].copy()
        perts = SUBQ_PERTS
        gb = None
    else:
        baseline = load_baseline(bl_path)
        baseline.obs["slice_id"] = slice_id
        perts = LUNG_PERTS
        gb = pooled_baseline

    pool = _sp23.load_pool(slice_id, args.data_root)
    pool.obs["slice_id"] = slice_id
    prep = baseline.copy()
    cmu.ensure_cluster_id(prep)
    beta_matrix, score_genes = cmu.build_beta_score_matrix(prep, bd)
    return prep, pool, beta_matrix, score_genes, bd, pd_dir, perts, gb


def run_control_variant(
    slice_id: str,
    variant: str,
    prep: sc.AnnData,
    pool: sc.AnnData,
    beta_matrix: np.ndarray,
    score_genes: list[str],
    perts: list[str],
    args,
    bd: Path,
    pd_dir: Path,
    global_baseline: sc.AnnData | None,
    leiden_kw: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prep = prep.copy()
    pool = pool.copy()

    if variant == "cnn":
        niche_key = "cnn_leiden"
        prep, pool = assign_pool_niches(prep, pool, beta_matrix, slice_id, leiden_kw=leiden_kw, niche_key=niche_key)
    elif variant == "random_niche":
        niche_key = "random_leiden"
        prep, pool = assign_pool_niches(prep, pool, beta_matrix, slice_id, leiden_kw=leiden_kw, niche_key="cnn_leiden")
        tumor_mask = pool.obs["cell_type"].astype(str) == "tumor"
        shuffled = cmu.shuffle_microniche_labels(pool.obs.loc[tumor_mask, "cnn_leiden"], seed=args.random_seed)
        pool.obs.loc[tumor_mask, niche_key] = shuffled.values
        prep_tumor = prep.obs["cell_type"].astype(str) == "tumor"
        prep.obs.loc[prep_tumor, niche_key] = cmu.shuffle_microniche_labels(
            prep.obs.loc[prep_tumor, "cnn_leiden"], seed=args.random_seed,
        ).values
    elif variant == "expr_leiden":
        niche_key = "expr_leiden"
        labels = cmu.assign_slice_expression_clusters(prep, slice_id, "tumor", **leiden_kw)
        prep, pool = propagate_niche_labels(prep, pool, slice_id, labels, niche_key)
    elif variant == "banksy":
        niche_key = "banksy"
        banksy_kw = {
            "resolution": leiden_kw.get("resolution", cmu.DEFAULT_LEIDEN_KW["resolution"]),
            "min_cells": leiden_kw.get("min_cells", cmu.DEFAULT_LEIDEN_KW["min_cells"]),
            "num_neighbours": args.banksy_neighbours,
            "lambda_param": args.banksy_lambda,
        }
        labels = cmu.assign_slice_banksy_clusters(
            prep, slice_id, "tumor", tmp_dir=args.banksy_tmp_dir, **banksy_kw,
        )
        prep, pool = propagate_niche_labels(prep, pool, slice_id, labels, niche_key)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    tag = {
        "cnn": args.tag,
        "random_niche": "random_niche",
        "expr_leiden": "expr_leiden",
        "banksy": "banksy",
    }[variant]
    enrich, corr, _, _ = run_slice_enrichment(
        slice_id,
        perts,
        args.data_root,
        prep,
        bd,
        pd_dir,
        tag,
        global_baseline=global_baseline,
        fallback_pred_dir=args.seed_pred_dir,
        leiden_kw=leiden_kw,
        min_ntc=args.min_ntc,
        min_pert=args.min_pert,
        niche_key=niche_key,
        prep=prep,
        pool=pool,
        score_genes=score_genes,
        beta_matrix=beta_matrix,
    )
    if not corr.empty:
        corr = corr.copy()
        corr["niche_method"] = variant
    if not enrich.empty:
        enrich = enrich.copy()
        enrich["niche_method"] = variant
    return enrich, corr


def summarize_corr(corr_df: pd.DataFrame, method: str) -> dict:
    sub = corr_df[corr_df.get("niche_method", method) == method] if "niche_method" in corr_df.columns else corr_df
    if sub.empty:
        sub = corr_df
    return {
        "method": method,
        "n_tests": int(len(sub)),
        "median_pearson_r": float(sub["pearson_r"].median()) if not sub.empty else None,
        "mean_pearson_r": float(sub["pearson_r"].mean()) if not sub.empty else None,
        "frac_positive_r": float((sub["pearson_r"] > 0).mean()) if not sub.empty else None,
    }


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
    ap.add_argument("--leiden-resolution", type=float, default=cmu.DEFAULT_LEIDEN_KW["resolution"])
    ap.add_argument("--spatial-weight", type=float, default=cmu.DEFAULT_LEIDEN_KW["spatial_weight"])
    ap.add_argument("--min-ntc", type=int, default=2)
    ap.add_argument("--min-pert", type=int, default=2)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--banksy-lambda", type=float, default=0.2, help="BANKSY neighborhood weight λ")
    ap.add_argument("--banksy-neighbours", type=int, default=15, help="BANKSY spatial neighbors k_geom")
    ap.add_argument("--banksy-tmp-dir", type=Path, default=ROOT / "results" / "cnn_enrichment" / "banksy_tmp")
    ap.add_argument("--variants", nargs="+", default=["cnn", "random_niche", "expr_leiden", "banksy"])
    args = ap.parse_args()

    leiden_kw = {
        "resolution": args.leiden_resolution,
        "spatial_weight": args.spatial_weight,
        "min_cells": cmu.DEFAULT_LEIDEN_KW["min_cells"],
    }

    out_dir = ROOT / "results" / "cnn_enrichment"
    fig_dir = ROOT / "figures" / "cnn_enrichment"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    pooled_baseline = load_baseline(args.baseline_h5ad)
    if "slice_id" not in pooled_baseline.obs.columns:
        pooled_baseline.obs["slice_id"] = pooled_baseline.obs_names.str.split("@").str[-1]

    enrich_by_method: dict[str, list[pd.DataFrame]] = {v: [] for v in args.variants}
    corr_by_method: dict[str, list[pd.DataFrame]] = {v: [] for v in args.variants}

    for sl in args.slices:
        prep, pool, beta_matrix, score_genes, bd, pd_dir, perts, gb = build_slice_context(sl, args, pooled_baseline)
        for variant in args.variants:
            print(f"Running {variant} on {sl}...")
            enrich, corr = run_control_variant(
                sl, variant, prep, pool, beta_matrix, score_genes, perts,
                args, bd, pd_dir, gb, leiden_kw,
            )
            if not enrich.empty:
                enrich_by_method[variant].append(enrich)
            if not corr.empty:
                corr_by_method[variant].append(corr)

    enrich_dfs = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in enrich_by_method.items()}
    corr_dfs = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in corr_by_method.items()}

    all_enrich = pd.concat([df for df in enrich_dfs.values() if not df.empty], ignore_index=True)
    all_corr = pd.concat([df for df in corr_dfs.values() if not df.empty], ignore_index=True)
    all_enrich.to_csv(out_dir / "niche_enrichment_controls.csv", index=False)
    all_corr.to_csv(out_dir / "enrichment_corr_controls.csv", index=False)

    summary = {
        "tag": args.tag,
        "random_seed": args.random_seed,
        "leiden_resolution": args.leiden_resolution,
        "spatial_weight": args.spatial_weight,
        "banksy_lambda": args.banksy_lambda,
        "banksy_neighbours": args.banksy_neighbours,
        "variants": args.variants,
        "by_method": [summarize_corr(corr_dfs[v], v) for v in args.variants],
    }
    (out_dir / "overall_controls.json").write_text(json.dumps(summary, indent=2))

    import nb_viz

    fig, _ = nb_viz.plot_microniche_control_comparison(corr_dfs)
    fig.savefig(fig_dir / "fig26_microniche_control_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, _ = nb_viz.plot_microniche_control_heatmap(corr_dfs)
    fig.savefig(fig_dir / "fig27_microniche_control_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, _ = nb_viz.plot_microniche_control_scatter(enrich_dfs, corr_dfs, top_n=3)
    fig.savefig(fig_dir / "fig28_microniche_control_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    for variant in ("random_niche", "expr_leiden", "banksy"):
        if variant in enrich_dfs and not enrich_dfs[variant].empty:
            title_map = {
                "random_niche": "Random label control",
                "expr_leiden": "Expression Leiden control",
                "banksy": "BANKSY spatial clusters",
            }
            fig, _ = nb_viz.plot_cnn_enrichment_scatter(
                enrich_dfs[variant], corr_dfs[variant], top_n=6, tag=variant,
                label_niches=True, color_by_niche=True, show_regression=True,
            )
            fig.suptitle(f"{title_map.get(variant, variant)} ({variant})", fontweight="bold", y=1.02)
            fig.savefig(fig_dir / f"fig20_enrichment_scatter_{variant}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            fig, _ = nb_viz.plot_cnn_enrichment_heatmap(corr_dfs[variant], tag=variant)
            fig.savefig(fig_dir / f"fig21_enrichment_heatmap_{variant}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
