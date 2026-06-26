#!/usr/bin/env python3
"""Pathway–microniche concordance: functional niches vs observed SPAC-seq vs SpaceTravLR."""

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

_spec05 = importlib.util.spec_from_file_location("fig05", HERE / "05_final_report_figures.py")
_fig05 = importlib.util.module_from_spec(_spec05)
_spec05.loader.exec_module(_fig05)
GENE_SETS = _fig05.GENE_SETS

_spec23 = importlib.util.spec_from_file_location("sp23", HERE / "23_cnn_microniche_enrichment.py")
_sp23 = importlib.util.module_from_spec(_spec23)
_spec23.loader.exec_module(_sp23)

resolve_paths = _sp23.resolve_paths
assign_pool_niches = _sp23.assign_pool_niches
run_slice_enrichment = _sp23.run_slice_enrichment
load_baseline = _sp23.load_baseline
load_pool = _sp23.load_pool
resolve_pred_path = _sp23.resolve_pred_path
SUBQ_SLICES = _sp23.SUBQ_SLICES
LUNG_SLICE = _sp23.LUNG_SLICE
SUBQ_PERTS = _sp23.SUBQ_PERTS
LUNG_PERTS = _sp23.LUNG_PERTS


def _plot_all(
    distinct_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    concord_df: pd.DataFrame,
    fig_dir: Path,
    tag: str,
) -> None:
    import nb_viz
    import spatial_histology as sh

    sh.apply_publication_style()
    fig, _ = nb_viz.plot_pathway_distinctness_bars(distinct_df, tag=tag)
    sh.save_figure_png_svg(fig, fig_dir / f"fig31_pathway_distinctness_{tag}.png", dpi=300)
    plt.close(fig)

    fig, _ = nb_viz.plot_pathway_obs_pred_concordance(concord_df, delta_df, tag=tag, expected_only=True)
    sh.save_figure_png_svg(fig, fig_dir / f"fig32_pathway_obs_pred_concordance_{tag}.png", dpi=300)
    plt.close(fig)

    fig, _ = nb_viz.plot_pathway_microniche_heatmap(concord_df, tag=tag, average_slices=True)
    sh.save_figure_png_svg(fig, fig_dir / f"fig33_pathway_microniche_heatmap_{tag}.png", dpi=300)
    plt.close(fig)

    fig, _ = nb_viz.plot_pathway_microniche_heatmap(concord_df, tag=tag, average_slices=False)
    sh.save_figure_png_svg(fig, fig_dir / f"fig33_pathway_microniche_heatmap_by_slice_{tag}.png", dpi=300)
    plt.close(fig)

    fig, _ = nb_viz.plot_pathway_enrichment_tie(delta_df, concord_df, tag=tag)
    sh.save_figure_png_svg(fig, fig_dir / f"fig34_pathway_enrichment_tie_{tag}.png", dpi=300)
    plt.close(fig)

    fig, _ = nb_viz.plot_pathway_niche_facets(delta_df, concord_df, top_n=6, tag=tag)
    sh.save_figure_png_svg(fig, fig_dir / f"fig35_pathway_niche_facets_{tag}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cnn_v2")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--baseline-h5ad", type=Path, default=ROOT / "data/pooled/baseline_ntc.h5ad")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "results/predictions_tuned")
    ap.add_argument("--seed-pred-dir", type=Path, default=ROOT / "results/predictions_tuned")
    ap.add_argument("--slices", nargs="+", default=SUBQ_SLICES + [LUNG_SLICE])
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures" / "cnn_microniche_v2_improved")
    ap.add_argument("--min-ntc", type=int, default=2)
    ap.add_argument("--min-pert", type=int, default=2)
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()

    pathways = cmu.microniche_pathway_gene_sets(GENE_SETS)
    out_dir = ROOT / "results" / "cnn_enrichment"
    fig_dir = args.fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        distinct_df = pd.read_csv(out_dir / f"pathway_distinctness_{args.tag}.csv")
        delta_df = pd.read_csv(out_dir / f"pathway_niche_deltas_{args.tag}.csv")
        concord_df = pd.read_csv(out_dir / f"pathway_concordance_{args.tag}.csv")
        _plot_all(distinct_df, delta_df, concord_df, fig_dir, args.tag)
        print((out_dir / f"pathway_microniche_overall_{args.tag}.json").read_text())
        return

    per_slice_leiden = cmu.load_per_slice_leiden_config(cmu.leiden_config_path_for_tag(args.tag))
    pooled_baseline = load_baseline(args.baseline_h5ad)
    if "slice_id" not in pooled_baseline.obs.columns:
        pooled_baseline.obs["slice_id"] = pooled_baseline.obs_names.str.split("@").str[-1]

    distinct_rows: list[dict] = []
    delta_rows: list[dict] = []
    concord_rows: list[dict] = []
    tie_rows: list[dict] = []

    for sl in args.slices:
        run_args = argparse.Namespace(
            betadata_dir=ROOT / "runs/baseline_pooled_cnn_v2",
            seed_betadata_dir=ROOT / "runs/baseline_pooled_seed",
            data_root=args.data_root,
            pred_dir=args.pred_dir,
            seed_pred_dir=args.seed_pred_dir,
            baseline_h5ad=args.baseline_h5ad,
        )
        bd, bl_path, pd_dir = resolve_paths(run_args, sl)
        if sl.startswith("subQ"):
            baseline = pooled_baseline[pooled_baseline.obs["slice_id"].astype(str) == sl].copy()
            perts = SUBQ_PERTS
            gb = None
        else:
            baseline = load_baseline(bl_path)
            baseline.obs["slice_id"] = sl
            perts = LUNG_PERTS
            gb = pooled_baseline

        pool = load_pool(sl, args.data_root)
        pool.obs["slice_id"] = sl
        if pool.X.max() > 30:
            sc.pp.normalize_total(pool, target_sum=10000)
            sc.pp.log1p(pool)

        prep = baseline.copy()
        cmu.ensure_cluster_id(prep)
        beta_matrix, score_genes = cmu.build_beta_score_matrix(
            prep, bd, gene_filter=cmu.MICRONICHE_CLUSTER_GENES,
        )
        prep, pool = assign_pool_niches(
            prep, pool, beta_matrix, sl, per_slice_leiden=per_slice_leiden, niche_key="cnn_leiden",
        )

        for row in cmu.pathway_distinctness_ntc(pool, pathways, niche_key="cnn_leiden", cell_type="tumor"):
            row["slice"] = sl
            distinct_rows.append(row)

        for pert in perts:
            if int((pool.obs["target_gene"].astype(str) == pert).sum()) < 5:
                continue
            pred_path = resolve_pred_path(pd_dir, args.seed_pred_dir, pert)
            if pred_path is None:
                continue
            pred = pd.read_feather(pred_path)
            if "CellID" in pred.columns:
                pred = pred.set_index("CellID")

            for pathway, genes in pathways.items():
                nd = cmu.pathway_niche_deltas(
                    prep, pool, pred, pert, pathway, genes, "cnn_leiden", sl,
                    cell_type="tumor", min_ntc=args.min_ntc, min_pert=args.min_pert,
                    global_baseline=gb,
                )
                if nd.empty:
                    continue
                nd = nd.copy()
                nd["slice"] = sl
                nd["perturbation"] = pert
                nd["pathway"] = pathway
                delta_rows.append(nd)

                stats = cmu.pathway_concordance_stats(nd)
                tie = cmu.pathway_enrichment_tie_stats(nd)
                exp_sign = cmu.PERT_PATHWAY_EXPECTED_SIGN.get((pert, pathway))
                bulk_obs = float(nd["obs_pathway_delta"].mean())
                bulk_pred = float(nd["pred_pathway_delta"].mean())
                sign_match = None
                if exp_sign is not None and np.isfinite(bulk_obs) and np.isfinite(bulk_pred):
                    sign_match = bool(np.sign(bulk_obs) == exp_sign and np.sign(bulk_pred) == exp_sign)
                concord_rows.append({
                    "slice": sl,
                    "perturbation": pert,
                    "pathway": pathway,
                    "expected_sign": exp_sign,
                    "bulk_obs_delta": bulk_obs,
                    "bulk_pred_delta": bulk_pred,
                    "sign_match": sign_match,
                    **stats,
                    **tie,
                })

    distinct_df = pd.DataFrame(distinct_rows)
    delta_df = pd.concat(delta_rows, ignore_index=True) if delta_rows else pd.DataFrame()
    concord_df = pd.DataFrame(concord_rows)
    tie_df = concord_df.copy()

    distinct_df.to_csv(out_dir / f"pathway_distinctness_{args.tag}.csv", index=False)
    delta_df.to_csv(out_dir / f"pathway_niche_deltas_{args.tag}.csv", index=False)
    concord_df.to_csv(out_dir / f"pathway_concordance_{args.tag}.csv", index=False)
    tie_df.to_csv(out_dir / f"pathway_enrichment_tie_{args.tag}.csv", index=False)

    summary = {
        "tag": args.tag,
        "n_pathways": len(pathways),
        "n_distinct_tests": int(len(distinct_df)),
        "frac_pathways_significant_ntc": float(distinct_df["significant"].mean()) if not distinct_df.empty else None,
        "n_concordance_tests": int(concord_df["pearson_r"].notna().sum()) if not concord_df.empty else 0,
        "median_pathway_concordance_r": float(concord_df["pearson_r"].median()) if not concord_df.empty else None,
        "frac_sign_match_expected": float(concord_df.loc[concord_df["sign_match"].notna(), "sign_match"].mean()) if concord_df["sign_match"].notna().any() else None,
        "median_or_vs_ntc_pathway_r": float(concord_df["or_vs_ntc_pathway_r"].median()) if "or_vs_ntc_pathway_r" in concord_df else None,
        "median_or_vs_obs_pathway_r": float(concord_df["or_vs_obs_pathway_r"].median()) if "or_vs_obs_pathway_r" in concord_df else None,
    }
    (out_dir / f"pathway_microniche_overall_{args.tag}.json").write_text(json.dumps(summary, indent=2))
    _plot_all(distinct_df, delta_df, concord_df, fig_dir, args.tag)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
