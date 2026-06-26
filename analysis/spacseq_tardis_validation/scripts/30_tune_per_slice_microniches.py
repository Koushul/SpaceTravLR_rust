#!/usr/bin/env python3
"""Per-slice Leiden hyperparameter sweep for CNN β-microniche enrichment."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _py_boot import ensure_boot

ensure_boot()

import numpy as np
import pandas as pd
import scanpy as sc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

import cnn_microniche_utils as cmu

_spec23 = importlib.util.spec_from_file_location("sp23", HERE / "23_cnn_microniche_enrichment.py")
_sp23 = importlib.util.module_from_spec(_spec23)
_spec23.loader.exec_module(_sp23)

_spec09 = importlib.util.spec_from_file_location("sp09", HERE / "09_spatial_validation.py")
_sp09 = importlib.util.module_from_spec(_spec09)
_spec09.loader.exec_module(_sp09)
load_baseline = _sp09.load_baseline
load_pool = _sp09.load_pool

SUBQ_SLICES = _sp23.SUBQ_SLICES
LUNG_SLICE = _sp23.LUNG_SLICE
SUBQ_PERTS = _sp23.SUBQ_PERTS
LUNG_PERTS = _sp23.LUNG_PERTS
resolve_paths = _sp23.resolve_paths
run_slice_enrichment = _sp23.run_slice_enrichment
propagate_niche_labels = _sp23.propagate_niche_labels


class SliceContext:
    def __init__(
        self,
        slice_id: str,
        prep: sc.AnnData,
        pool: sc.AnnData,
        beta_matrix: np.ndarray,
        score_genes: list[str],
        perts: list[str],
        global_baseline: sc.AnnData | None,
        pred_dir: Path,
        seed_pred_dir: Path,
        data_root: Path,
        betadata_dir: Path,
    ):
        self.slice_id = slice_id
        self.prep = prep
        self.pool = pool
        self.beta_matrix = beta_matrix
        self.score_genes = score_genes
        self.perts = perts
        self.global_baseline = global_baseline
        self.pred_dir = pred_dir
        self.seed_pred_dir = seed_pred_dir
        self.data_root = data_root
        self.betadata_dir = betadata_dir
        self.tumor_mask = prep.obs["cell_type"].astype(str) == "tumor"
        self.tumor_prep = prep[self.tumor_mask].copy()
        self.tumor_scores = beta_matrix[np.where(self.tumor_mask.values)[0]]


def load_slice_context(
    slice_id: str,
    betadata_dir: Path,
    pred_dir: Path,
    seed_pred_dir: Path,
    data_root: Path,
    baseline_h5ad: Path,
    pooled_baseline: sc.AnnData,
) -> SliceContext:
    args = argparse.Namespace(
        betadata_dir=betadata_dir,
        seed_betadata_dir=ROOT / "runs/baseline_pooled_seed",
        pred_dir=pred_dir,
        seed_pred_dir=seed_pred_dir,
        data_root=data_root,
        baseline_h5ad=baseline_h5ad,
    )
    bd, bl_path, pd_dir = resolve_paths(args, slice_id)
    if slice_id.startswith("subQ"):
        baseline = pooled_baseline[pooled_baseline.obs["slice_id"].astype(str) == slice_id].copy()
        perts, gb = SUBQ_PERTS, None
    else:
        baseline = load_baseline(bl_path)
        baseline.obs["slice_id"] = slice_id
        perts, gb = LUNG_PERTS, pooled_baseline

    pool = load_pool(slice_id, data_root)
    pool.obs["slice_id"] = slice_id
    prep = baseline.copy()
    cmu.ensure_cluster_id(prep)
    beta_matrix, score_genes = cmu.build_beta_score_matrix(
        prep, bd, gene_filter=cmu.MICRONICHE_CLUSTER_GENES,
    )
    return SliceContext(
        slice_id, prep, pool, beta_matrix, score_genes, perts, gb, pd_dir, seed_pred_dir, data_root, bd,
    )


def score_slice_kw(ctx: SliceContext, leiden_kw: dict) -> tuple[float, float, pd.DataFrame]:
    kw = {**cmu.DEFAULT_LEIDEN_KW, **leiden_kw}
    sub_labels = cmu.leiden_microniches(ctx.tumor_prep, ctx.tumor_scores, **kw)
    labels = pd.Series(index=ctx.prep.obs_names, dtype=str)
    labels.loc[ctx.tumor_prep.obs_names] = sub_labels.astype(str).radd(f"{ctx.slice_id}|tumor|").values
    labels = labels.fillna("unassigned")

    prep, pool = propagate_niche_labels(ctx.prep, ctx.pool, ctx.slice_id, labels, "cnn_leiden")
    _, corr, _, _ = run_slice_enrichment(
        ctx.slice_id, ctx.perts, ctx.data_root, prep, ctx.betadata_dir, ctx.pred_dir, "per_slice_tune",
        global_baseline=ctx.global_baseline, fallback_pred_dir=ctx.seed_pred_dir,
        leiden_kw=kw, prep=prep, pool=pool,
        score_genes=ctx.score_genes, beta_matrix=ctx.beta_matrix,
    )
    if corr.empty:
        return float("nan"), float("nan"), corr
    valid = corr["pearson_r"].dropna()
    med = float(valid.median()) if len(valid) else float("nan")
    mean = float(valid.mean()) if len(valid) else float("nan")
    return med, mean, corr


def sweep_slice(ctx: SliceContext, grid: list[dict]) -> tuple[dict, pd.DataFrame]:
    rows = []
    best = {"composite_score": float("-inf")}
    for kw in grid:
        med, mean, corr = score_slice_kw(ctx, kw)
        valid = corr["pearson_r"].dropna() if not corr.empty else pd.Series(dtype=float)
        n_valid = int(len(valid))
        n_niches_med = float(corr["n_niches"].median()) if not corr.empty else 0.0
        row = {
            "slice": ctx.slice_id,
            "resolution": kw["resolution"],
            "spatial_weight": kw["spatial_weight"],
            "n_pcs": kw["n_pcs"],
            "median_pearson_r": med,
            "mean_pearson_r": mean,
            "n_tests": int(len(corr)),
            "n_valid_corr": n_valid,
            "median_n_niches": n_niches_med,
        }
        rows.append(row)
        print(
            f"  {ctx.slice_id} r={kw['resolution']} sw={kw['spatial_weight']} npc={kw['n_pcs']}"
            f" → med r={med:+.3f} niches={n_niches_med:.1f}",
            flush=True,
        )
        score = med if not pd.isna(med) else float("-inf")
        min_r = float(valid.min()) if len(valid) else float("-inf")
        n_missing = int(len(corr) - len(valid)) if not corr.empty else len(ctx.perts)
        composite = min_r - 0.5 * n_missing
        if composite > best.get("composite_score", float("-inf")):
            best = {
                **row,
                "leiden_kw": {**cmu.DEFAULT_LEIDEN_KW, **kw},
                "min_pearson_r": min_r,
                "composite_score": composite,
            }

    return best, pd.DataFrame(rows)


def build_grid(
    resolutions: list[float],
    spatial_weights: list[float],
    n_pcs_list: list[int],
) -> list[dict]:
    grid = []
    for res, sw, npc in itertools.product(resolutions, spatial_weights, n_pcs_list):
        grid.append({
            "resolution": res,
            "spatial_weight": sw,
            "n_pcs": npc,
            "min_cells": cmu.DEFAULT_LEIDEN_KW["min_cells"],
        })
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--betadata-dir", type=Path, default=ROOT / "runs/baseline_pooled_cnn_v2")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "results/predictions_tuned")
    ap.add_argument("--seed-pred-dir", type=Path, default=ROOT / "results/predictions_tuned")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--baseline-h5ad", type=Path, default=ROOT / "data/pooled/baseline_ntc.h5ad")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/cnn_enrichment/tune")
    ap.add_argument("--slices", nargs="+", default=SUBQ_SLICES + [LUNG_SLICE])
    ap.add_argument("--coarse", action="store_true", help="Smaller grid for quick iteration")
    args = ap.parse_args()

    if args.coarse:
        resolutions = [0.55, 0.65, 0.75, 0.85]
        spatial_weights = [0.35, 0.45, 0.55]
        n_pcs_list = [10, 14, 18]
    else:
        resolutions = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        spatial_weights = [0.25, 0.35, 0.4, 0.45, 0.55]
        n_pcs_list = [8, 10, 12, 14, 16, 18]

    grid = build_grid(resolutions, spatial_weights, n_pcs_list)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pooled baseline...", flush=True)
    pooled = load_baseline(args.baseline_h5ad)
    if "slice_id" not in pooled.obs.columns:
        pooled.obs["slice_id"] = pooled.obs_names.str.split("@").str[-1]

    per_slice_best: dict[str, dict] = {}
    all_rows: list[pd.DataFrame] = []

    for sl in args.slices:
        print(f"\n=== Loading {sl} ===", flush=True)
        ctx = load_slice_context(
            sl, args.betadata_dir, args.pred_dir, args.seed_pred_dir,
            args.data_root, args.baseline_h5ad, pooled,
        )
        print(f"=== Sweeping {sl} ({len(grid)} combos) ===", flush=True)
        best, sweep_df = sweep_slice(ctx, grid)
        per_slice_best[sl] = best
        all_rows.append(sweep_df)

    sweep_all = pd.concat(all_rows, ignore_index=True)
    sweep_all.to_csv(args.out_dir / "per_slice_leiden_sweep.csv", index=False)

    config = {
        "description": "Per-slice Leiden hyperparameters maximizing median obs vs pred enrichment r",
        "slices": {
            sl: per_slice_best[sl].get("leiden_kw", cmu.DEFAULT_LEIDEN_KW)
            for sl in args.slices
        },
        "summary": {
            sl: {
                k: per_slice_best[sl][k]
                for k in (
                    "median_pearson_r", "mean_pearson_r", "n_tests",
                    "n_valid_corr", "median_n_niches",
                    "resolution", "spatial_weight", "n_pcs",
                )
                if k in per_slice_best[sl]
            }
            for sl in args.slices
        },
    }
    out_path = args.out_dir / "per_slice_leiden.json"
    out_path.write_text(json.dumps(config, indent=2))
    print(f"\nWrote {out_path}", flush=True)

    medians = [per_slice_best[sl]["median_pearson_r"] for sl in args.slices]
    valid = [m for m in medians if not pd.isna(m)]
    print(f"Cohort median of per-slice medians: {pd.Series(valid).median():+.3f}", flush=True)


if __name__ == "__main__":
    main()
