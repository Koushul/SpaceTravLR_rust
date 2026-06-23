#!/usr/bin/env python3
"""Statistically compare SpaceTravLR predicted KO effects vs SPAC-seq observed
perturbation transcriptomes.

For each perturbation gene P with both a predicted KO feather and an observed
cohort (cells with sgP guides), we compute:

  predicted_log2FC[g]  = log2( mean_pred_KO[g, NTC cells] / mean_baseline[g, NTC cells] )
  observed_log2FC[g]   = log2( mean_obs[g, sgP cells]     / mean_obs[g, NTC cells] )

Then we run several tests across genes g:

  - Spearman / Pearson correlation between predicted and observed log2FC
  - Sign agreement fraction (excluding small effects)
  - Specificity matrix: cross-correlations cor(predicted[P], observed[Q])
    Diagonal P==Q should dominate the off-diagonal under a true model.
  - Permutation null: shuffle gene labels in observed and recompute correlation
    to derive an empirical p-value for the diagonal correlation.
  - Per cell type breakdowns (matched within fibroblast / immune / myeloid / tumor).

Outputs (under results/):
  comparison_summary.csv     per-perturbation correlation + p-values
  specificity_matrix.csv     P x Q Spearman matrix
  per_celltype_summary.csv   per (perturbation, cell_type) statistics
  observed_predicted_log2fc/<P>.parquet  per-gene long table for inspection
  figures/*.png              scatter + heatmap visuals
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EPS = 1e-3


def get_dense_genes(adata: sc.AnnData, genes: list[str]) -> pd.DataFrame:
    """Return per-cell DataFrame with the requested genes (raw values)."""
    keep = [g for g in genes if g in adata.var_names]
    sub = adata[:, keep]
    if sparse.issparse(sub.X):
        arr = sub.X.toarray()
    else:
        arr = np.asarray(sub.X)
    return pd.DataFrame(arr, index=adata.obs_names, columns=keep)


def log2fc(mean_p: pd.Series, mean_c: pd.Series, eps: float = EPS) -> pd.Series:
    return np.log2((mean_p + eps) / (mean_c + eps))


def compute_observed_log2fc(
    pool: sc.AnnData,
    target_gene: str,
    genes: list[str],
    per_celltype: bool = False,
) -> pd.Series | pd.DataFrame:
    """Observed log2FC across genes between sgP cells and NTC cells.

    If per_celltype, return a DataFrame indexed by gene, columns are cell types
    where both NTC and sgP are present.
    """
    ntc_mask = (pool.obs.target_gene == "non-targeting").values
    pert_mask = (pool.obs.target_gene == target_gene).values
    expr = get_dense_genes(pool, genes)
    if per_celltype:
        out = {}
        for ct in pool.obs.cell_type.unique():
            ct_mask = (pool.obs.cell_type == ct).values
            n_ntc = (ct_mask & ntc_mask).sum()
            n_pert = (ct_mask & pert_mask).sum()
            if n_ntc < 5 or n_pert < 5:
                continue
            mc = expr.loc[ct_mask & ntc_mask].mean(axis=0)
            mp = expr.loc[ct_mask & pert_mask].mean(axis=0)
            out[ct] = log2fc(mp, mc)
        return pd.DataFrame(out)
    mc = expr.loc[ntc_mask].mean(axis=0)
    mp = expr.loc[pert_mask].mean(axis=0)
    return log2fc(mp, mc)


def compute_predicted_log2fc(
    baseline: sc.AnnData,
    pred_feather: Path,
    genes: list[str],
    per_celltype: bool = False,
) -> pd.Series | pd.DataFrame:
    """Predicted log2FC across genes from in-silico KO."""
    pred = pd.read_feather(pred_feather)
    pred = pred.set_index("CellID")
    common = [g for g in genes if g in pred.columns]
    pred = pred[common]
    base = get_dense_genes(baseline, common)
    base = base.loc[pred.index]
    if per_celltype:
        ct_map = baseline.obs["cell_type"].loc[pred.index]
        out = {}
        for ct in ct_map.unique():
            mask = (ct_map == ct).values
            if mask.sum() < 5:
                continue
            mc = base.loc[mask].mean(axis=0)
            mp = pred.loc[mask].mean(axis=0)
            out[ct] = log2fc(mp, mc)
        return pd.DataFrame(out)
    return log2fc(pred.mean(axis=0), base.mean(axis=0))


def correlation_pvalue_permutation(
    pred_vec: pd.Series, obs_vec: pd.Series, n_perm: int = 2000, seed: int = 0,
) -> tuple[float, float, float]:
    """Spearman correlation + permutation p-value (shuffle observed labels)."""
    common = pred_vec.index.intersection(obs_vec.index)
    p = pred_vec.loc[common].to_numpy()
    o = obs_vec.loc[common].to_numpy()
    mask = np.isfinite(p) & np.isfinite(o)
    if mask.sum() < 20:
        return (np.nan, np.nan, np.nan)
    p, o = p[mask], o[mask]
    rho, _ = stats.spearmanr(p, o)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float32)
    n = len(o)
    for i in range(n_perm):
        idx = rng.permutation(n)
        null[i], _ = stats.spearmanr(p, o[idx])
    pval = float((np.abs(null) >= abs(rho)).mean())
    return float(rho), pval, float(np.nanmedian(null))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-h5ad",
                    default=ROOT / "runs/baseline_ntc_seed/spacetravlr_prep",
                    type=Path,
                    help="Dir containing the SpaceTravLR-preprocessed baseline h5ad.")
    ap.add_argument("--perturbed-h5ad", default=ROOT / "data/perturbed_pool.h5ad", type=Path)
    ap.add_argument("--pred-dir", default=ROOT / "results/predictions", type=Path)
    ap.add_argument("--out-dir", default=ROOT / "results", type=Path)
    ap.add_argument("--fig-dir", default=ROOT / "figures", type=Path)
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline_h5ad.is_dir():
        baseline_path = sorted(args.baseline_h5ad.glob("*.h5ad"))[0]
    else:
        baseline_path = args.baseline_h5ad
    print(f"baseline (prep) h5ad: {baseline_path}")
    baseline = sc.read_h5ad(baseline_path)
    # Use the imputed_count layer that perturb operates on
    if "imputed_count" in baseline.layers:
        baseline.X = baseline.layers["imputed_count"]
    print(f"  baseline: {baseline.shape}, cell_types={dict(baseline.obs.cell_type.value_counts())}")

    print(f"perturbed_pool h5ad: {args.perturbed_h5ad}")
    pool = sc.read_h5ad(args.perturbed_h5ad)
    # match preprocessing: normalize_total + log1p (no MAGIC) so observed counts
    # are on a comparable additive log scale; the GRN-predicted values live on a
    # MAGIC-smoothed log1p scale, but log2FC ratios are largely scale-invariant.
    sc.pp.normalize_total(pool, target_sum=10000)
    sc.pp.log1p(pool)
    print(f"  pool: {pool.shape}, cohorts={dict(pool.obs.target_gene.value_counts())}")

    pred_files = sorted(args.pred_dir.glob("predicted_KO_*.feather"))
    pert_genes = [p.stem.replace("predicted_KO_", "") for p in pred_files]
    print(f"perturbations to evaluate: {pert_genes}")

    # gene panel: intersection of genes present in both baseline (var_names) and
    # pool, excluding the perturbed-gene-of-interest itself.
    common_genes = sorted(set(baseline.var_names) & set(pool.var_names))
    print(f"genes shared baseline ∩ pool: {len(common_genes)}")

    summary_rows = []
    per_ct_rows = []
    pred_logfc_store = {}
    obs_logfc_store = {}

    for gene in pert_genes:
        pool_count_pert = int((pool.obs.target_gene == gene).sum())
        if pool_count_pert < 20:
            print(f"  skip {gene}: only {pool_count_pert} cells in pool")
            continue
        # exclude the perturbed gene itself from the correlation panel so we
        # don't trivially boost correlation via the on-target near-zero value
        panel = [g for g in common_genes if g != gene]
        try:
            pred = compute_predicted_log2fc(baseline, args.pred_dir / f"predicted_KO_{gene}.feather", panel)
            obs = compute_observed_log2fc(pool, gene, panel)
        except KeyError as e:
            print(f"  warn {gene}: {e}; skipping")
            continue
        pred_logfc_store[gene] = pred
        obs_logfc_store[gene] = obs
        rho, pval, null_med = correlation_pvalue_permutation(pred, obs, n_perm=args.n_perm)
        pearson_r, pearson_p = stats.pearsonr(pred, obs)
        # sign agreement on genes with |obs|>thresh
        thr = 0.1
        mask = (np.abs(obs) >= thr) & np.isfinite(pred) & np.isfinite(obs)
        sign_agree = float((np.sign(pred[mask]) == np.sign(obs[mask])).mean()) if mask.sum() >= 10 else np.nan
        summary_rows.append({
            "perturbation": gene,
            "n_pert_cells": pool_count_pert,
            "n_genes_evaluated": int(np.isfinite(pred & np.isfinite(obs)).sum() if False else mask.sum() + int(mask.sum() == 0)),
            "spearman_rho": rho,
            "spearman_perm_p": pval,
            "spearman_null_median": null_med,
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "sign_agreement": sign_agree,
            "n_sign_genes": int(mask.sum()),
        })

        # per-cell-type
        pred_ct = compute_predicted_log2fc(baseline, args.pred_dir / f"predicted_KO_{gene}.feather",
                                           panel, per_celltype=True)
        obs_ct = compute_observed_log2fc(pool, gene, panel, per_celltype=True)
        common_ct = [c for c in pred_ct.columns if c in obs_ct.columns]
        for ct in common_ct:
            r, p = stats.spearmanr(pred_ct[ct], obs_ct[ct])
            per_ct_rows.append({
                "perturbation": gene,
                "cell_type": ct,
                "spearman_rho": float(r),
                "spearman_p": float(p),
                "n_genes": int(pred_ct[ct].dropna().shape[0]),
            })

        # save per-gene tables
        comp = pd.DataFrame({"gene": panel, "predicted_log2fc": pred.values, "observed_log2fc": obs.values})
        comp.to_parquet(args.out_dir / f"observed_predicted_log2fc_{gene}.parquet", index=False)

        # scatter plot
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(pred, obs, s=4, alpha=0.4, color="#1f77b4")
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.axvline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_xlabel(f"Predicted log2FC (in-silico {gene} KO)")
        ax.set_ylabel(f"Observed log2FC (sg{gene} vs sgNTC)")
        ax.set_title(
            f"{gene} | Spearman ρ={rho:.3f} (perm p={pval:.3g}) | n_obs_cells={pool_count_pert}",
            fontsize=10,
        )
        # annotate strongest disagreement and agreement
        rank_score = (pred.rank() + obs.rank()).values
        top = np.argsort(np.abs(pred.values * obs.values))[-8:][::-1]
        for i in top:
            ax.annotate(panel[i], (pred.values[i], obs.values[i]), fontsize=7, alpha=0.7)
        fig.tight_layout()
        fig.savefig(args.fig_dir / f"scatter_pred_vs_obs_{gene}.png", dpi=160)
        plt.close(fig)
        print(f"  {gene}: ρ={rho:.3f}  perm p={pval:.3g}  sign agree={sign_agree}  (n={pool_count_pert})")

    # specificity matrix
    print("\nComputing specificity matrix ...")
    rows = []
    pert_list = list(pred_logfc_store.keys())
    common_genes_eval = sorted(
        set.intersection(*[set(s.index) for s in pred_logfc_store.values()])
        & set.intersection(*[set(s.index) for s in obs_logfc_store.values()])
    )
    mat = pd.DataFrame(index=pert_list, columns=pert_list, dtype=float)
    for p in pert_list:
        pred_v = pred_logfc_store[p].loc[common_genes_eval]
        for q in pert_list:
            obs_v = obs_logfc_store[q].loc[common_genes_eval]
            mask = np.isfinite(pred_v) & np.isfinite(obs_v)
            if mask.sum() < 20:
                continue
            r, _ = stats.spearmanr(pred_v[mask], obs_v[mask])
            mat.loc[p, q] = float(r)
    mat.to_csv(args.out_dir / "specificity_matrix.csv")

    # diagonal vs off-diagonal hypothesis test (one-sided)
    diag = np.array([mat.loc[p, p] for p in pert_list if not np.isnan(mat.loc[p, p])])
    off = mat.values.copy()
    np.fill_diagonal(off, np.nan)
    off_flat = off[~np.isnan(off)]
    if len(diag) >= 3 and len(off_flat) >= 3:
        u_stat, u_p = stats.mannwhitneyu(diag, off_flat, alternative="greater")
        specificity_text = (
            f"Diagonal (n={len(diag)}) median ρ={np.median(diag):.3f} vs "
            f"off-diagonal (n={len(off_flat)}) median ρ={np.median(off_flat):.3f} | "
            f"one-sided Mann–Whitney p={u_p:.3g}"
        )
    else:
        u_p = np.nan
        specificity_text = f"insufficient data for diag/off test (diag={len(diag)})"

    # plot specificity heatmap
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(pert_list)), max(4, 0.4 * len(pert_list))))
    vmax = float(np.nanmax(np.abs(mat.values)))
    im = ax.imshow(mat.astype(float).values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pert_list))); ax.set_xticklabels(pert_list, rotation=45, ha="right")
    ax.set_yticks(range(len(pert_list))); ax.set_yticklabels(pert_list)
    ax.set_xlabel("Observed cohort (sg{Q})")
    ax.set_ylabel("Predicted KO (in-silico {P})")
    ax.set_title(f"Spearman ρ(predicted P, observed Q)\n{specificity_text}", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(args.fig_dir / "specificity_matrix.png", dpi=160)
    plt.close(fig)

    # write summary
    summary_df = pd.DataFrame(summary_rows).sort_values("spearman_rho", ascending=False)
    summary_df.to_csv(args.out_dir / "comparison_summary.csv", index=False)
    pd.DataFrame(per_ct_rows).to_csv(args.out_dir / "per_celltype_summary.csv", index=False)

    overall = {
        "n_perturbations_evaluated": int(len(summary_rows)),
        "median_spearman_rho": float(summary_df.spearman_rho.median()) if len(summary_df) else None,
        "n_positive_corr": int((summary_df.spearman_rho > 0).sum()),
        "n_significant_perm_p05": int((summary_df.spearman_perm_p < 0.05).sum()),
        "median_sign_agreement": float(summary_df.sign_agreement.median()) if len(summary_df) else None,
        "specificity_diag_median": float(np.median(diag)) if len(diag) else None,
        "specificity_off_diag_median": float(np.median(off_flat)) if len(off_flat) else None,
        "specificity_mw_p_one_sided": float(u_p) if not np.isnan(u_p) else None,
    }
    (args.out_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2))
    print("\n=== OVERALL ===")
    print(json.dumps(overall, indent=2))
    print("\nDetailed:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
