#!/usr/bin/env python3
"""Pseudobulk-level statistical validation of SpaceTravLR predictions vs SPAC-seq.

For each perturbation gene P with both a predicted KO feather and an observed
cohort (cells with sgP guides), we now do the analysis CELL-TYPE-STRATIFIED to
remove composition confounding (perturbations enrich for tumor cells purely
because of fitness/selection, not transcriptional KO response).

For every (perturbation P, cell type CT) we compute, per gene g:

  predicted_delta[g]   = mean(predicted_KO[g, NTC cells of type CT])
                       - mean(baseline_imputed[g, NTC cells of type CT])
  observed_delta[g]    = mean(observed_log1pnorm[g, sgP cells of type CT])
                       - mean(observed_log1pnorm[g, NTC cells of type CT])

We then compute:
  - Spearman / Pearson(predicted, observed) per (P, CT) and pooled
  - Sign-agreement restricted to top-K predicted-effect genes
  - One-sided permutation p-value for the directionality test
  - Cosine similarity across genes
  - Specificity matrix: cor(predicted[P], observed[Q]) for all P,Q
  - Pathway-level: M1 vs M2 macrophage gene-set scores for Bcam, Cd83, Cd74
    (the antigen-presentation / macrophage-relevant perturbations)

Outputs go under results/v2/ and figures/v2/.
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


# Curated gene sets for pathway-level validation
GENE_SETS = {
    "M1_macrophage": ["Tnf", "Il1b", "Il6", "Nos2", "Cxcl9", "Cxcl10", "Cxcl11",
                       "Cxcl2", "Stat1", "H2-Aa", "H2-Ab1", "Ifit3", "Isg15",
                       "Ifi27l2a", "Rsad2", "Gbp2", "Sp100"],
    "M2_macrophage": ["Arg1", "Mrc1", "Cd163", "Il10", "Trem2", "Vegfa", "Tgfb1",
                       "Chil3", "Retnla", "Mertk", "Mmp12", "Stab1", "Ms4a4a",
                       "Ms4a7", "Ms4a6c", "Selenop", "Gpnmb", "Lgals3"],
    "MHC_class_I": ["H2-K1", "H2-D1", "B2m", "Tap1", "Tap2", "Nlrc5", "H2-Q6", "H2-Q7"],
    "MHC_class_II": ["H2-Aa", "H2-Ab1", "Cd74"],
    "T_cell_effector": ["Cd8a", "Cd8b1", "Cd3d", "Cd3e", "Gzmb", "Gzma", "Prf1",
                         "Ifng", "Nkg7", "Klrk1"],
    "T_cell_exhaustion": ["Pdcd1", "Lag3", "Tigit", "Havcr2", "Foxp3", "Ctla2a"],
    "ECM_fibroblast": ["Col1a2", "Col5a2", "Col12a1", "Col15a1", "Dcn", "Fbn1",
                        "Postn", "Tnc", "Lox", "Loxl1", "Lrrc15", "Acta2",
                        "Mmp2", "Timp3", "Bgn", "Tagln", "Igfbp4", "Igfbp5", "Igfbp7"],
    "Interferon_response": ["Ifit3", "Isg15", "Rsad2", "Irf7", "Stat1", "Gbp2",
                             "Cxcl9", "Cxcl10", "Sp100"],
    "Antigen_processing": ["B2m", "H2-K1", "H2-D1", "Tap1", "Tap2", "Nlrc5",
                            "Cd74", "H2-Aa", "H2-Ab1"],
}


def dense_genes(adata: sc.AnnData, genes: list[str]) -> pd.DataFrame:
    keep = [g for g in genes if g in adata.var_names]
    sub = adata[:, keep]
    if sparse.issparse(sub.X):
        arr = sub.X.toarray()
    else:
        arr = np.asarray(sub.X)
    return pd.DataFrame(arr, index=adata.obs_names, columns=keep)


def fit_celltype_block(pool: sc.AnnData, gene_pool: list[str]) -> dict[str, dict[str, pd.Series]]:
    """{ celltype -> { target_gene -> per-gene mean expression Series } }."""
    out: dict[str, dict[str, pd.Series]] = {}
    expr = dense_genes(pool, gene_pool)
    for ct in pool.obs.cell_type.unique():
        out[str(ct)] = {}
        ct_mask = (pool.obs.cell_type == ct).values
        for tg in pool.obs.target_gene.unique():
            mask = ct_mask & (pool.obs.target_gene == tg).values
            if mask.sum() < 5:
                continue
            out[str(ct)][str(tg)] = expr.loc[mask].mean(axis=0)
    return out


def predicted_means_per_celltype(baseline: sc.AnnData, pred_path: Path,
                                  genes: list[str]) -> dict[str, tuple[pd.Series, pd.Series]]:
    """For one perturbation: per cell type, return (baseline_mean, predicted_mean) Series across genes."""
    pred = pd.read_feather(pred_path).set_index("CellID")
    common = [g for g in genes if g in pred.columns and g in baseline.var_names]
    pred = pred[common]
    base = dense_genes(baseline, common)
    base = base.loc[pred.index]
    out: dict[str, tuple[pd.Series, pd.Series]] = {}
    ct_map = baseline.obs["cell_type"].loc[pred.index]
    for ct in ct_map.unique():
        m = (ct_map == ct).values
        if m.sum() < 5:
            continue
        out[str(ct)] = (base.loc[m].mean(axis=0), pred.loc[m].mean(axis=0))
    return out


def compute_corrs(pred_delta: pd.Series, obs_delta: pd.Series, n_perm: int = 1000,
                   seed: int = 0) -> dict[str, float]:
    common = pred_delta.index.intersection(obs_delta.index)
    p = pred_delta.loc[common].to_numpy(); o = obs_delta.loc[common].to_numpy()
    mask = np.isfinite(p) & np.isfinite(o)
    if mask.sum() < 30:
        return {"n_genes": int(mask.sum())}
    p, o = p[mask], o[mask]
    rho, sp_p = stats.spearmanr(p, o)
    r, pr_p = stats.pearsonr(p, o)
    # cosine
    cos = float(np.dot(p, o) / (np.linalg.norm(p) * np.linalg.norm(o) + 1e-12))
    # permutation: shuffle obs labels
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float32)
    n = len(o)
    for i in range(n_perm):
        idx = rng.permutation(n)
        null[i], _ = stats.spearmanr(p, o[idx])
    perm_p = float((np.abs(null) >= abs(rho)).mean())
    # top-K direction-of-effect agreement (focus on genes the model thinks change)
    abs_pred = np.abs(p)
    out = {"n_genes": int(mask.sum()), "spearman_rho": float(rho), "spearman_p": float(sp_p),
           "spearman_perm_p": perm_p, "pearson_r": float(r), "pearson_p": float(pr_p),
           "cosine": cos}
    for k in (50, 100, 200):
        if mask.sum() < k:
            continue
        topk = np.argsort(abs_pred)[-k:]
        sgn = float(np.mean(np.sign(p[topk]) == np.sign(o[topk])))
        # binomial test vs 0.5
        binom_p = float(stats.binomtest(int(round(sgn * k)), k, p=0.5,
                                          alternative="greater").pvalue)
        out[f"sign_agree_top{k}"] = sgn
        out[f"sign_agree_top{k}_binom_p"] = binom_p
    return out


def signature_score(adata: sc.AnnData, genes: list[str], cell_mask: np.ndarray) -> float:
    """Mean expression across the gene set in masked cells (mean of log1p)."""
    g = [x for x in genes if x in adata.var_names]
    if not g:
        return float("nan")
    sub = adata[cell_mask, g]
    if sparse.issparse(sub.X):
        arr = sub.X.toarray()
    else:
        arr = np.asarray(sub.X)
    return float(arr.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-h5ad",
                    default=ROOT / "runs/baseline_ntc_full/spacetravlr_prep",
                    type=Path)
    ap.add_argument("--perturbed-h5ad", default=ROOT / "data/perturbed_pool.h5ad", type=Path)
    ap.add_argument("--pred-dir", default=ROOT / "results/predictions", type=Path)
    ap.add_argument("--out-dir", default=ROOT / "results/v2", type=Path)
    ap.add_argument("--fig-dir", default=ROOT / "figures/v2", type=Path)
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline_h5ad.is_dir():
        candidates = sorted(args.baseline_h5ad.glob("*.h5ad"))
        if not candidates:
            raise SystemExit(f"No h5ad under {args.baseline_h5ad}")
        baseline_path = candidates[0]
    else:
        baseline_path = args.baseline_h5ad
    print(f"baseline (prep) h5ad: {baseline_path}")
    baseline = sc.read_h5ad(baseline_path)
    # baseline.X is log1p-normalized in prep adata; layers['imputed_count'] is MAGIC
    if "imputed_count" in baseline.layers:
        # SpaceTravLR perturbation operates on imputed_count; align here too
        baseline.X = baseline.layers["imputed_count"]
    print(f"  baseline: {baseline.shape}  ct={dict(baseline.obs.cell_type.value_counts())}")

    pool = sc.read_h5ad(args.perturbed_h5ad)
    sc.pp.normalize_total(pool, target_sum=10000)
    sc.pp.log1p(pool)
    print(f"  pool: {pool.shape}  cohorts={dict(pool.obs.target_gene.value_counts())}")

    pred_files = sorted(args.pred_dir.glob("predicted_KO_*.feather"))
    pert_genes = [p.stem.replace("predicted_KO_", "") for p in pred_files]
    eligible = [g for g in pert_genes if (pool.obs.target_gene == g).sum() >= 100]
    print(f"perturbations to evaluate (>=100 cells in pool): {eligible}")
    if not eligible:
        raise SystemExit("No eligible perturbations.")

    common_genes = sorted(set(baseline.var_names) & set(pool.var_names))
    print(f"shared genes: {len(common_genes)}")

    summary_rows = []
    pred_delta_store: dict[str, dict[str, pd.Series]] = {}  # ct -> P -> series
    obs_delta_store: dict[str, dict[str, pd.Series]] = {}

    cell_types = sorted(set(baseline.obs.cell_type.unique()) & set(pool.obs.cell_type.unique()))
    print(f"cell types: {cell_types}")

    for ct in cell_types:
        pred_delta_store[ct] = {}
        obs_delta_store[ct] = {}
    pathway_rows = []

    for gene in eligible:
        pred_path = args.pred_dir / f"predicted_KO_{gene}.feather"
        pred_ct = predicted_means_per_celltype(baseline, pred_path, common_genes)
        # observed pseudobulk per cell type
        for ct in cell_types:
            ct_mask = (pool.obs.cell_type == ct).values
            ntc_mask = ct_mask & (pool.obs.target_gene == "non-targeting").values
            pert_mask = ct_mask & (pool.obs.target_gene == gene).values
            if ntc_mask.sum() < 10 or pert_mask.sum() < 20:
                continue
            obs_ntc_mean = dense_genes(pool[ntc_mask], common_genes).mean(axis=0)
            obs_pert_mean = dense_genes(pool[pert_mask], common_genes).mean(axis=0)
            obs_delta = obs_pert_mean - obs_ntc_mean  # log1p-norm delta
            if ct not in pred_ct:
                continue
            base_mean, pred_mean = pred_ct[ct]
            # align both to the same gene index
            common = list(set(base_mean.index) & set(obs_delta.index))
            common = [g for g in common if g != gene]  # exclude self
            pred_delta = (pred_mean - base_mean).loc[common]
            obs_d = obs_delta.loc[common]
            corrs = compute_corrs(pred_delta, obs_d, n_perm=args.n_perm)
            corrs.update({"perturbation": gene, "cell_type": ct,
                            "n_ntc": int(ntc_mask.sum()),
                            "n_pert": int(pert_mask.sum())})
            summary_rows.append(corrs)
            pred_delta_store[ct][gene] = pred_delta
            obs_delta_store[ct][gene] = obs_d

            # pathway-level signature differences (observed)
            for set_name, gset in GENE_SETS.items():
                obs_score = signature_score(pool, gset, pert_mask) - signature_score(pool, gset, ntc_mask)
                # predicted signature delta on this cell type
                gset_in = [g for g in gset if g in pred_delta.index]
                pred_score = float(pred_delta.loc[gset_in].mean()) if gset_in else float("nan")
                pathway_rows.append({"perturbation": gene, "cell_type": ct,
                                      "gene_set": set_name, "n_genes_in_set": len(gset_in),
                                      "observed_delta": obs_score, "predicted_delta": pred_score})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.out_dir / "per_celltype_corr.csv", index=False)
    pathway_df = pd.DataFrame(pathway_rows)
    pathway_df.to_csv(args.out_dir / "pathway_signature.csv", index=False)

    if summary_df.empty:
        raise SystemExit("No (perturbation, cell_type) rows passed the cell-count filter.")
    # specificity matrix per cell type
    spec_rows = []
    for ct in cell_types:
        p_map = pred_delta_store.get(ct, {})
        o_map = obs_delta_store.get(ct, {})
        common_perts = sorted(set(p_map) & set(o_map))
        if len(common_perts) < 2:
            continue
        # align gene index across all
        idx = sorted(set.intersection(*[set(p_map[g].index) for g in common_perts]))
        for p_g in common_perts:
            for o_g in common_perts:
                pv = p_map[p_g].loc[idx]
                ov = o_map[o_g].loc[idx]
                mask = np.isfinite(pv) & np.isfinite(ov)
                if mask.sum() < 30:
                    continue
                rho, _ = stats.spearmanr(pv[mask], ov[mask])
                spec_rows.append({"cell_type": ct, "pred": p_g, "obs": o_g,
                                   "spearman_rho": float(rho)})
    spec_df = pd.DataFrame(spec_rows)
    spec_df.to_csv(args.out_dir / "specificity_matrix.csv", index=False)

    # diag-vs-off-diag specificity test, per cell type
    spec_test_rows = []
    for ct in cell_types:
        sub = spec_df[spec_df.cell_type == ct]
        if sub.empty:
            continue
        mat = sub.pivot(index="pred", columns="obs", values="spearman_rho")
        diag = np.array([mat.loc[p, p] for p in mat.index if p in mat.columns and not np.isnan(mat.loc[p, p])])
        off = mat.values.copy(); np.fill_diagonal(off, np.nan)
        off_flat = off[~np.isnan(off)]
        if len(diag) < 2 or len(off_flat) < 2:
            continue
        _, u_p = stats.mannwhitneyu(diag, off_flat, alternative="greater")
        spec_test_rows.append({"cell_type": ct,
                                "diag_median": float(np.median(diag)),
                                "off_diag_median": float(np.median(off_flat)),
                                "n_diag": len(diag), "n_off": len(off_flat),
                                "mw_p_one_sided": float(u_p)})
        # heatmap
        fig, ax = plt.subplots(figsize=(0.7 * len(mat) + 2, 0.7 * len(mat) + 1.5))
        vmax = max(0.001, float(np.nanmax(np.abs(mat.values))))
        im = ax.imshow(mat.astype(float).values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index)
        ax.set_xlabel("Observed cohort"); ax.set_ylabel("Predicted KO")
        ax.set_title(f"{ct}: diag={np.median(diag):.3f} off-diag={np.median(off_flat):.3f} MW p={u_p:.3g}",
                       fontsize=9)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(args.fig_dir / f"specificity_{ct}.png", dpi=160)
        plt.close(fig)
    pd.DataFrame(spec_test_rows).to_csv(args.out_dir / "specificity_test.csv", index=False)

    # overall summary
    overall = {
        "n_rows_per_celltype": int(len(summary_df)),
        "median_spearman_rho_overall": float(summary_df.spearman_rho.median()),
        "n_positive_corr": int((summary_df.spearman_rho > 0).sum()),
        "n_significant_perm_p05": int((summary_df.spearman_perm_p < 0.05).sum()),
        "by_cell_type": (
            summary_df.groupby("cell_type")["spearman_rho"]
            .agg(["median", "mean", "count"]).round(4).to_dict()
        ),
        "specificity_by_celltype": pd.DataFrame(spec_test_rows).to_dict(orient="records"),
    }
    # add sign-agreement summary
    for k in (50, 100, 200):
        col = f"sign_agree_top{k}"
        if col in summary_df:
            overall[f"median_{col}"] = float(summary_df[col].median())
            overall[f"frac_pert_celltype_top{k}_above_0.55"] = float((summary_df[col] >= 0.55).mean())

    (args.out_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2, default=str))
    print(json.dumps(overall, indent=2, default=str))
    print("\nPer-celltype rows:")
    print(summary_df.to_string(index=False))

    # Quick pathway visual
    if not pathway_df.empty:
        pivot = pathway_df.pivot_table(index=["gene_set"], columns=["perturbation", "cell_type"],
                                          values="observed_delta")
        pivot.to_csv(args.out_dir / "pathway_observed_pivot.csv")
        ppred = pathway_df.pivot_table(index=["gene_set"], columns=["perturbation", "cell_type"],
                                          values="predicted_delta")
        ppred.to_csv(args.out_dir / "pathway_predicted_pivot.csv")


if __name__ == "__main__":
    main()
