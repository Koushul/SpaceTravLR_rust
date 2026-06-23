#!/usr/bin/env python3
"""Final, report-quality validation of SpaceTravLR predictions vs SPAC-seq.

Produces a single multi-panel figure plus tidy tables summarizing how well
SpaceTravLR — trained on SPAC-seq sgNTC ("non-perturb") cells only — predicts
the transcriptional consequences of CRISPR knockouts observed in the matched
SPAC-seq perturbation cohorts.

Headline metrics (per perturbation × cell type):
  - Pearson r (predicted_delta, observed_delta) and permutation p
  - Cosine similarity
  - Top-K sign-agreement (binomial test vs 0.5)
  - On-target validation: predicted log2FC of the perturbed gene itself

A specificity heatmap (predicted P vs observed Q for all P, Q) shows the
diagonal-dominance test (Mann–Whitney one-sided).

This script can run on either the seed-mode or CNN-mode training output;
pass --baseline-h5ad to point at the chosen prep h5ad and --pred-dir to the
matching prediction folder.
"""

from __future__ import annotations

import argparse
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


GENE_SETS = {
    "M1/inflam macrophage": ["Tnf", "Il1b", "Il6", "Nos2", "Cxcl9", "Cxcl10",
                              "Cxcl11", "Cxcl2", "Stat1", "H2-Aa", "H2-Ab1",
                              "Ifit3", "Isg15", "Rsad2", "Gbp2", "Sp100"],
    "M2/suppressive macrophage": ["Arg1", "Mrc1", "Cd163", "Il10", "Trem2",
                                   "Vegfa", "Tgfb1", "Chil3", "Retnla", "Mertk",
                                   "Mmp12", "Stab1", "Ms4a4a", "Ms4a7", "Selenop"],
    "Antigen processing (MHC-I)": ["H2-K1", "H2-D1", "B2m", "Tap1", "Tap2", "Nlrc5"],
    "Antigen presentation (MHC-II)": ["H2-Aa", "H2-Ab1", "Cd74"],
    "T-cell effector": ["Cd8a", "Cd3d", "Cd3e", "Gzmb", "Prf1", "Ifng", "Nkg7"],
    "T-cell exhaustion / Treg": ["Pdcd1", "Lag3", "Tigit", "Havcr2", "Foxp3"],
    "ECM/fibroblast": ["Col1a2", "Col5a2", "Col12a1", "Dcn", "Fbn1", "Postn",
                        "Tnc", "Lox", "Loxl1", "Acta2", "Mmp2", "Bgn"],
    "Interferon response": ["Ifit3", "Isg15", "Rsad2", "Stat1", "Gbp2",
                              "Cxcl9", "Cxcl10", "Sp100", "Ifi27l2a"],
}


def dense(adata: sc.AnnData, genes: list[str]) -> pd.DataFrame:
    keep = [g for g in genes if g in adata.var_names]
    sub = adata[:, keep]
    arr = sub.X.toarray() if sparse.issparse(sub.X) else np.asarray(sub.X)
    return pd.DataFrame(arr, index=adata.obs_names, columns=keep)


def per_celltype_predicted_delta(baseline, pred_path, genes):
    pred = pd.read_feather(pred_path).set_index("CellID")
    common = [g for g in genes if g in pred.columns and g in baseline.var_names]
    pred = pred[common]
    base = dense(baseline, common).loc[pred.index]
    ct = baseline.obs["cell_type"].loc[pred.index]
    out = {}
    for c in ct.unique():
        m = (ct == c).values
        if m.sum() < 5:
            continue
        out[c] = (pred[m].mean(0) - base[m].mean(0))
    return out


def stratified_observed_delta(pool, perturb_gene, genes):
    common = [g for g in genes if g in pool.var_names]
    expr = dense(pool, common)
    out = {}
    for c in pool.obs.cell_type.unique():
        ct_mask = (pool.obs.cell_type == c).values
        ntc = ct_mask & (pool.obs.target_gene == "non-targeting").values
        per = ct_mask & (pool.obs.target_gene == perturb_gene).values
        if ntc.sum() < 10 or per.sum() < 20:
            continue
        out[c] = (expr.loc[per].mean(0) - expr.loc[ntc].mean(0))
    return out


def compute_metrics(p, o, n_perm=2000, seed=0):
    common = p.index.intersection(o.index)
    p, o = p.loc[common].to_numpy(), o.loc[common].to_numpy()
    mask = np.isfinite(p) & np.isfinite(o)
    if mask.sum() < 30:
        return None
    p, o = p[mask], o[mask]
    rho, _ = stats.spearmanr(p, o)
    r, _ = stats.pearsonr(p, o)
    cos = float(p @ o / (np.linalg.norm(p) * np.linalg.norm(o) + 1e-12))
    rng = np.random.default_rng(seed)
    n = len(o)
    null = np.array([stats.pearsonr(p, o[rng.permutation(n)])[0] for _ in range(n_perm)])
    pval = float((np.abs(null) >= abs(r)).mean())
    out = {"n_genes": int(mask.sum()), "spearman_rho": float(rho),
           "pearson_r": float(r), "pearson_perm_p": pval, "cosine": cos}
    for K in (25, 50, 100):
        if mask.sum() < K:
            continue
        topk = np.argsort(np.abs(p))[-K:]
        agree = int((np.sign(p[topk]) == np.sign(o[topk])).sum())
        binom_p = float(stats.binomtest(agree, K, p=0.5, alternative="greater").pvalue)
        out[f"top{K}_sign_agree"] = agree / K
        out[f"top{K}_binom_p"] = binom_p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-h5ad", type=Path, required=True,
                    help="prep h5ad (e.g. runs/baseline_ntc_seed/spacetravlr_prep/*.h5ad)")
    ap.add_argument("--pred-dir", type=Path, required=True,
                    help="directory of predicted_KO_<gene>.feather files")
    ap.add_argument("--perturbed-h5ad", type=Path, default=ROOT / "data/perturbed_pool.h5ad")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, required=True)
    ap.add_argument("--tag", default="seed", help="Output filename suffix.")
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline_h5ad.is_dir():
        cands = sorted(args.baseline_h5ad.glob("*.h5ad"))
        if not cands:
            raise SystemExit(f"No h5ad in {args.baseline_h5ad}")
        baseline_path = cands[0]
    else:
        baseline_path = args.baseline_h5ad
    print(f"baseline: {baseline_path}")
    baseline = sc.read_h5ad(baseline_path)
    if "imputed_count" in baseline.layers:
        baseline.X = baseline.layers["imputed_count"]
    pool = sc.read_h5ad(args.perturbed_h5ad)
    sc.pp.normalize_total(pool, target_sum=10000)
    sc.pp.log1p(pool)
    common_genes = sorted(set(baseline.var_names) & set(pool.var_names))
    print(f"shared genes: {len(common_genes)}")

    pred_files = sorted(args.pred_dir.glob("predicted_KO_*.feather"))
    pert_genes = [p.stem.replace("predicted_KO_", "") for p in pred_files]
    eligible = [g for g in pert_genes if (pool.obs.target_gene == g).sum() >= 100]
    print(f"eligible perturbations: {eligible}")

    pred_store = {}
    obs_store = {}
    rows = []
    on_target = []
    for gene in eligible:
        pred_path = args.pred_dir / f"predicted_KO_{gene}.feather"
        pred_ct = per_celltype_predicted_delta(baseline, pred_path, common_genes)
        obs_ct = stratified_observed_delta(pool, gene, common_genes)
        for c, p_delta in pred_ct.items():
            if c not in obs_ct:
                continue
            panel = [g for g in p_delta.index if g != gene]
            metrics = compute_metrics(p_delta.loc[panel], obs_ct[c].loc[panel],
                                       n_perm=args.n_perm)
            if metrics is None:
                continue
            metrics.update({"perturbation": gene, "cell_type": c,
                             "n_pert_in_pool": int((pool.obs.target_gene == gene).sum())})
            rows.append(metrics)
            pred_store[(gene, c)] = p_delta
            obs_store[(gene, c)] = obs_ct[c]
        # on-target
        for c, p_delta in pred_ct.items():
            if c not in obs_ct:
                continue
            if gene in p_delta.index and gene in obs_ct[c].index:
                on_target.append({"perturbation": gene, "cell_type": c,
                                    "predicted_delta_on_target": float(p_delta[gene]),
                                    "observed_delta_on_target": float(obs_ct[c][gene])})

    summary = pd.DataFrame(rows).sort_values(["perturbation", "cell_type"])
    summary.to_csv(args.out_dir / f"per_celltype_corr_{args.tag}.csv", index=False)
    on_target_df = pd.DataFrame(on_target).sort_values(["perturbation", "cell_type"])
    on_target_df.to_csv(args.out_dir / f"on_target_{args.tag}.csv", index=False)

    # ---- specificity matrix per cell type ----
    cell_types = sorted({c for (_, c) in pred_store.keys()})
    spec_rows = []
    for c in cell_types:
        pert_in_ct = sorted({g for (g, cc) in pred_store if cc == c} & {g for (g, cc) in obs_store if cc == c})
        if len(pert_in_ct) < 3:
            continue
        idx = sorted(set.intersection(*[set(pred_store[(g, c)].index) for g in pert_in_ct]))
        mat = pd.DataFrame(index=pert_in_ct, columns=pert_in_ct, dtype=float)
        for pg in pert_in_ct:
            pv = pred_store[(pg, c)].loc[idx]
            for og in pert_in_ct:
                ov = obs_store[(og, c)].loc[idx]
                m = np.isfinite(pv) & np.isfinite(ov)
                if m.sum() < 30:
                    continue
                r_p, _ = stats.pearsonr(pv[m], ov[m])
                mat.loc[pg, og] = float(r_p)
        diag = np.array([mat.loc[p, p] for p in pert_in_ct if not np.isnan(mat.loc[p, p])])
        off = mat.values.copy()
        np.fill_diagonal(off, np.nan)
        off_flat = off[~np.isnan(off)]
        if len(diag) and len(off_flat):
            _, u_p = stats.mannwhitneyu(diag, off_flat, alternative="greater")
        else:
            u_p = np.nan
        spec_rows.append({"cell_type": c, "diag_median_r": float(np.median(diag)),
                           "off_diag_median_r": float(np.median(off_flat)),
                           "n_pert": len(pert_in_ct),
                           "diag_off_mw_p_onesided": float(u_p)})
        mat.to_csv(args.out_dir / f"specificity_matrix_{c}_{args.tag}.csv")
    spec_df = pd.DataFrame(spec_rows)
    spec_df.to_csv(args.out_dir / f"specificity_test_{args.tag}.csv", index=False)

    # ---- pathway-level deltas (observed) per (P, ct) ----
    pathway_rows = []
    for gene in eligible:
        for c in cell_types:
            if (gene, c) not in obs_store:
                continue
            o = obs_store[(gene, c)]
            p = pred_store[(gene, c)]
            for name, gset in GENE_SETS.items():
                gin = [g for g in gset if g in o.index and g in p.index]
                if len(gin) < 3:
                    continue
                pathway_rows.append({
                    "perturbation": gene, "cell_type": c, "pathway": name,
                    "n_genes_in_set": len(gin),
                    "observed_mean_delta": float(o.loc[gin].mean()),
                    "predicted_mean_delta": float(p.loc[gin].mean()),
                })
    pw_df = pd.DataFrame(pathway_rows)
    pw_df.to_csv(args.out_dir / f"pathway_signature_{args.tag}.csv", index=False)

    # ---- overall summary ----
    overall = {
        "tag": args.tag,
        "n_pert_celltype_rows": int(len(summary)),
        "n_unique_perturbations": int(summary.perturbation.nunique()),
        "median_pearson_r": float(summary.pearson_r.median()),
        "median_spearman_rho": float(summary.spearman_rho.median()),
        "median_cosine": float(summary.cosine.median()),
        "n_pearson_p05": int((summary.pearson_perm_p < 0.05).sum()),
        "n_pearson_p001": int((summary.pearson_perm_p < 0.001).sum()),
        "median_top50_sign_agree": float(summary["top50_sign_agree"].median()) if "top50_sign_agree" in summary else None,
        "n_top50_binom_p05": int((summary["top50_binom_p"] < 0.05).sum()) if "top50_binom_p" in summary else None,
        "by_cell_type_median_pearson": summary.groupby("cell_type").pearson_r.median().round(4).to_dict(),
        "best_5_rows": summary.nlargest(5, "pearson_r")[
            ["perturbation", "cell_type", "pearson_r", "pearson_perm_p", "cosine"]
        ].to_dict("records"),
        "specificity_per_celltype": spec_rows,
        "on_target_summary": {
            "median_predicted_delta": float(on_target_df.predicted_delta_on_target.median()),
            "median_observed_delta": float(on_target_df.observed_delta_on_target.median()),
            "frac_both_negative": float(((on_target_df.predicted_delta_on_target < 0)
                                          & (on_target_df.observed_delta_on_target < 0)).mean()),
        },
    }
    (args.out_dir / f"overall_summary_{args.tag}.json").write_text(json.dumps(overall, indent=2, default=str))
    print(json.dumps(overall, indent=2, default=str))

    # ---------------- FIGURES ----------------
    # Figure 1: Pearson r heatmap (perturbation × cell_type), with stars for permutation p<0.05.
    pivot_r = summary.pivot(index="perturbation", columns="cell_type", values="pearson_r")
    pivot_p = summary.pivot(index="perturbation", columns="cell_type", values="pearson_perm_p")
    cts = sorted(pivot_r.columns)
    perts = sorted(pivot_r.index)
    fig, ax = plt.subplots(figsize=(0.9 * len(cts) + 2.5, 0.45 * len(perts) + 1.5))
    vmax = max(0.3, float(np.nanmax(np.abs(pivot_r.values))))
    im = ax.imshow(pivot_r.loc[perts, cts].values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(cts))); ax.set_xticklabels(cts, rotation=20, ha="right")
    ax.set_yticks(range(len(perts))); ax.set_yticklabels([f"sg{p}" for p in perts])
    for i, p in enumerate(perts):
        for j, c in enumerate(cts):
            r = pivot_r.loc[p, c]; pp = pivot_p.loc[p, c]
            if pd.isna(r):
                continue
            stars = "***" if pp < 0.001 else ("**" if pp < 0.01 else ("*" if pp < 0.05 else ""))
            ax.text(j, i, f"{r:+.2f}{stars}", ha="center", va="center", fontsize=7,
                       color="white" if abs(r) > vmax * 0.6 else "black")
    ax.set_title(f"SpaceTravLR predicted vs SPAC-seq observed perturbation\nPearson r per (perturbation × cell type) — {args.tag}", fontsize=10)
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(args.fig_dir / f"fig1_pearson_heatmap_{args.tag}.png", dpi=180)
    plt.close(fig)

    # Figure 2: Scatter panel - best per perturbation
    eligible_with_signal = (summary.groupby("perturbation")["pearson_r"]
                              .max().reset_index().sort_values("pearson_r", ascending=False))
    show = eligible_with_signal.perturbation.head(6).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, gene in zip(axes.flat, show):
        row = summary[(summary.perturbation == gene)].sort_values("pearson_r", ascending=False).head(1)
        if row.empty:
            continue
        c = row.cell_type.iloc[0]
        p_v = pred_store[(gene, c)]
        o_v = obs_store[(gene, c)]
        common = list(set(p_v.index) & set(o_v.index))
        p_v = p_v.loc[common]; o_v = o_v.loc[common]
        ax.scatter(p_v, o_v, s=4, alpha=0.45, color="#1f77b4")
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        # highlight on-target gene
        if gene in p_v.index and gene in o_v.index:
            ax.scatter([p_v[gene]], [o_v[gene]], s=70, marker="*", color="#d62728",
                       edgecolor="k", linewidth=0.5, label=f"on-target ({gene})")
        # top 8 abs predicted
        topk_idx = np.argsort(np.abs(p_v.values))[-8:]
        for i in topk_idx:
            ax.annotate(common[i], (p_v.values[i], o_v.values[i]), fontsize=7, alpha=0.7)
        ax.set_xlabel(f"Predicted Δ ({gene} KO)")
        ax.set_ylabel("Observed Δ (sg{} − sgNTC)".format(gene))
        ax.set_title(
            f"sg{gene} | {c}: r={row.pearson_r.iloc[0]:+.3f} (p={row.pearson_perm_p.iloc[0]:.1e}) "
            f"cos={row.cosine.iloc[0]:+.3f}", fontsize=9,
        )
        if gene in p_v.index:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(f"Top perturbations: predicted vs observed (best cell type) — {args.tag}", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.fig_dir / f"fig2_top_perturbation_scatter_{args.tag}.png", dpi=180)
    plt.close(fig)

    # Figure 3: pathway scatter (observed vs predicted mean delta)
    if not pw_df.empty:
        fig, ax = plt.subplots(figsize=(7, 6))
        for c in pw_df.cell_type.unique():
            sub = pw_df[pw_df.cell_type == c]
            ax.scatter(sub.predicted_mean_delta, sub.observed_mean_delta, alpha=0.6, label=c, s=24)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        # correlation
        m = np.isfinite(pw_df.predicted_mean_delta) & np.isfinite(pw_df.observed_mean_delta)
        if m.sum() > 5:
            rho, p = stats.pearsonr(pw_df.predicted_mean_delta[m], pw_df.observed_mean_delta[m])
        else:
            rho, p = float("nan"), float("nan")
        ax.set_xlabel("Predicted mean Δ across pathway genes")
        ax.set_ylabel("Observed mean Δ across pathway genes")
        ax.set_title(f"Pathway-level validation (mean Δ per perturbation × cell type)\nPearson r={rho:+.3f} (p={p:.2g}) — {args.tag}", fontsize=10)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(args.fig_dir / f"fig3_pathway_scatter_{args.tag}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
