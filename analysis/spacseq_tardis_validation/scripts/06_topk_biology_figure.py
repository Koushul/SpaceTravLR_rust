#!/usr/bin/env python3
"""Per-perturbation top-K predicted-effect biology table + figure.

For the (perturbation, cell type) pairs that pass the per-cell-type test,
take the K genes with the largest |predicted Δ| and tabulate the observed Δ
on those genes. Show sign-agreement and direction match with paper biology.
"""

from __future__ import annotations

import argparse
import glob
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


def dense(adata, genes):
    keep = [g for g in genes if g in adata.var_names]
    sub = adata[:, keep]
    arr = sub.X.toarray() if sparse.issparse(sub.X) else np.asarray(sub.X)
    return pd.DataFrame(arr, index=adata.obs_names, columns=keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-h5ad", type=Path, required=True)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--perturbed-h5ad", type=Path, default=ROOT / "data/perturbed_pool.h5ad")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, required=True)
    ap.add_argument("--tag", default="seed")
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--pairs", nargs="*", default=[
        "Il4ra:immune", "Il4ra:myeloid", "Cd83:immune", "Cd83:myeloid",
        "Cks1b:immune", "Bcam:immune", "Cd74:fibroblast",
    ])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline_h5ad.is_dir():
        baseline_path = sorted(args.baseline_h5ad.glob("*.h5ad"))[0]
    else:
        baseline_path = args.baseline_h5ad
    baseline = sc.read_h5ad(baseline_path)
    if "imputed_count" in baseline.layers:
        baseline.X = baseline.layers["imputed_count"]
    pool = sc.read_h5ad(args.perturbed_h5ad)
    sc.pp.normalize_total(pool, target_sum=10000)
    sc.pp.log1p(pool)

    pairs = [tuple(p.split(":")) for p in args.pairs]
    long_rows = []
    panel_panels = {}

    for gene, ct in pairs:
        pred_path = args.pred_dir / f"predicted_KO_{gene}.feather"
        if not pred_path.exists():
            print(f"skip {gene}: no prediction")
            continue
        pred = pd.read_feather(pred_path).set_index("CellID")
        common = sorted(set(pred.columns) & set(pool.var_names) & set(baseline.var_names))
        pred = pred[common]
        base = dense(baseline, common).loc[pred.index]
        ct_map = baseline.obs.cell_type.loc[pred.index]
        m_pred = (ct_map == ct).values
        if m_pred.sum() < 5:
            print(f"skip {gene}:{ct} — too few baseline cells")
            continue
        pred_d = (pred[m_pred].mean(0) - base[m_pred].mean(0))

        expr = dense(pool, common)
        ctp = (pool.obs.cell_type == ct).values
        ntc = ctp & (pool.obs.target_gene == "non-targeting").values
        per = ctp & (pool.obs.target_gene == gene).values
        if ntc.sum() < 10 or per.sum() < 20:
            print(f"skip {gene}:{ct} — too few observed cells")
            continue
        obs_d = expr.loc[per].mean(0) - expr.loc[ntc].mean(0)

        df = pd.DataFrame({"gene": common, "pred": pred_d.values, "obs": obs_d.values})
        df = df[df.gene != gene]
        df["abs_pred"] = df.pred.abs()
        df["sign_match"] = (np.sign(df.pred) == np.sign(df.obs)) & (df.pred != 0)
        topk = df.sort_values("abs_pred", ascending=False).head(args.topk).copy()
        topk["pair"] = f"sg{gene} | {ct}"
        long_rows.append(topk[["pair", "gene", "pred", "obs", "sign_match"]])
        panel_panels[(gene, ct)] = topk

    out_long = pd.concat(long_rows, ignore_index=True)
    out_long.to_csv(args.out_dir / f"top{args.topk}_perturbation_genes_{args.tag}.csv", index=False)

    # tabulate sign agreement per pair
    by_pair = (out_long.groupby("pair")
                .agg(n_genes=("gene", "size"),
                     n_sign_match=("sign_match", "sum"))
                .reset_index())
    by_pair["sign_match_rate"] = by_pair.n_sign_match / by_pair.n_genes
    by_pair["binom_p_one_sided"] = by_pair.apply(
        lambda r: stats.binomtest(int(r.n_sign_match), int(r.n_genes), p=0.5,
                                    alternative="greater").pvalue, axis=1)
    by_pair.to_csv(args.out_dir / f"top{args.topk}_sign_agreement_{args.tag}.csv", index=False)
    print(by_pair.to_string(index=False))

    # heatmap: rows = pairs, cols = top genes (per pair), color = observed Δ,
    # sign markers (▲▼) on predicted direction
    if not panel_panels:
        return
    fig, axes = plt.subplots(len(panel_panels), 1,
                              figsize=(0.9 + 0.45 * args.topk, 1.0 + 0.6 * len(panel_panels)),
                              squeeze=False)
    for ax, ((gene, ct), top) in zip(axes[:, 0], panel_panels.items()):
        order = top.sort_values("pred").index  # sort by predicted Δ ascending
        top = top.loc[order]
        x = np.arange(len(top))
        # observed Δ as a bar
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in top.obs]
        ax.bar(x, top.obs, color=colors, edgecolor="k", linewidth=0.3)
        # gene labels with predicted-Δ direction arrows
        labels = [f"{g}\n{'↓' if p < 0 else ('↑' if p > 0 else '·')}" for g, p in zip(top.gene, top.pred)]
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("Observed Δ")
        rate = float(top["sign_match"].mean())
        ax.set_title(f"sg{gene} | {ct} — observed Δ on top-{args.topk} predicted-magnitude genes "
                     f"(arrow = predicted direction; sign match = {rate:.0%})",
                     fontsize=9)
    fig.tight_layout()
    fig.savefig(args.fig_dir / f"fig4_top{args.topk}_genes_per_pair_{args.tag}.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
