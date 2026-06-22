#!/usr/bin/env python3
"""Refined cell-type annotations and tumor mitochondrial receiver table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse

from celltype_annotation import LINEAGE_ORDER, refine_cell_types
from merci_port import donor_mt_signature_score, merci_loo_mt_est, merci_receiver_pre
from run_human_merci_analysis import HUMAN_MARKERS, annotate_bins, load_p1_bins

EPI_MARKERS = ["MUC2", "SI", "KRT20", "FABP1", "TFF3", "CLDN4"]


def load_tissue(data_dir: Path) -> sc.AnnData:
    adata = load_p1_bins(data_dir)
    mask = (adata.obs["in_tissue"] == 1) & (adata.obs["DeconvolutionClass"] == "singlet")
    adata = adata[mask].copy()
    adata = annotate_bins(adata)
    present = [g for g in EPI_MARKERS if g in adata.var_names]
    if len(present) >= 2:
        sc.tl.score_genes(adata, present, score_name="score_epithelial")
    else:
        adata.obs["score_epithelial"] = 0.0
    adata.obs = refine_cell_types(adata.obs)
    return adata


def expression_matrix(adata: sc.AnnData) -> pd.DataFrame:
    if sparse.issparse(adata.X):
        return pd.DataFrame(adata.X.toarray().T, index=adata.var_names, columns=adata.obs_names)
    return pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)


def stratified_sample(names: list[str], labels: pd.Series, per_type: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for ct in sorted(labels.unique()):
        ct_names = [n for n in names if labels[n] == ct]
        if not ct_names:
            continue
        k = min(len(ct_names), per_type)
        if len(ct_names) <= k:
            chosen.extend(ct_names)
        else:
            chosen.extend(rng.choice(ct_names, size=k, replace=False).tolist())
    return chosen


def build_receiver_table(
    pred: pd.DataFrame,
    obs: pd.DataFrame,
    cell_type_col: str,
) -> pd.DataFrame:
    rows = []
    tested = pred.copy()
    tested[cell_type_col] = obs.loc[tested.index, cell_type_col].values
    tested["lineage_refined"] = obs.loc[tested.index, "lineage_refined"].values

    for ct in sorted(obs[cell_type_col].dropna().unique()):
        if str(ct).startswith("Tumor"):
            continue
        total_n = int((obs[cell_type_col] == ct).sum())
        sub = tested[tested[cell_type_col] == ct]
        tested_n = len(sub)
        if tested_n == 0:
            continue
        n_rec = int((sub["prediction"] == "Receiver").sum())
        rate = n_rec / tested_n
        extrapolated = int(round(rate * total_n))
        rows.append({
            "cell_type": ct,
            "lineage": obs.loc[obs[cell_type_col] == ct, "lineage_refined"].mode().iloc[0],
            "n_bins_total": total_n,
            "n_bins_tested": tested_n,
            "n_receivers_tested": n_rec,
            "receiver_rate": rate,
            "n_receivers_estimated": extrapolated if tested_n < total_n else n_rec,
            "fully_tested": tested_n == total_n,
        })
    tab = pd.DataFrame(rows)
    if len(tab):
        tab = tab.sort_values("n_receivers_estimated", ascending=False)
    return tab


def make_figures(tab: pd.DataFrame, pred: pd.DataFrame, obs: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    top = tab.head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=top, y="cell_type", x="n_receivers_estimated", hue="lineage", dodge=False, ax=ax)
    ax.set_xlabel("Estimated tumor-mt receivers (MERCI)")
    ax.set_ylabel("")
    ax.set_title("Top cell types receiving tumor mitochondria")
    fig.savefig(fig_dir / "celltype_tumor_mt_receivers_barplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    rate = tab[tab["n_bins_tested"] >= 50].head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=rate, y="cell_type", x="receiver_rate", hue="lineage", dodge=False, ax=ax)
    ax.set_xlabel("Receiver rate (tested bins)")
    ax.set_title("Tumor mitochondrial receiver rate by cell type")
    fig.savefig(fig_dir / "celltype_tumor_mt_receiver_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    lineage_tab = (
        tab.groupby("lineage", observed=True)
        .agg(
            n_bins_total=("n_bins_total", "sum"),
            n_receivers_estimated=("n_receivers_estimated", "sum"),
        )
        .reset_index()
    )
    lineage_tab["receiver_rate"] = lineage_tab["n_receivers_estimated"] / lineage_tab["n_bins_total"]
    lineage_tab["lineage"] = pd.Categorical(lineage_tab["lineage"], categories=LINEAGE_ORDER, ordered=True)
    lineage_tab = lineage_tab.sort_values("lineage")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=lineage_tab, x="lineage", y="receiver_rate", ax=ax)
    ax.set_ylabel("Estimated receiver rate")
    ax.set_title("Tumor mt transfer by lineage")
    fig.savefig(fig_dir / "lineage_tumor_mt_receiver_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "P1-CRC")
    parser.add_argument("--sample-per-type", type=int, default=500)
    parser.add_argument("--top-rank", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.data_dir / "results" / "celltype_mt_receivers"
    fig_dir = args.data_dir / "figures" / "celltype_mt_receivers"
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = load_tissue(args.data_dir)
    obs = adata.obs
    tumor_names = obs.index[obs["lineage_refined"] == "tumor"].tolist()
    receiver_candidates = obs.index[obs["lineage_refined"] != "tumor"].tolist()

    sampled = stratified_sample(
        receiver_candidates,
        obs["cell_type_refined"],
        per_type=args.sample_per_type,
        seed=args.seed,
    )

    exp = expression_matrix(adata)
    dna = donor_mt_signature_score(exp, tumor_names, sampled, organism="human")
    rna = merci_loo_mt_est(
        exp,
        receiver_cells=sampled,
        donor_cells=tumor_names,
        organism="human",
        max_receivers=None,
        seed=args.seed,
    )
    pred = merci_receiver_pre(dna, rna, top_rank=args.top_rank)
    pred = pred.join(obs[["cell_type_refined", "lineage_refined", "DeconvolutionLabel1", "Periphery"]])

    fine_tab = build_receiver_table(pred, obs, "cell_type_refined")
    label1_tab = build_receiver_table(pred, obs, "DeconvolutionLabel1")

    obs_out = obs[["DeconvolutionLabel1", "DeconvolutionLabel2", "UnsupervisedL1", "UnsupervisedL2",
                   "lineage_refined", "cell_type_refined", "annotation_source"]].copy()
    obs_out.to_csv(out_dir / "refined_celltype_annotations.csv")

    pred.to_csv(out_dir / "merci_tumor_to_receiver_predictions.csv")
    fine_tab.to_csv(out_dir / "tumor_mt_receivers_by_refined_celltype.csv", index=False)
    label1_tab.to_csv(out_dir / "tumor_mt_receivers_by_label1.csv", index=False)

    summary = {
        "n_tumor_donor_bins": len(tumor_names),
        "n_non_tumor_bins": len(receiver_candidates),
        "n_receivers_tested_total": int(len(sampled)),
        "n_merci_receivers_tested": int((pred["prediction"] == "Receiver").sum()),
        "overall_receiver_rate_tested": float((pred["prediction"] == "Receiver").mean()),
        "n_receivers_estimated_total": int(fine_tab["n_receivers_estimated"].sum()),
        "sample_per_type": args.sample_per_type,
        "top_rank_pct": args.top_rank,
        "annotation_changes": int((obs["annotation_source"] != "label1+ul1_agree").sum()),
        "top_cell_types_by_estimated_receivers": fine_tab.head(15).to_dict("records"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    make_figures(fine_tab, pred, obs, fig_dir)
    print(json.dumps(summary, indent=2))
    print("\nTop refined cell types:")
    print(fine_tab.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
