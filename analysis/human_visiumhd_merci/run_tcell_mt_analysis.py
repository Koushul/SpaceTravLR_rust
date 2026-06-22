#!/usr/bin/env python3
"""T cell mitochondrial transfer analysis: tumor/macrophage donors and T cell state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse, stats
from scipy.spatial import cKDTree

from merci_port import donor_mt_signature_score, merci_loo_mt_est, merci_receiver_pre
from run_human_merci_analysis import annotate_bins, load_p1_bins

TCELL_LABEL1 = {"CD4 T cell", "CD8 T cell"}
TCELL_LABEL2 = {"CD4 T cell", "CD8 T cell"}
MAC_LABEL1 = {"Macrophage", "Proliferating Macrophages"}
MAC_LABEL2 = {"Macrophage", "Proliferating Macrophages"}

ACTIVATION_GENES = ["CD69", "IL2RA", "TNF", "IFNG", "ICOS", "CD44", "TNFRSF9", "CD28"]
EXHAUSTION_GENES = ["PDCD1", "CTLA4", "HAVCR2", "LAG3", "TIGIT", "TOX", "ENTPD1", "CXCL13"]
CYTOTOXIC_GENES = ["GZMB", "PRF1", "NKG7", "GNLY", "GZMA"]
MEMORY_GENES = ["IL7R", "CCR7", "SELL", "LEF1"]
TREG_GENES = ["FOXP3", "IKZF2", "CTLA4", "IL2RA"]


def is_tcell(obs_row) -> bool:
    l1 = str(obs_row.get("DeconvolutionLabel1", ""))
    l2 = str(obs_row.get("DeconvolutionLabel2", ""))
    l2u = str(obs_row.get("UnsupervisedL2", ""))
    return l1 in TCELL_LABEL1 or l2 in TCELL_LABEL2 or l2u.startswith("Tcells")


def is_macrophage(obs_row) -> bool:
    l1 = str(obs_row.get("DeconvolutionLabel1", ""))
    l2 = str(obs_row.get("DeconvolutionLabel2", ""))
    return l1 in MAC_LABEL1 or l2 in MAC_LABEL2


def is_tumor(obs_row) -> bool:
    return str(obs_row.get("cell_type_deconv", "")) == "tumor" or str(
        obs_row.get("DeconvolutionLabel1", "")
    ).startswith("Tumor")


def load_tissue(data_dir: Path) -> sc.AnnData:
    adata = load_p1_bins(data_dir)
    adata = annotate_bins(adata)
    mask = (adata.obs["in_tissue"] == 1) & (adata.obs["DeconvolutionClass"] == "singlet")
    adata = adata[mask].copy()
    adata.obs["is_tcell"] = [is_tcell(r) for _, r in adata.obs.iterrows()]
    adata.obs["is_mac"] = [is_macrophage(r) for _, r in adata.obs.iterrows()]
    adata.obs["is_tumor"] = [is_tumor(r) for _, r in adata.obs.iterrows()]
    adata.obs["tcell_subtype"] = "other"
    adata.obs.loc[adata.obs["DeconvolutionLabel1"] == "CD4 T cell", "tcell_subtype"] = "CD4"
    adata.obs.loc[adata.obs["DeconvolutionLabel1"] == "CD8 T cell", "tcell_subtype"] = "CD8"
    unsup = adata.obs["UnsupervisedL2"].astype(str)
    mask_t = adata.obs["is_tcell"] & (adata.obs["tcell_subtype"] == "other")
    adata.obs.loc[mask_t & unsup.str.startswith("Tcells"), "tcell_subtype"] = "T_unsup"
    return adata


def expression_matrix(adata: sc.AnnData) -> pd.DataFrame:
    if sparse.issparse(adata.X):
        return pd.DataFrame(adata.X.toarray().T, index=adata.var_names, columns=adata.obs_names)
    return pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)


def score_gene_set(adata: sc.AnnData, genes: list[str], name: str) -> None:
    present = [g for g in genes if g in adata.var_names]
    if len(present) >= 2:
        sc.tl.score_genes(adata, present, score_name=name)
    else:
        adata.obs[name] = 0.0


def transfer_from_donors(
    exp: pd.DataFrame,
    donors: list[str],
    receivers: list[str],
    organism: str = "human",
) -> pd.Series:
    rank = donor_mt_signature_score(exp, donors, receivers, organism=organism)
    mx = rank["MTvar_rank"].max()
    return rank["MTvar_rank"] / mx if mx > 0 else rank["MTvar_rank"]


def add_distances(adata: sc.AnnData, tcell_names: list[str]) -> pd.DataFrame:
    coords = adata.obsm["spatial"]
    tumor_idx = adata.obs["is_tumor"].to_numpy()
    mac_idx = adata.obs["is_mac"].to_numpy()
    tdf = adata.obs.loc[tcell_names].copy()
    tdf["x"] = coords[adata.obs_names.isin(tcell_names), 0]
    tdf["y"] = coords[adata.obs_names.isin(tcell_names), 1]

    if tumor_idx.sum():
        d_t, _ = cKDTree(coords[tumor_idx]).query(coords[adata.obs_names.isin(tcell_names)])
        tdf["dist_tumor_um"] = d_t
    if mac_idx.sum():
        d_m, _ = cKDTree(coords[mac_idx]).query(coords[adata.obs_names.isin(tcell_names)])
        tdf["dist_mac_um"] = d_m
    return tdf


def run_tcell_merci_loo(
    exp: pd.DataFrame,
    tcell_names: list[str],
    donor_names: list[str],
    max_receivers: int,
    seed: int,
) -> pd.DataFrame:
    return merci_loo_mt_est(
        exp,
        receiver_cells=tcell_names,
        donor_cells=donor_names,
        organism="human",
        max_receivers=max_receivers,
        seed=seed,
    )


def correlation_table(tdf: pd.DataFrame, score_cols: list[str], transfer_cols: list[str]) -> pd.DataFrame:
    rows = []
    for tc in transfer_cols:
        for sc_col in score_cols:
            if tc not in tdf.columns or sc_col not in tdf.columns:
                continue
            valid = tdf[[tc, sc_col]].dropna()
            if len(valid) < 30:
                continue
            r, p = stats.spearmanr(valid[tc], valid[sc_col])
            rows.append({"transfer_metric": tc, "state_score": sc_col, "spearman_r": r, "p_value": p, "n": len(valid)})
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr"] = stats.false_discovery_control(out["p_value"].fillna(1), method="bh")
    return out.sort_values("spearman_r", ascending=False)


def make_figures(out_df: pd.DataFrame, fig_dir: Path, corr_df: pd.DataFrame) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(
        axes,
        ["transfer_from_tumor", "transfer_from_mac", "transfer_from_tumor_mac"],
        ["From tumor donors", "From macrophage donors", "From tumor+mac donors"],
    ):
        if col not in out_df.columns:
            continue
        sns.boxplot(data=out_df, x="tcell_subtype", y=col, ax=ax, order=["CD4", "CD8", "T_unsup"])
        ax.set_title(title)
        ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(fig_dir / "tcell_transfer_by_subtype.png", dpi=150)
    plt.close(fig)

    if "dist_tumor_um" in out_df.columns and "transfer_from_tumor" in out_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        out_df["dist_bin"] = pd.cut(
            out_df["dist_tumor_um"],
            bins=[0, 50, 100, 200, 500, np.inf],
            labels=["0-50", "50-100", "100-200", "200-500", ">500"],
        )
        sns.boxplot(data=out_df, x="dist_bin", y="transfer_from_tumor", ax=ax)
        ax.set_xlabel("Distance to nearest tumor bin (µm)")
        ax.set_ylabel("T cell transfer from tumor")
        ax.set_title("T cell mt transfer vs tumor proximity")
        fig.savefig(fig_dir / "tcell_transfer_vs_tumor_distance.png", dpi=150)
        plt.close(fig)

    if "dist_mac_um" in out_df.columns and "transfer_from_mac" in out_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        out_df["mac_dist_bin"] = pd.cut(
            out_df["dist_mac_um"],
            bins=[0, 50, 100, 200, 500, np.inf],
            labels=["0-50", "50-100", "100-200", "200-500", ">500"],
        )
        sns.boxplot(data=out_df, x="mac_dist_bin", y="transfer_from_mac", ax=ax)
        ax.set_xlabel("Distance to nearest macrophage bin (µm)")
        ax.set_ylabel("T cell transfer from macrophage")
        ax.set_title("T cell mt transfer vs macrophage proximity")
        fig.savefig(fig_dir / "tcell_transfer_vs_mac_distance.png", dpi=150)
        plt.close(fig)

    state_cols = [c for c in ["activation_score", "exhaustion_score", "cytotoxic_score", "memory_score", "treg_score"] if c in out_df.columns]
    transfer_col = "Donor_MT_frac" if "Donor_MT_frac" in out_df.columns else "transfer_from_tumor_mac"
    if state_cols and transfer_col in out_df.columns:
        fig, axes = plt.subplots(1, len(state_cols), figsize=(4 * len(state_cols), 4))
        if len(state_cols) == 1:
            axes = [axes]
        for ax, sc_col in zip(axes, state_cols):
            sns.scatterplot(data=out_df, x=transfer_col, y=sc_col, hue="tcell_subtype", ax=ax, s=12, alpha=0.5)
            ax.set_title(sc_col.replace("_score", ""))
        fig.savefig(fig_dir / "tcell_state_vs_transfer.png", dpi=150)
        plt.close(fig)

    if len(corr_df):
        top = corr_df.head(12)
        fig, ax = plt.subplots(figsize=(8, 5))
        top["label"] = top["transfer_metric"] + " × " + top["state_score"]
        sns.barplot(data=top, y="label", x="spearman_r", ax=ax)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Spearman r")
        ax.set_title("Transfer metrics vs T cell state scores")
        fig.savefig(fig_dir / "tcell_transfer_state_correlation.png", dpi=150)
        plt.close(fig)

    if "x" in out_df.columns and "transfer_from_tumor_mac" in out_df.columns:
        fig, ax = plt.subplots(figsize=(7, 6))
        sca = ax.scatter(
            out_df["x"], out_df["y"], c=out_df["transfer_from_tumor_mac"],
            s=3, cmap="magma", vmax=out_df["transfer_from_tumor_mac"].quantile(0.99),
        )
        plt.colorbar(sca, ax=ax, label="T cell transfer score")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title("Spatial: T cell mitochondrial transfer (tumor+mac donors)")
        fig.savefig(fig_dir / "tcell_transfer_spatial.png", dpi=150)
        plt.close(fig)

    if "exhaustion_score" in out_df.columns and "activation_score" in out_df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        q = out_df["transfer_from_tumor_mac"].quantile(0.75)
        high = out_df["transfer_from_tumor_mac"] >= q
        ax.scatter(
            out_df.loc[~high, "activation_score"], out_df.loc[~high, "exhaustion_score"],
            s=8, alpha=0.4, label="low transfer", c="#7f7f7f",
        )
        ax.scatter(
            out_df.loc[high, "activation_score"], out_df.loc[high, "exhaustion_score"],
            s=8, alpha=0.6, label="high transfer (top 25%)", c="#d62728",
        )
        ax.set_xlabel("Activation score")
        ax.set_ylabel("Exhaustion score")
        ax.legend()
        ax.set_title("T cell activation vs exhaustion by transfer level")
        fig.savefig(fig_dir / "tcell_activation_exhaustion_by_transfer.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "P1-CRC")
    parser.add_argument("--max-tcells-loo", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.data_dir / "results" / "tcell_analysis"
    fig_dir = args.data_dir / "figures" / "tcell_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = load_tissue(args.data_dir)
    tcell_names = adata.obs_names[adata.obs["is_tcell"]].tolist()
    tumor_names = adata.obs_names[adata.obs["is_tumor"]].tolist()
    mac_names = adata.obs_names[adata.obs["is_mac"]].tolist()

    score_gene_set(adata, ACTIVATION_GENES, "activation_score")
    score_gene_set(adata, EXHAUSTION_GENES, "exhaustion_score")
    score_gene_set(adata, CYTOTOXIC_GENES, "cytotoxic_score")
    score_gene_set(adata, MEMORY_GENES, "memory_score")
    score_gene_set(adata, TREG_GENES, "treg_score")

    exp = expression_matrix(adata)

    transfer_tumor = transfer_from_donors(exp, tumor_names, tcell_names)
    transfer_mac = transfer_from_donors(exp, mac_names, tcell_names)
    transfer_both = transfer_from_donors(exp, tumor_names + mac_names, tcell_names)

    out_df = adata.obs.loc[tcell_names].copy()
    out_df["transfer_from_tumor"] = transfer_tumor.loc[tcell_names]
    out_df["transfer_from_mac"] = transfer_mac.loc[tcell_names]
    out_df["transfer_from_tumor_mac"] = transfer_both.loc[tcell_names]

    dist_df = add_distances(adata, tcell_names)
    for c in dist_df.columns:
        if c not in out_df.columns:
            out_df[c] = dist_df[c]

    loo = run_tcell_merci_loo(exp, tcell_names, tumor_names + mac_names, args.max_tcells_loo, args.seed)
    out_df["Donor_MT_frac"] = loo.loc[out_df.index.intersection(loo.index), "Donor_MT_frac"]
    out_df["Receiver_MT_frac"] = loo.loc[out_df.index.intersection(loo.index), "Receiver_MT_frac"]

    dna_t = donor_mt_signature_score(exp, tumor_names, loo.index.tolist(), organism="human")
    dna_m = donor_mt_signature_score(exp, mac_names, loo.index.tolist(), organism="human")
    pred = merci_receiver_pre(dna_t, loo, top_rank=50)
    pred_mac = merci_receiver_pre(dna_m, loo, top_rank=50)
    out_df.loc[pred.index, "receiver_from_tumor"] = pred["prediction"].values
    out_df.loc[pred_mac.index, "receiver_from_mac"] = pred_mac["prediction"].values

    corr_df = correlation_table(
        out_df,
        ["activation_score", "exhaustion_score", "cytotoxic_score", "memory_score", "treg_score"],
        ["transfer_from_tumor", "transfer_from_mac", "transfer_from_tumor_mac", "Donor_MT_frac"],
    )

    subtype_stats = []
    for st in ["CD4", "CD8", "T_unsup"]:
        sub = out_df[out_df["tcell_subtype"] == st]
        if len(sub) < 20:
            continue
        subtype_stats.append({
            "subtype": st,
            "n": len(sub),
            "median_transfer_tumor": float(sub["transfer_from_tumor"].median()),
            "median_transfer_mac": float(sub["transfer_from_mac"].median()),
            "median_activation": float(sub["activation_score"].median()),
            "median_exhaustion": float(sub["exhaustion_score"].median()),
            "median_cytotoxic": float(sub["cytotoxic_score"].median()),
        })

    high = out_df["transfer_from_tumor_mac"] >= out_df["transfer_from_tumor_mac"].quantile(0.75)
    low = out_df["transfer_from_tumor_mac"] <= out_df["transfer_from_tumor_mac"].quantile(0.25)
    state_compare = {}
    for sc_col in ["activation_score", "exhaustion_score", "cytotoxic_score", "memory_score"]:
        if sc_col not in out_df.columns:
            continue
        _, p = stats.mannwhitneyu(out_df.loc[high, sc_col], out_df.loc[low, sc_col], alternative="two-sided")
        state_compare[sc_col] = {
            "high_transfer_median": float(out_df.loc[high, sc_col].median()),
            "low_transfer_median": float(out_df.loc[low, sc_col].median()),
            "mannwhitney_p": float(p),
        }

    if "dist_tumor_um" in out_df.columns:
        r_t, p_t = stats.spearmanr(out_df["dist_tumor_um"], out_df["transfer_from_tumor"], nan_policy="omit")
    else:
        r_t, p_t = np.nan, np.nan
    if "dist_mac_um" in out_df.columns:
        r_m, p_m = stats.spearmanr(out_df["dist_mac_um"], out_df["transfer_from_mac"], nan_policy="omit")
    else:
        r_m, p_m = np.nan, np.nan

    summary = {
        "n_tcell_bins": len(tcell_names),
        "n_cd4": int((out_df["tcell_subtype"] == "CD4").sum()),
        "n_cd8": int((out_df["tcell_subtype"] == "CD8").sum()),
        "n_tumor_donors": len(tumor_names),
        "n_mac_donors": len(mac_names),
        "median_transfer_from_tumor": float(out_df["transfer_from_tumor"].median()),
        "median_transfer_from_mac": float(out_df["transfer_from_mac"].median()),
        "receiver_rate_tumor_donor_loo": float((out_df["receiver_from_tumor"] == "Receiver").mean()) if "receiver_from_tumor" in out_df else None,
        "receiver_rate_mac_donor_loo": float((out_df["receiver_from_mac"] == "Receiver").mean()) if "receiver_from_mac" in out_df else None,
        "spearman_dist_tumor_vs_transfer": {"r": float(r_t), "p": float(p_t)},
        "spearman_dist_mac_vs_transfer": {"r": float(r_m), "p": float(p_m)},
        "subtype_stats": subtype_stats,
        "high_vs_low_transfer_state": state_compare,
        "top_correlations": corr_df.head(10).to_dict("records") if len(corr_df) else [],
    }

    out_df.to_csv(out_dir / "tcell_transfer_and_state.csv")
    corr_df.to_csv(out_dir / "tcell_transfer_state_correlations.csv", index=False)
    pd.DataFrame(subtype_stats).to_csv(out_dir / "tcell_subtype_summary.csv", index=False)
    with open(out_dir / "tcell_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    make_figures(out_df, fig_dir, corr_df)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
