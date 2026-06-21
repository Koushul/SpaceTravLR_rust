#!/usr/bin/env python3
"""Spatial niche analysis of mitochondrial transfer patterns in P1 CRC Visium HD."""

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
from sklearn.cluster import KMeans

from merci_port import donor_mt_signature_score
from run_human_merci_analysis import annotate_bins, load_p1_bins


def load_full_tissue(data_dir: Path) -> sc.AnnData:
    adata = load_p1_bins(data_dir)
    adata = annotate_bins(adata)
    mask = (adata.obs["in_tissue"] == 1) & (adata.obs["DeconvolutionClass"] == "singlet")
    return adata[mask].copy()


def compute_transfer_scores(adata: sc.AnnData, tumor_mask: np.ndarray) -> pd.Series:
    if sparse.issparse(adata.X):
        exp = pd.DataFrame(adata.X.toarray().T, index=adata.var_names, columns=adata.obs_names)
    else:
        exp = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)

    donors = adata.obs_names[~tumor_mask & adata.obs["cell_type_deconv"].isin(["immune", "myeloid"])].tolist()
    receivers = adata.obs_names[tumor_mask].tolist()
    if len(donors) < 50:
        donors = adata.obs_names[adata.obs["cell_type_deconv"].isin(["immune", "myeloid"])].tolist()
    dna_rank = donor_mt_signature_score(exp, donors, receivers, organism="human")
    return dna_rank["MTvar_rank"], dna_rank["MTvar_rank"] / dna_rank["MTvar_rank"].max()


def add_spatial_context(adata: sc.AnnData, tumor_mask: np.ndarray, radii_um: tuple[float, ...] = (50.0, 100.0, 200.0)) -> sc.AnnData:
    coords = adata.obsm["spatial"]
    immune_mask = adata.obs["cell_type_deconv"].isin(["immune", "myeloid"]).to_numpy()
    immune_coords = coords[immune_mask]
    tree = cKDTree(immune_coords) if immune_mask.sum() else None

    tumor_idx = np.where(tumor_mask)[0]
    for r in radii_um:
        col = f"immune_count_{int(r)}um"
        adata.obs[col] = 0
        if tree is not None:
            counts = tree.query_ball_point(coords[tumor_idx], r=r, return_length=True)
            adata.obs.iloc[tumor_idx, adata.obs.columns.get_loc(col)] = counts

    adata.obs["dist_immune_um"] = np.nan
    if tree is not None and tumor_mask.sum():
        dist, _ = tree.query(coords[tumor_mask], k=1)
        adata.obs.loc[tumor_mask, "dist_immune_um"] = dist

    x, y = coords[:, 0], coords[:, 1]
    adata.obs["spatial_sector"] = pd.cut(
        np.arctan2(y - y.mean(), x - x.mean()),
        bins=8,
        labels=[f"S{i}" for i in range(8)],
    ).astype(str)
    return adata


def niche_enrichment_tests(tumor_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for niche_col in [
        "Periphery", "DeconvolutionLabel1", "DeconvolutionLabel2",
        "UnsupervisedL1", "UnsupervisedL2", "spatial_sector",
    ]:
        if niche_col not in tumor_df.columns:
            continue
        for niche, grp in tumor_df.groupby(niche_col):
            if len(grp) < 30:
                continue
            med = grp["transfer_score"].median()
            global_med = tumor_df["transfer_score"].median()
            _, p = stats.mannwhitneyu(grp["transfer_score"], tumor_df["transfer_score"], alternative="two-sided")
            rows.append({
                "niche_type": niche_col,
                "niche": str(niche),
                "n_bins": len(grp),
                "median_transfer": float(med),
                "global_median": float(global_med),
                "enrichment_vs_global": float(med - global_med),
                "mannwhitney_p": float(p),
            })
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr"] = stats.false_discovery_control(out["mannwhitney_p"].fillna(1.0), method="bh")
    return out.sort_values("enrichment_vs_global", ascending=False)


def receiver_rate_by_niche(tumor_df: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    merged = tumor_df.join(pred[["prediction", "Donor_MT_frac"]], how="inner")
    rows = []
    for niche_col in ["Periphery", "DeconvolutionLabel1", "DeconvolutionLabel2", "UnsupervisedL2", "spatial_sector"]:
        for niche, grp in merged.groupby(niche_col):
            if len(grp) < 20:
                continue
            rec = (grp["prediction"] == "Receiver").mean()
            rows.append({
                "niche_type": niche_col,
                "niche": str(niche),
                "n_bins": len(grp),
                "receiver_rate": float(rec),
                "median_donor_mt_frac": float(grp["Donor_MT_frac"].median()),
            })
    return pd.DataFrame(rows).sort_values("receiver_rate", ascending=False)


def spatial_hotspots(tumor_df: pd.DataFrame, n_clusters: int = 12) -> pd.DataFrame:
    coords = tumor_df[["x", "y"]].to_numpy()
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    tumor_df = tumor_df.copy()
    tumor_df["spatial_cluster"] = km.fit_predict(coords).astype(str)

    rows = []
    for cl, grp in tumor_df.groupby("spatial_cluster"):
        rows.append({
            "spatial_cluster": cl,
            "n_bins": len(grp),
            "centroid_x": float(grp["x"].mean()),
            "centroid_y": float(grp["y"].mean()),
            "median_transfer": float(grp["transfer_score"].median()),
            "mean_transfer": float(grp["transfer_score"].mean()),
            "frac_high_transfer": float((grp["transfer_score"] > grp["transfer_score"].quantile(0.75)).mean()),
        })
    return pd.DataFrame(rows).sort_values("median_transfer", ascending=False), tumor_df


def distance_stratified_analysis(tumor_df: pd.DataFrame) -> pd.DataFrame:
    df = tumor_df.dropna(subset=["dist_immune_um"]).copy()
    df["dist_bin"] = pd.cut(
        df["dist_immune_um"],
        bins=[0, 50, 100, 200, 400, 1000, np.inf],
        labels=["0-50", "50-100", "100-200", "200-400", "400-1000", ">1000"],
    )
    rows = []
    for b, grp in df.groupby("dist_bin", observed=True):
        rows.append({
            "dist_bin_um": str(b),
            "n_bins": len(grp),
            "median_transfer": float(grp["transfer_score"].median()),
            "mean_transfer": float(grp["transfer_score"].mean()),
        })
    return pd.DataFrame(rows)


def local_immune_density_analysis(tumor_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in tumor_df.columns if c.startswith("immune_count_")]:
        radius = col.replace("immune_count_", "").replace("um", "")
        q = tumor_df[col].quantile([0.25, 0.5, 0.75])
        low = tumor_df[tumor_df[col] <= q.iloc[0]]
        high = tumor_df[tumor_df[col] >= q.iloc[2]]
        if len(low) < 30 or len(high) < 30:
            continue
        _, p = stats.mannwhitneyu(high["transfer_score"], low["transfer_score"], alternative="greater")
        rows.append({
            "radius_um": radius,
            "median_transfer_low_density": float(low["transfer_score"].median()),
            "median_transfer_high_density": float(high["transfer_score"].median()),
            "n_low": len(low),
            "n_high": len(high),
            "mannwhitney_p_high_gt_low": float(p),
        })
    return pd.DataFrame(rows)


def make_figures(
    adata: sc.AnnData,
    tumor_df: pd.DataFrame,
    niche_stats: pd.DataFrame,
    hotspot_df: pd.DataFrame,
    dist_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 7))
    sca = ax.scatter(
        tumor_df["x"], tumor_df["y"], c=tumor_df["transfer_score"],
        s=0.6, cmap="magma", vmax=tumor_df["transfer_score"].quantile(0.99), alpha=0.85,
    )
    plt.colorbar(sca, ax=ax, label="Mitochondrial transfer score")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("Spatial distribution of mitochondrial transfer (all tumor bins)")
    fig.savefig(out_dir / "niche_transfer_spatial_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    immune = adata[adata.obs["cell_type_deconv"].isin(["immune", "myeloid"])]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(tumor_df["x"], tumor_df["y"], c=tumor_df["transfer_score"], s=0.5, cmap="magma",
               vmax=tumor_df["transfer_score"].quantile(0.99), alpha=0.7)
    ax.scatter(immune.obsm["spatial"][:, 0], immune.obsm["spatial"][:, 1],
               c="#2ca02c", s=2, alpha=0.5, label="Immune/myeloid bins")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(markerscale=3)
    ax.set_title("Transfer score overlaid on immune/myeloid locations")
    fig.savefig(out_dir / "niche_transfer_with_immune.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    top = niche_stats[niche_stats["fdr"] < 0.05].head(15) if "fdr" in niche_stats.columns else niche_stats.head(15)
    if len(top):
        fig, ax = plt.subplots(figsize=(9, 5))
        top["label"] = top["niche_type"] + ": " + top["niche"]
        sns.barplot(data=top, y="label", x="enrichment_vs_global", ax=ax, palette="RdBu_r")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Median transfer score enrichment vs global")
        ax.set_title("Top enriched niches (FDR < 0.05)")
        fig.savefig(out_dir / "niche_enrichment_barplot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=dist_df, x="dist_bin_um", y="median_transfer", ax=ax, color="#4c72b0")
    ax.set_xlabel("Distance to nearest immune/myeloid bin (µm)")
    ax.set_ylabel("Median transfer score")
    ax.set_title("Transfer vs distance from immune infiltrate")
    fig.savefig(out_dir / "niche_transfer_by_immune_distance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for _, row in hotspot_df.head(6).iterrows():
        cl = str(row["spatial_cluster"])
        sub = tumor_df[tumor_df["spatial_cluster"] == cl]
        ax.scatter(sub["x"], sub["y"], s=1, label=f"C{cl} (med={row['median_transfer']:.2f})")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(fontsize=7, markerscale=3)
    ax.set_title("Top spatial clusters by transfer score")
    fig.savefig(out_dir / "niche_spatial_hotspot_clusters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    deconv = tumor_df.groupby("DeconvolutionLabel2")["transfer_score"].median().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    deconv.head(12).plot(kind="barh", ax=ax, color="#d62728")
    ax.set_xlabel("Median transfer score")
    ax.set_title("Transfer by tumor deconvolution subtype (Label2)")
    fig.savefig(out_dir / "niche_transfer_by_tumor_subtype.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    periphery = tumor_df.groupby("Periphery")["transfer_score"].apply(list)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot(
        [periphery.get("Tumor", []), periphery.get("50 micron", [])],
        tick_labels=["Tumor core", "50 µm band"],
    )
    ax.set_ylabel("Transfer score")
    ax.set_title("Tumor core vs invasive front (50 µm band)")
    fig.savefig(out_dir / "niche_core_vs_front.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "P1-CRC")
    args = parser.parse_args()

    out_dir = args.data_dir / "results" / "niche_analysis"
    fig_dir = args.data_dir / "figures" / "niche_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    adata = load_full_tissue(args.data_dir)
    tumor_mask = (adata.obs["cell_type_deconv"] == "tumor").to_numpy()
    transfer_rank, transfer_score = compute_transfer_scores(adata, tumor_mask)
    adata = add_spatial_context(adata, tumor_mask)

    tumor_df = adata.obs.loc[tumor_mask].copy()
    tumor_df["transfer_rank"] = transfer_rank.loc[tumor_df.index]
    tumor_df["transfer_score"] = transfer_score.loc[tumor_df.index]
    tumor_df["x"] = adata.obsm["spatial"][tumor_mask, 0]
    tumor_df["y"] = adata.obsm["spatial"][tumor_mask, 1]

    pred_path = args.data_dir / "results" / "merci_receiver_predictions.csv"
    pred = pd.read_csv(pred_path, index_col=0) if pred_path.exists() else pd.DataFrame()

    niche_stats = niche_enrichment_tests(tumor_df)
    receiver_niche = receiver_rate_by_niche(tumor_df, pred) if len(pred) else pd.DataFrame()
    hotspot_df, tumor_df = spatial_hotspots(tumor_df)
    dist_df = distance_stratified_analysis(tumor_df)
    density_df = local_immune_density_analysis(tumor_df)

    niche_stats.to_csv(out_dir / "niche_enrichment_stats.csv", index=False)
    receiver_niche.to_csv(out_dir / "receiver_rate_by_niche.csv", index=False)
    hotspot_df.to_csv(out_dir / "spatial_hotspot_clusters.csv", index=False)
    dist_df.to_csv(out_dir / "transfer_by_immune_distance.csv", index=False)
    density_df.to_csv(out_dir / "transfer_by_local_immune_density.csv", index=False)
    tumor_df.to_csv(out_dir / "tumor_bins_with_transfer_scores.csv")

    make_figures(adata, tumor_df, niche_stats, hotspot_df, dist_df, fig_dir)

    summary = {
        "n_tumor_bins": int(tumor_mask.sum()),
        "global_median_transfer": float(tumor_df["transfer_score"].median()),
        "top_enriched_niches_fdr05": niche_stats[niche_stats["fdr"] < 0.05].head(10)[
            ["niche_type", "niche", "median_transfer", "enrichment_vs_global", "fdr"]
        ].to_dict("records") if "fdr" in niche_stats.columns else [],
        "top_spatial_hotspots": hotspot_df.head(5).to_dict("records"),
        "distance_gradient": dist_df.to_dict("records"),
        "immune_density_association": density_df.to_dict("records"),
        "core_vs_front": {
            "tumor_core_median": float(tumor_df.loc[tumor_df["Periphery"] == "Tumor", "transfer_score"].median()),
            "front_50um_median": float(tumor_df.loc[tumor_df["Periphery"] == "50 micron", "transfer_score"].median()),
        },
    }
    if len(receiver_niche):
        summary["top_receiver_niches"] = receiver_niche.head(8).to_dict("records")

    with open(out_dir / "niche_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
