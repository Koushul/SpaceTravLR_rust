#!/usr/bin/env python3
"""MC38 Visium HD (SPAC-seq subQ-1) segmentation + MERCI-inspired mitochondrial transfer analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
from scipy import sparse, stats
from sklearn.neighbors import NearestNeighbors

from merci_port import (
    cell_number_test,
    donor_mt_signature_score,
    merci_loo_mt_est,
    merci_receiver_pre,
)

MARKERS = {
    "tumor": ["Epcam", "Krt8", "Krt18", "Msln", "Pecam1"],
    "immune": ["Ptprc", "Cd3d", "Cd3e", "Cd8a", "Cd4", "Nkg7"],
    "myeloid": ["Adgre1", "Cd68", "Itgam", "Ly6c2"],
    "fibroblast": ["Col1a1", "Col1a2", "Dcn", "Pdgfra"],
}


def cellid_to_barcode(cell_id: int) -> str:
    return f"cellid_{int(cell_id):09d}-1"


def load_cell_adata(data_dir: Path) -> sc.AnnData:
    h5 = data_dir / "segmentation/extracted/segmentation/filtered_feature_cell_matrix.h5"
    geo = data_dir / "segmentation/extracted/segmentation/graphclust_annotated_cell_segmentations.geojson"
    adata = sc.read_10x_h5(h5)
    adata.var_names_make_unique()

    gdf = gpd.read_file(geo)
    gdf["barcode"] = gdf["cell_id"].map(cellid_to_barcode)
    gdf["graphclust"] = gdf["classification"].apply(
        lambda x: x["name"] if isinstance(x, dict) else str(x)
    )
    gdf_indexed = gdf.set_index("barcode")
    adata = adata[adata.obs_names.isin(gdf_indexed.index)].copy()
    adata.obs["graphclust"] = gdf_indexed.loc[adata.obs_names, "graphclust"].astype("category").values
    centroid_map = {
        barcode: (geom.centroid.x, geom.centroid.y)
        for barcode, geom in zip(gdf["barcode"], gdf.geometry)
    }
    adata.obsm["spatial"] = np.array([centroid_map[b] for b in adata.obs_names])
    return adata


def score_cell_types(adata: sc.AnnData) -> sc.AnnData:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    for ct, genes in MARKERS.items():
        present = [g for g in genes if g in adata.var_names]
        if not present:
            adata.obs[f"score_{ct}"] = 0.0
            continue
        sc.tl.score_genes(adata, present, score_name=f"score_{ct}")
    scores = adata.obs[[f"score_{k}" for k in MARKERS]]
    adata.obs["cell_type"] = scores.idxmax(axis=1).str.replace("score_", "")
    adata.obs.loc[adata.obs["score_immune"] < 0.1, "cell_type"] = np.where(
        adata.obs.loc[adata.obs["score_immune"] < 0.1, "score_tumor"]
        > adata.obs.loc[adata.obs["score_immune"] < 0.1, "score_fibroblast"],
        "tumor",
        "fibroblast",
    )
    adata.obs.loc[adata.obs["score_immune"] >= 0.25, "cell_type"] = "immune"
    adata.obs.loc[adata.obs["score_myeloid"] >= 0.3, "cell_type"] = "myeloid"
    return adata


def run_merci(adata: sc.AnnData, out_dir: Path, max_receivers: int, seed: int) -> pd.DataFrame:
    if sparse.issparse(adata.X):
        exp = pd.DataFrame(
            adata.X.toarray().T,
            index=adata.var_names,
            columns=adata.obs_names,
        )
    else:
        exp = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)

    donors = adata.obs_names[adata.obs["cell_type"].isin(["immune", "myeloid"])].tolist()
    receivers = adata.obs_names[adata.obs["cell_type"] == "tumor"].tolist()
    if len(donors) < 50:
        donors = adata.obs_names[adata.obs["score_immune"] > 0].tolist()[: max(50, len(donors))]
    if len(receivers) < 100:
        receivers = adata.obs_names[adata.obs["score_tumor"] > 0].tolist()

    pred_path = out_dir / "merci_receiver_predictions.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path, index_col=0)
        if (out_dir / "merci_summary.png").exists():
            return pred
        is_rec = pred["prediction"] == "Receiver"
        rcm = pd.read_csv(out_dir / "merci_rcm_statistics.csv")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].boxplot(
            [pred.loc[is_rec, "Donor_MT_frac"], pred.loc[~is_rec, "Donor_MT_frac"]],
            tick_labels=["Receiver", "non-Receiver"],
        )
        axes[0].set_title("Donor MT fraction")
        ct_counts = pred.groupby(["cell_type", "prediction"]).size().unstack(fill_value=0)
        ct_counts.plot(kind="bar", ax=axes[1], rot=45)
        axes[1].set_title("Receiver calls by cell type")
        axes[2].plot(rcm["cutoff"], rcm["Rcm"], marker="o")
        axes[2].axhline(1.0, color="red", ls="--", lw=1)
        axes[2].set_title("MERCI Rcm significance")
        plt.tight_layout()
        fig.savefig(out_dir / "merci_summary.png", dpi=150)
        plt.close(fig)
        return pred

    dna_rank = donor_mt_signature_score(exp, donors, receivers, organism="mouse")
    rna_rank = merci_loo_mt_est(
        exp,
        receiver_cells=receivers,
        donor_cells=donors,
        organism="mouse",
        max_receivers=max_receivers,
        seed=seed,
    )
    rcm = cell_number_test(dna_rank, rna_rank, number_r=300, seed=seed)
    pred = merci_receiver_pre(dna_rank, rna_rank, top_rank=50)

    pred["graphclust"] = adata.obs.loc[pred.index, "graphclust"].values
    pred["cell_type"] = adata.obs.loc[pred.index, "cell_type"].values
    pred["Donor_MT_frac"] = rna_rank.loc[pred.index, "Donor_MT_frac"]
    pred["Donor_MT_ind"] = rna_rank.loc[pred.index, "Donor_MT_ind"]
    pred["DNA_rank"] = dna_rank.loc[pred.index, "MTvar_rank"]

    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_dir / "merci_receiver_predictions.csv")
    rna_rank.to_csv(out_dir / "merci_rna_ranks.csv")
    dna_rank.to_csv(out_dir / "merci_dna_proxy_ranks.csv")
    rcm.to_csv(out_dir / "merci_rcm_statistics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    is_rec = pred["prediction"] == "Receiver"
    axes[0].boxplot(
        [
            pred.loc[is_rec, "Donor_MT_frac"],
            pred.loc[~is_rec, "Donor_MT_frac"],
        ],
        tick_labels=["Receiver", "non-Receiver"],
    )
    axes[0].set_title("Donor MT fraction")
    axes[0].set_ylabel("Donor_MT_frac")

    ct_counts = pred.groupby(["cell_type", "prediction"]).size().unstack(fill_value=0)
    ct_counts.plot(kind="bar", ax=axes[1], rot=45)
    axes[1].set_title("Receiver calls by cell type")
    axes[1].set_xlabel("")

    axes[2].plot(rcm["cutoff"], rcm["Rcm"], marker="o")
    axes[2].axhline(1.0, color="red", ls="--", lw=1)
    axes[2].set_title("MERCI Rcm significance")
    axes[2].set_xlabel("Rank cutoff fraction")
    axes[2].set_ylabel("Rcm")
    plt.tight_layout()
    fig.savefig(out_dir / "merci_summary.png", dpi=150)
    plt.close(fig)
    return pred


def spatial_microniche_analysis(adata: sc.AnnData, pred: pd.DataFrame, out_dir: Path) -> None:
    sub = adata[pred.index].copy()
    sub.obs["mt_receiver"] = pred.loc[sub.obs_names, "prediction"].astype("category").values
    sub.obs["Donor_MT_frac"] = pred.loc[sub.obs_names, "Donor_MT_frac"].values
    sub.obs["cell_type"] = sub.obs["cell_type"].astype("category")
    rec_idx = sub.obs["mt_receiver"] == "Receiver"

    sq.gr.spatial_neighbors_delaunay(sub, spatial_key="spatial")
    sq.gr.nhood_enrichment(sub, cluster_key="mt_receiver")
    sq.pl.nhood_enrichment(sub, cluster_key="mt_receiver", show=False)
    plt.savefig(out_dir / "nhood_enrichment_mt_receiver.png", dpi=150, bbox_inches="tight")
    plt.close()

    sq.gr.spatial_neighbors_delaunay(sub, spatial_key="spatial")
    sq.gr.interaction_matrix(sub, cluster_key="cell_type")
    sq.pl.interaction_matrix(sub, cluster_key="cell_type", show=False)
    plt.savefig(out_dir / "celltype_interaction_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # immune proximity to receivers
    immune_mask = adata.obs["cell_type"].isin(["immune", "myeloid"]).to_numpy()
    if immune_mask.sum() > 0 and rec_idx.sum() > 0:
        nn = NearestNeighbors(n_neighbors=1).fit(adata.obsm["spatial"][immune_mask])
        tumor_rec = sub.obs_names[rec_idx].tolist()
        tumor_non = sub.obs_names[~rec_idx].tolist()
        dist_rec, _ = nn.kneighbors(adata[adata.obs_names.isin(tumor_rec)].obsm["spatial"])
        dist_non, _ = nn.kneighbors(adata[adata.obs_names.isin(tumor_non)].obsm["spatial"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.boxplot(
            [dist_rec.ravel(), dist_non.ravel()],
            tick_labels=["Receiver tumor", "non-Receiver tumor"],
        )
        ax.set_ylabel("Distance to nearest immune cell")
        ax.set_title("Immune proximity by MT receiver status")
        stat, p = stats.mannwhitneyu(dist_rec.ravel(), dist_non.ravel(), alternative="less")
        ax.text(0.5, 0.95, f"MW p={p:.2e}", transform=ax.transAxes, ha="center")
        fig.savefig(out_dir / "immune_proximity_receiver.png", dpi=150)
        plt.close(fig)

    sc.tl.rank_genes_groups(
        sub,
        groupby="mt_receiver",
        groups=["Receiver"],
        reference="non-Receiver",
        method="wilcoxon",
    )
    de = sc.get.rank_genes_groups_df(sub, group="Receiver")
    de.to_csv(out_dir / "de_receiver_vs_nonreceiver.csv", index=False)

    coords = sub.obsm["spatial"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sca = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=sub.obs["Donor_MT_frac"],
        s=1.5,
        cmap="viridis",
        vmax=np.quantile(sub.obs["Donor_MT_frac"], 0.99),
    )
    plt.colorbar(sca, ax=ax, label="Donor_MT_frac")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("Donor mitochondrial fraction")
    fig.savefig(out_dir / "spatial_donor_mt_frac.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = sub.obs["mt_receiver"].map({"Receiver": "#d62728", "non-Receiver": "#7f7f7f"})
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=1.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("MERCI receiver prediction")
    fig.savefig(out_dir / "spatial_mt_receivers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def impact_analysis(adata: sc.AnnData, pred: pd.DataFrame, out_dir: Path) -> dict:
    sub = adata[pred.index].copy()
    sub.obs["mt_receiver"] = pred.loc[sub.obs_names, "prediction"].values
    is_rec = sub.obs["mt_receiver"] == "Receiver"

    mt_genes = [g for g in sub.var_names if g.lower().startswith("mt-")]
    if sparse.issparse(sub.X):
        mt_sum = np.array(sub[:, mt_genes].X.sum(axis=1)).ravel()
        total_sum = np.array(sub.X.sum(axis=1)).ravel()
    else:
        mt_sum = sub[:, mt_genes].X.sum(axis=1)
        total_sum = sub.X.sum(axis=1)
    pct_mt = mt_sum / (total_sum + 1e-9)

    stress_genes = [g for g in ["Hspa1a", "Hspa1b", "Atf4", "Ddit3", "Bax"] if g in sub.var_names]
    if stress_genes:
        sc.tl.score_genes(sub, stress_genes, score_name="stress_score")

    summary = {
        "n_cells_analyzed": int(sub.n_obs),
        "n_receivers": int(is_rec.sum()),
        "receiver_fraction": float(is_rec.mean()),
        "pct_mt_receiver_median": float(np.median(pct_mt[is_rec])),
        "pct_mt_nonreceiver_median": float(np.median(pct_mt[~is_rec])),
        "receiver_by_celltype": pred.groupby(["cell_type", "prediction"]).size().unstack(fill_value=0).to_dict(),
    }
    if "stress_score" in sub.obs:
        summary["stress_receiver_median"] = float(sub.obs.loc[is_rec, "stress_score"].median())
        summary["stress_nonreceiver_median"] = float(sub.obs.loc[~is_rec, "stress_score"].median())
        _, p = stats.mannwhitneyu(
            sub.obs.loc[is_rec, "stress_score"],
            sub.obs.loc[~is_rec, "stress_score"],
            alternative="two-sided",
        )
        summary["stress_mannwhitney_p"] = float(p)

    with open(out_dir / "biological_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "subQ-1",
    )
    parser.add_argument("--max-receivers", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = args.data_dir / "results"
    fig = args.data_dir / "figures"
    proc = args.data_dir / "processed"
    for d in (out, fig, proc):
        d.mkdir(parents=True, exist_ok=True)

    adata = load_cell_adata(args.data_dir)
    h5ad_path = proc / "mc38_subq1_cells_annotated.h5ad"
    if h5ad_path.exists():
        adata = sc.read_h5ad(h5ad_path)
    else:
        sc.pp.filter_cells(adata, min_genes=50)
        sc.pp.filter_genes(adata, min_cells=10)
        adata = score_cell_types(adata)
        adata.write_h5ad(h5ad_path)

    pred = run_merci(adata, out, max_receivers=args.max_receivers, seed=args.seed)
    spatial_microniche_analysis(adata, pred, fig)
    summary = impact_analysis(adata, pred, out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
