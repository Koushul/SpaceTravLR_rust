#!/usr/bin/env python3
"""Human Visium HD (GSE280315 P1 CRC) mitochondrial transfer analysis via MERCI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse, stats
from sklearn.neighbors import NearestNeighbors

try:
    import squidpy as sq
except ImportError:
    sq = None

from merci_port import (
    cell_number_test,
    donor_mt_signature_score,
    merci_loo_mt_est,
    merci_mtvar_cal,
    merci_receiver_pre,
)

HUMAN_MARKERS = {
    "tumor": ["EPCAM", "KRT8", "KRT18", "CEACAM5", "MSLN"],
    "immune": ["PTPRC", "CD3D", "CD3E", "CD8A", "CD4", "NKG7", "MS4A1"],
    "myeloid": ["CD68", "CD14", "LYZ", "ITGAM", "CD163", "AIF1"],
    "stroma": ["COL1A1", "COL1A2", "DCN", "PDGFRA", "ACTA2"],
}

IMMUNE_LABELS = {
    "CD4 T cell", "CD8 T cell", "Mature B", "Plasma cell", "NK cell",
    "Treg", "Macrophage", "Neutrophil", "Mast cell", "Dendritic cell",
    "Monocyte", "Proliferating Macrophages",
}
TUMOR_LABEL_PREFIX = "Tumor"


def load_p1_bins(data_dir: Path) -> sc.AnnData:
    h5 = data_dir / "GSM8594567_P1CRC_filtered_feature_bc_matrix.h5"
    meta = data_dir / "GSM8594567_P1CRC_Metadata.parquet"
    pos = data_dir / "GSM8594567_P1CRC_tissue_positions.parquet"
    for p in (meta, pos):
        gz = Path(str(p) + ".gz")
        if not p.exists() and gz.exists():
            subprocess.run(["gunzip", "-kf", str(gz)], check=True)

    adata = sc.read_10x_h5(h5)
    adata.var_names_make_unique()
    meta_df = pd.read_parquet(meta).set_index("barcode")
    pos_df = pd.read_parquet(pos).set_index("barcode")

    common = adata.obs_names.intersection(meta_df.index).intersection(pos_df.index)
    adata = adata[common].copy()
    for col in meta_df.columns:
        adata.obs[col] = meta_df.loc[common, col].values
    adata.obs["in_tissue"] = pos_df.loc[common, "in_tissue"].astype(int).values
    adata.obsm["spatial"] = pos_df.loc[common, ["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy()
    return adata


def annotate_bins(adata: sc.AnnData) -> sc.AnnData:
    labels = adata.obs["DeconvolutionLabel1"].astype(str)
    ct = []
    for lab in labels:
        if lab.startswith(TUMOR_LABEL_PREFIX):
            ct.append("tumor")
        elif lab in IMMUNE_LABELS or "T cell" in lab or lab == "Macrophage":
            if "Macrophage" in lab or lab in {"Neutrophil", "Monocyte", "Dendritic cell", "Mast cell"}:
                ct.append("myeloid")
            else:
                ct.append("immune")
        elif lab in {"CAF", "Fibroblast", "Proliferating Fibroblast", "Pericytes", "Smooth muscle"}:
            ct.append("stroma")
        elif lab in {"Endothelial", "Enterocyte", "Goblet", "Adipocyte"}:
            ct.append("epithelial_other")
        else:
            ct.append("other")
    adata.obs["cell_type_deconv"] = pd.Categorical(ct)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    for ct_name, genes in HUMAN_MARKERS.items():
        present = [g for g in genes if g in adata.var_names]
        if present:
            sc.tl.score_genes(adata, present, score_name=f"score_{ct_name}")
    return adata


def filter_analysis_bins(adata: sc.AnnData, tumor_periphery_only: bool) -> sc.AnnData:
    mask = (
        (adata.obs["in_tissue"] == 1)
        & (adata.obs["DeconvolutionClass"] == "singlet")
        & (adata.obs["cell_type_deconv"].isin(["tumor", "immune", "myeloid"]))
    )
    if tumor_periphery_only:
        mask &= adata.obs["Periphery"].isin(["Tumor", "50 micron"])
    return adata[mask].copy()


def run_merci(
    adata: sc.AnnData,
    out_dir: Path,
    organism: str,
    max_receivers: int,
    seed: int,
    dna_rank: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if sparse.issparse(adata.X):
        exp = pd.DataFrame(adata.X.toarray().T, index=adata.var_names, columns=adata.obs_names)
    else:
        exp = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)

    donors = adata.obs_names[adata.obs["cell_type_deconv"].isin(["immune", "myeloid"])].tolist()
    receivers = adata.obs_names[adata.obs["cell_type_deconv"] == "tumor"].tolist()
    if len(donors) < 100:
        raise RuntimeError(f"Too few donor bins: {len(donors)}")
    if len(receivers) < 100:
        raise RuntimeError(f"Too few tumor bins: {len(receivers)}")

    if dna_rank is None:
        dna_rank = donor_mt_signature_score(exp, donors, receivers, organism=organism)

    rna_rank = merci_loo_mt_est(
        exp,
        receiver_cells=receivers,
        donor_cells=donors,
        organism=organism,
        max_receivers=max_receivers,
        seed=seed,
    )
    rcm = cell_number_test(dna_rank, rna_rank, number_r=300, seed=seed)
    pred = merci_receiver_pre(dna_rank, rna_rank, top_rank=50)

    pred["DeconvolutionLabel1"] = adata.obs.loc[pred.index, "DeconvolutionLabel1"].values
    pred["cell_type_deconv"] = adata.obs.loc[pred.index, "cell_type_deconv"].values
    pred["Periphery"] = adata.obs.loc[pred.index, "Periphery"].values
    pred["Donor_MT_frac"] = rna_rank.loc[pred.index, "Donor_MT_frac"]
    pred["Donor_MT_ind"] = rna_rank.loc[pred.index, "Donor_MT_ind"]
    pred["DNA_rank"] = dna_rank.loc[pred.index, "MTvar_rank"]
    pred["DMTvar_count"] = dna_rank.loc[pred.index].get("DMTvar_count", pd.Series(0, index=pred.index))

    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_dir / "merci_receiver_predictions.csv")
    rna_rank.to_csv(out_dir / "merci_rna_ranks.csv")
    dna_rank.to_csv(out_dir / "merci_dna_ranks.csv")
    rcm.to_csv(out_dir / "merci_rcm_statistics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    is_rec = pred["prediction"] == "Receiver"
    axes[0].boxplot(
        [pred.loc[is_rec, "Donor_MT_frac"], pred.loc[~is_rec, "Donor_MT_frac"]],
        tick_labels=["Receiver", "non-Receiver"],
    )
    axes[0].set_title("Donor MT fraction")
    ct_counts = pred.groupby(["cell_type_deconv", "prediction"]).size().unstack(fill_value=0)
    ct_counts.plot(kind="bar", ax=axes[1], rot=45)
    axes[1].set_title("Receiver calls by cell type")
    axes[2].plot(rcm["cutoff"], rcm["Rcm"], marker="o")
    axes[2].axhline(1.0, color="red", ls="--", lw=1)
    axes[2].set_title("MERCI Rcm significance")
    plt.tight_layout()
    fig.savefig(out_dir / "merci_summary.png", dpi=150)
    plt.close(fig)
    return pred


def spatial_analysis(adata: sc.AnnData, pred: pd.DataFrame, fig_dir: Path) -> None:
    sub = adata[pred.index].copy()
    sub.obs["mt_receiver"] = pred.loc[sub.obs_names, "prediction"].astype("category").values
    sub.obs["Donor_MT_frac"] = pred.loc[sub.obs_names, "Donor_MT_frac"].values
    sub.obs["cell_type_deconv"] = sub.obs["cell_type_deconv"].astype("category")
    rec_idx = sub.obs["mt_receiver"] == "Receiver"

    if sq is not None and sub.n_obs <= 100_000:
        sq.gr.spatial_neighbors_delaunay(sub, spatial_key="spatial")
        sq.gr.nhood_enrichment(sub, cluster_key="mt_receiver")
        sq.pl.nhood_enrichment(sub, cluster_key="mt_receiver", show=False)
        plt.savefig(fig_dir / "nhood_enrichment_mt_receiver.png", dpi=150, bbox_inches="tight")
        plt.close()

    immune_mask = adata.obs["cell_type_deconv"].isin(["immune", "myeloid"]).to_numpy()
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
        ax.set_ylabel("Distance to nearest immune/myeloid bin")
        ax.set_title("Immune proximity by MT receiver status")
        _, p = stats.mannwhitneyu(dist_rec.ravel(), dist_non.ravel(), alternative="less")
        ax.text(0.5, 0.95, f"MW p={p:.2e}", transform=ax.transAxes, ha="center")
        fig.savefig(fig_dir / "immune_proximity_receiver.png", dpi=150)
        plt.close(fig)

    sc.tl.rank_genes_groups(sub, groupby="mt_receiver", groups=["Receiver"], reference="non-Receiver", method="wilcoxon")
    de = sc.get.rank_genes_groups_df(sub, group="Receiver")
    de.to_csv(fig_dir.parent / "results" / "de_receiver_vs_nonreceiver.csv", index=False)

    coords = sub.obsm["spatial"]
    fig, ax = plt.subplots(figsize=(7, 6))
    sca = ax.scatter(
        coords[:, 0], coords[:, 1], c=sub.obs["Donor_MT_frac"], s=0.8, cmap="viridis",
        vmax=np.quantile(sub.obs["Donor_MT_frac"], 0.99),
    )
    plt.colorbar(sca, ax=ax, label="Donor_MT_frac")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("Donor mitochondrial fraction (P1 CRC Visium HD)")
    fig.savefig(fig_dir / "spatial_donor_mt_frac.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = sub.obs["mt_receiver"].map({"Receiver": "#d62728", "non-Receiver": "#7f7f7f"})
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=0.8)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("MERCI receiver prediction")
    fig.savefig(fig_dir / "spatial_mt_receivers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    palette = {"tumor": "#e41a1c", "immune": "#377eb8", "myeloid": "#4daf4a", "stroma": "#984ea3"}
    for ct, color in palette.items():
        m = sub.obs["cell_type_deconv"] == ct
        if m.sum():
            ax.scatter(coords[m, 0], coords[m, 1], c=color, s=0.5, label=ct, alpha=0.7)
    ax.legend(markerscale=4, frameon=False)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("Deconvolved cell types (tumor periphery)")
    fig.savefig(fig_dir / "spatial_cell_types.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def impact_analysis(adata: sc.AnnData, pred: pd.DataFrame, out_dir: Path) -> dict:
    sub = adata[pred.index].copy()
    sub.obs["mt_receiver"] = pred.loc[sub.obs_names, "prediction"].values
    is_rec = sub.obs["mt_receiver"] == "Receiver"

    mt_genes = [g for g in sub.var_names if g.startswith("MT-")]
    if sparse.issparse(sub.X):
        mt_sum = np.array(sub[:, mt_genes].X.sum(axis=1)).ravel()
        total_sum = np.array(sub.X.sum(axis=1)).ravel()
    else:
        mt_sum = sub[:, mt_genes].X.sum(axis=1)
        total_sum = sub.X.sum(axis=1)
    pct_mt = mt_sum / (total_sum + 1e-9)

    stress_genes = [g for g in ["HSPA1A", "HSPA1B", "ATF4", "DDIT3", "BAX"] if g in sub.var_names]
    if stress_genes:
        sc.tl.score_genes(sub, stress_genes, score_name="stress_score")

    summary = {
        "dataset": "GSE280315 P1 human CRC Visium HD (8um bins, singlet deconvolution)",
        "n_bins_analyzed": int(sub.n_obs),
        "n_receivers": int(is_rec.sum()),
        "receiver_fraction": float(is_rec.mean()),
        "pct_mt_receiver_median": float(np.median(pct_mt[is_rec])),
        "pct_mt_nonreceiver_median": float(np.median(pct_mt[~is_rec])),
        "receiver_by_deconv_label": pred.groupby(["DeconvolutionLabel1", "prediction"]).size().unstack(fill_value=0).to_dict(),
        "receiver_by_periphery": pred.groupby(["Periphery", "prediction"]).size().unstack(fill_value=0).to_dict(),
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


def load_mtsnp_dna_rank(merci_dir: Path, adata: sc.AnnData) -> pd.DataFrame | None:
    var_file = list(merci_dir.glob("*.MT_variants.txt"))
    cov_file = list(merci_dir.glob("*.Coverage_Cell.csv"))
    if not var_file or not cov_file:
        return None

    var_df = pd.read_csv(var_file[0], sep="\t")
    cov_df = pd.read_csv(cov_file[0], sep="\t")
    if "Cell" not in var_df.columns:
        return None

    pivot = var_df.pivot_table(index="ID", columns="Cell", values="AF", aggfunc="max")
    coverage = cov_df.set_index("Cell")["Covered_MT"] if "Covered_MT" in cov_df.columns else cov_df.iloc[:, 1]
    donors = adata.obs_names[adata.obs["cell_type_deconv"].isin(["immune", "myeloid"])].tolist()
    receivers = adata.obs_names[adata.obs["cell_type_deconv"] == "tumor"].tolist()
    return merci_mtvar_cal(pivot, coverage, donors, receivers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "P1-CRC")
    parser.add_argument("--max-receivers", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tumor-periphery-only", action="store_true", default=True)
    parser.add_argument("--use-mtsnp", action="store_true", help="Use MERCI-mtSNP DNA ranks if available")
    args = parser.parse_args()

    out = args.data_dir / "results"
    fig = args.data_dir / "figures"
    proc = args.data_dir / "processed"
    merci_mtsnp = args.data_dir / "merci_mtsnp"
    for d in (out, fig, proc):
        d.mkdir(parents=True, exist_ok=True)

    h5ad_path = proc / "p1_crc_bins_annotated.h5ad"
    if h5ad_path.exists():
        adata = sc.read_h5ad(h5ad_path)
    else:
        adata = load_p1_bins(args.data_dir)
        adata = annotate_bins(adata)
        adata = filter_analysis_bins(adata, tumor_periphery_only=args.tumor_periphery_only)
        for col in adata.obs.columns:
            adata.obs[col] = adata.obs[col].astype(str)
        adata.write_h5ad(h5ad_path)

    dna_rank = None
    if args.use_mtsnp:
        dna_rank = load_mtsnp_dna_rank(merci_mtsnp, adata)
        if dna_rank is not None:
            print("Using MERCI-mtSNP DNA ranks")

    pred = run_merci(adata, out, organism="human", max_receivers=args.max_receivers, seed=args.seed, dna_rank=dna_rank)
    spatial_analysis(adata, pred, fig)
    summary = impact_analysis(adata, pred, out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
