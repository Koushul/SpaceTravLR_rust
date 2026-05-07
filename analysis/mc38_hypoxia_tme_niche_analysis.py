#!/usr/bin/env python3
"""
Spatial TME niche analysis for MC38 hypoxia / immunotherapy-relevant designs.

Default data root: /ix1/ylee/shared/MC38_Hypoxia_001/ (override with --data-root).
When that path is unavailable, pass --fallback-h5ad for a demo AnnData.

Implements an externally constrained niche view:
- O2 / hypoxia program as an environmental axis (gene-set score).
- CD8 effector program as a readout of allowed cytotoxic function.
- Cellular neighborhood context via kNN on spatial coordinates.

Outputs figures under --out-dir illustrating niche "rules", spatial maps, and (if
present in obs) CRISPR / perturbation stratification toward correlation→causation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import patches
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MOUSE_HYPOXIA = [
    "Slc2a1",
    "Vegfa",
    "Bnip3",
    "Bnip3l",
    "Pgk1",
    "Ldha",
    "Pfkfb3",
    "Eno1",
    "Hk2",
    "Mif",
    "Egln3",
    "Slc2a3",
    "Pim1",
    "Gpi1",
    "Tpi1",
]
MOUSE_CD8_EFFECTOR = [
    "Cd8a",
    "Cd8b1",
    "Gzma",
    "Gzmb",
    "Prf1",
    "Ifng",
    "Fasl",
    "Klrd1",
    "Klrk1",
    "Nkg7",
]
MOUSE_EXHAUSTION = ["Pdcd1", "Ctla4", "Lag3", "Havcr2", "Tigit", "Cd244a"]
MOUSE_M2_LIKE = ["Arg1", "Mrc1", "Il10", "Tgfb1", "Ccl22"]

HUMAN_HYPOXIA = [
    "SLC2A1",
    "VEGFA",
    "BNIP3",
    "BNIP3L",
    "PGK1",
    "LDHA",
    "PFKFB3",
    "ENO1",
    "HK2",
    "MIF",
    "EGLN3",
    "SLC2A3",
    "PIM1",
    "GPI1",
    "TPI1",
]
HUMAN_CD8 = [
    "CD8A",
    "CD8B",
    "GZMA",
    "GZMB",
    "PRF1",
    "IFNG",
    "FASLG",
    "KLRD1",
    "KLRK1",
    "NKG7",
]
HUMAN_EXHAUSTION = ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "CD244"]
HUMAN_M2 = ["ARG1", "MRC1", "IL10", "TGFB1", "CCL22"]


def _detect_species(var_names: pd.Index) -> str:
    upper_frac = sum(1 for g in var_names[:500] if str(g).isupper()) / max(
        min(500, len(var_names)), 1
    )
    return "human" if upper_frac > 0.6 else "mouse"


def _subset_present(genes: list[str], var_names: np.ndarray) -> list[str]:
    vn = set(var_names.astype(str))
    return [g for g in genes if g in vn]


def discover_h5ad(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.h5ad"))


def load_concatenate(paths: list[Path]) -> ad.AnnData:
    ads = [sc.read_h5ad(p) for p in paths]
    if len(ads) == 1:
        a = ads[0]
    else:
        for i, a in enumerate(ads):
            a.obs["library_id"] = paths[i].stem
        a = ad.concat(ads, join="outer", label="batch", keys=[p.stem for p in paths])
    return a


def ensure_spatial(adata: ad.AnnData) -> np.ndarray:
    if "spatial" in adata.obsm:
        xy = np.asarray(adata.obsm["spatial"])
        if xy.shape[1] >= 2:
            return xy[:, :2]
    for k in adata.obsm:
        if "spatial" in k.lower() or "x_umap" in k.lower():
            xy = np.asarray(adata.obsm[k])
            if xy.shape[1] >= 2:
                return xy[:, :2]
    raise ValueError(
        "No spatial coordinates found (expected obsm['spatial'] or similar)."
    )


def preprocess(adata: ad.AnnData) -> None:
    x = adata.X
    if hasattr(x, "data"):
        mx = float(np.max(x.data)) if x.nnz else 0.0
    else:
        mx = float(np.max(x))
    if mx > 50:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)


def score_signature(
    adata: ad.AnnData, name: str, genes: list[str], species: str
) -> tuple[list[str], int]:
    present = _subset_present(genes, adata.var_names.values)
    if len(present) < 3:
        logger.warning(
            "Signature %s: only %d genes present — scores may be noisy.",
            name,
            len(present),
        )
    if len(present) == 0:
        adata.obs[name] = 0.0
        return [], 0
    sc.tl.score_genes(adata, gene_list=present, score_name=name, use_raw=False)
    return present, len(present)


def knn_local_mean(values: np.ndarray, xy: np.ndarray, k: int = 15) -> np.ndarray:
    tree = cKDTree(xy)
    k = min(k, len(xy))
    _, idx = tree.query(xy, k=k)
    neigh = values[idx]
    return np.nanmean(neigh, axis=1)


def assign_niches(
    hyp: np.ndarray,
    cd8: np.ndarray,
    loc_cd8: np.ndarray,
    hyp_q_hi: float = 0.75,
    cd8_q_lo: float = 0.25,
) -> pd.Series:
    h_thr = np.quantile(hyp, hyp_q_hi)
    c_thr = np.quantile(cd8, cd8_q_lo)
    lc_thr = np.quantile(loc_cd8, cd8_q_lo)
    labels = np.full(len(hyp), "intermediate", dtype=object)
    labels[(hyp >= h_thr) & (cd8 <= c_thr)] = "hypoxic_CD8_low"
    labels[(hyp >= h_thr) & (loc_cd8 <= lc_thr)] = "hypoxic_low_neighbor_CD8"
    labels[(hyp < np.median(hyp)) & (cd8 >= np.quantile(cd8, 0.5))] = "permissive_high_CD8"
    return pd.Series(labels, index=np.arange(len(hyp)))


def crispr_columns(obs: pd.DataFrame) -> list[str]:
    pat = re.compile(
        r"(guide|grna|perturb|crispr|ko|knock|target|sgRNA)", re.IGNORECASE
    )
    return [c for c in obs.columns if pat.search(str(c))]


def plot_spatial_grid(
    xy: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    titles: tuple[str, str, str],
    out: Path,
    c_cat: np.ndarray | None = None,
    cat_labels: np.ndarray | None = None,
    third_cmap: str = "coolwarm",
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, arr, title, cmap in zip(
        axes[:3],
        (a, b, c),
        titles,
        ("viridis", "viridis", third_cmap),
        strict=False,
    ):
        sca = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=arr,
            s=4,
            cmap=cmap,
            linewidths=0,
            rasterized=True,
        )
        plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.invert_yaxis()
    if c_cat is not None and cat_labels is not None:
        labs = np.asarray(cat_labels).astype(str)
        uniques = list(dict.fromkeys(labs.tolist()))
        cmap = plt.colormaps["tab10"].resampled(max(len(uniques), 1))
        norm = Normalize(vmin=0, vmax=max(len(uniques) - 1, 0))
        idx_map = {u: i for i, u in enumerate(uniques)}
        col_idx = np.array([idx_map[x] for x in labs])
        axes[2].clear()
        sca = axes[2].scatter(
            xy[:, 0],
            xy[:, 1],
            c=col_idx,
            cmap=cmap,
            norm=norm,
            s=4,
            linewidths=0,
            rasterized=True,
        )
        cbar = plt.colorbar(
            sca, ax=axes[2], fraction=0.046, pad=0.04, ticks=range(len(uniques))
        )
        cbar.ax.set_yticklabels(uniques)
        axes[2].set_aspect("equal")
        axes[2].set_title(titles[2])
        axes[2].invert_yaxis()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_hex_hyp_cd8(hyp: np.ndarray, cd8: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    hb = ax.hexbin(hyp, cd8, gridsize=40, cmap="Blues", mincnt=1)
    plt.colorbar(hb, ax=ax, label="spots / bins")
    ax.set_xlabel("Hypoxia score")
    ax.set_ylabel("CD8 effector score")
    r, p = stats.spearmanr(hyp, cd8)
    ax.text(
        0.02,
        0.98,
        f"Spearman ρ = {r:.3f}\np = {p:.2e}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_niche_stats(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    order = df.groupby("niche").size().sort_values(ascending=False).index.tolist()
    sub = df[df["niche"].isin(order)]
    sns_df = sub.copy()

    sns.boxplot(
        data=sns_df,
        x="niche",
        y="hypoxia_score",
        order=order,
        ax=axes[0],
        hue="niche",
        palette="pastel",
        legend=False,
    )
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Hypoxia score")
    sns.boxplot(
        data=sns_df,
        x="niche",
        y="cd8_effector_score",
        order=order,
        ax=axes[1],
        hue="niche",
        palette="pastel",
        legend=False,
    )
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylabel("CD8 effector score")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_celltype_niche_heatmap(df: pd.DataFrame, out: Path) -> None:
    if "cell_type" not in df.columns:
        return
    ct = pd.crosstab(df["cell_type"], df["niche"], normalize="columns")
    fig_h = max(4.0, min(16.0, 0.28 * len(ct)))
    fig, ax = plt.subplots(figsize=(8, fig_h), constrained_layout=True)
    sns.heatmap(ct, cmap="Blues", ax=ax, linewidths=0.5, vmin=0, vmax=min(1.0, ct.max().max()))
    ax.set_title("Cell-type fraction within niche (columns sum to 1)")
    ax.set_xlabel("niche")
    ax.set_ylabel("cell_type")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_local_vs_focal(
    hyp: np.ndarray, loc_cd8: np.ndarray, out: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.scatter(hyp, loc_cd8, s=3, alpha=0.35, c="steelblue", rasterized=True)
    ax.set_xlabel("Focal hypoxia score")
    ax.set_ylabel("Neighborhood mean CD8 score (kNN)")
    r, p = stats.spearmanr(hyp, loc_cd8)
    ax.text(
        0.02,
        0.98,
        f"Spearman ρ = {r:.3f}\np = {p:.2e}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_workflow_schematic(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    stages = [
        "Description\n(spatial maps)",
        "Rules\n(niches / constraints)",
        "Prediction\n(local coupling)",
        "CRISPR test\n(cause)",
        "Modification\n(rescue)",
    ]
    xs = np.linspace(0.5, 9.5, len(stages))
    for i, (x, s) in enumerate(zip(xs, stages, strict=False)):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x - 0.55, 0.55),
                1.1,
                1.0,
                boxstyle="round,pad=0.02",
                edgecolor="black",
                facecolor="#eef6ff",
            )
        )
        ax.text(x, 1.05, s, ha="center", va="center", fontsize=9)
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.55, 1.05),
                xytext=(x + 0.55, 1.05),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
            )
    ax.set_title("From spatial description to causal modification (spatial CRISPR loop)")
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_crispr_if_any(
    adata: ad.AnnData, score_col: str, cols: list[str], out_dir: Path
) -> None:
    for col in cols:
        sub = adata.obs.dropna(subset=[col])
        if sub[col].nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        sns.boxplot(data=sub, x=col, y=score_col, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
        ax.set_ylabel(score_col)
        safe = re.sub(r"[^\w\-]+", "_", col)[:80]
        fig.savefig(out_dir / f"crispr_{safe}_vs_{score_col}.png", dpi=200)
        plt.close(fig)


def run(
    data_root: Path | None,
    fallback_h5ad: Path | None,
    out_dir: Path,
    knn_k: int,
    save_adata: bool,
) -> dict[str, Any]:
    sns.set_theme(style="whitegrid", context="notebook")

    paths: list[Path] = []
    if data_root and data_root.is_dir():
        paths = discover_h5ad(data_root)
    if not paths and fallback_h5ad and fallback_h5ad.is_file():
        logger.warning(
            "Using fallback AnnData %s (set data root to MC38 shared path on your cluster).",
            fallback_h5ad,
        )
        paths = [fallback_h5ad]
    if not paths:
        raise FileNotFoundError(
            f"No .h5ad under {data_root}. Provide --fallback-h5ad or mount MC38 data."
        )

    adata = load_concatenate(paths)
    xy = ensure_spatial(adata)
    preprocess(adata)
    species = _detect_species(adata.var_names)
    if species == "human":
        hyp_genes, cd8_genes = HUMAN_HYPOXIA, HUMAN_CD8
        exh_genes, m2_genes = HUMAN_EXHAUSTION, HUMAN_M2
    else:
        hyp_genes, cd8_genes = MOUSE_HYPOXIA, MOUSE_CD8_EFFECTOR
        exh_genes, m2_genes = MOUSE_EXHAUSTION, MOUSE_M2_LIKE

    hyp_used, _ = score_signature(adata, "hypoxia_score", hyp_genes, species)
    cd8_used, _ = score_signature(adata, "cd8_effector_score", cd8_genes, species)
    score_signature(adata, "exhaustion_score", exh_genes, species)
    score_signature(adata, "m2_like_score", m2_genes, species)

    hyp = np.asarray(adata.obs["hypoxia_score"].values, dtype=float)
    cd8 = np.asarray(adata.obs["cd8_effector_score"].values, dtype=float)
    loc_cd8 = knn_local_mean(cd8, xy, k=knn_k)
    loc_m2 = knn_local_mean(
        np.asarray(adata.obs["m2_like_score"].values, dtype=float), xy, k=knn_k
    )
    adata.obs["local_cd8_knn"] = loc_cd8
    adata.obs["local_m2_knn"] = loc_m2

    niches = assign_niches(hyp, cd8, loc_cd8)
    adata.obs["niche_label"] = niches.values

    def _safe_z(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        s = np.nanstd(x)
        if s == 0 or np.isnan(s):
            return np.zeros_like(x)
        return stats.zscore(x, nan_policy="omit")

    constraint_z = _safe_z(hyp) - _safe_z(cd8)
    adata.obs["constraint_index"] = constraint_z

    out_dir.mkdir(parents=True, exist_ok=True)

    niche_df = pd.DataFrame(
        {
            "hypoxia_score": hyp,
            "cd8_effector_score": cd8,
            "local_cd8_knn": loc_cd8,
            "niche": niches.values,
            "constraint_index": constraint_z,
        }
    )
    if "cell_type" in adata.obs.columns:
        niche_df["cell_type"] = adata.obs["cell_type"].values

    niche_df.to_csv(out_dir / "niche_table.csv", index=False)

    labels = adata.obs["niche_label"].astype(str).values
    plot_spatial_grid(
        xy,
        hyp,
        cd8,
        constraint_z,
        ("Hypoxia score", "CD8 effector score", "Constraint index (hypoxia↑ − CD8↑)"),
        out_dir / "fig1_spatial_scores.png",
        third_cmap="coolwarm",
    )
    plot_spatial_grid(
        xy,
        hyp,
        cd8,
        constraint_z,
        ("Hypoxia score", "CD8 effector score", "Niche label"),
        out_dir / "fig1b_spatial_niches.png",
        c_cat=np.zeros(1),
        cat_labels=labels,
    )

    plot_hex_hyp_cd8(hyp, cd8, out_dir / "fig2_hypoxia_vs_cd8_hexbin.png")
    plot_niche_stats(niche_df, out_dir / "fig3_niche_boxplots.png")
    plot_celltype_niche_heatmap(niche_df, out_dir / "fig3b_celltype_by_niche.png")
    plot_local_vs_focal(hyp, loc_cd8, out_dir / "fig4_hypoxia_vs_local_cd8.png")

    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    sca = ax.scatter(
        loc_m2,
        loc_cd8,
        c=hyp,
        cmap="magma",
        s=5,
        alpha=0.6,
        linewidths=0,
        rasterized=True,
    )
    plt.colorbar(sca, ax=ax, label="Focal hypoxia score")
    ax.set_xlabel("Neighborhood myeloid/M2-like score (kNN)")
    ax.set_ylabel("Neighborhood CD8 effector score (kNN)")
    fig.savefig(out_dir / "fig5_neighborhood_m2_vs_cd8_colored_hypoxia.png", dpi=200)
    plt.close(fig)

    plot_workflow_schematic(out_dir / "fig6_workflow_description_to_modification.png")

    ccols = crispr_columns(adata.obs)
    if ccols:
        plot_crispr_if_any(adata, "cd8_effector_score", ccols, out_dir)
        plot_crispr_if_any(adata, "hypoxia_score", ccols, out_dir)
    else:
        logger.info(
            "No CRISPR/perturbation columns in obs; workflow figure only. "
            "Add guide/target columns to enable perturbation panels."
        )

    summary = {
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "species_guess": species,
        "h5ad_paths": [str(p) for p in paths],
        "hypoxia_genes_used": hyp_used,
        "cd8_genes_used": cd8_used,
        "knn_k": knn_k,
        "crispr_columns_found": ccols,
        "spearman_hypoxia_cd8": float(stats.spearmanr(hyp, cd8)[0]),
        "spearman_hypoxia_local_cd8": float(stats.spearmanr(hyp, loc_cd8)[0]),
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if save_adata:
        adata.write(out_dir / "processed_with_niches.h5ad")
    return summary


def main() -> None:
    default_root = os.environ.get(
        "MC38_HYPOXIA_ROOT", "/ix1/ylee/shared/MC38_Hypoxia_001"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(default_root),
        help="Directory containing MC38 hypoxia spatial .h5ad files (recursive).",
    )
    parser.add_argument(
        "--fallback-h5ad",
        type=Path,
        default=Path("/workspace/data/h5ad/Xenium_mouse_skin.h5ad"),
        help="Used when --data-root has no h5ad files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/workspace/figures/mc38_hypoxia_tme"),
        help="Output directory for figures and tables.",
    )
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument(
        "--save-adata",
        action="store_true",
        help="Write processed AnnData with niche columns (can be large).",
    )
    args = parser.parse_args()

    try:
        summary = run(
            args.data_root,
            args.fallback_h5ad,
            args.out_dir,
            args.knn_k,
            args.save_adata,
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    logger.info("Done. summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
