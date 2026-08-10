#!/usr/bin/env python3
"""Prepare public/slideseq_banksy assets: Slide-seq V2 + BANKSY + UMAP grid."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import scanpy as sc
import squidpy as sq
from banksy.initialize_banksy import initialize_banksy
from banksy.run_banksy import generate_banksy_matrix, pca_umap, run_Leiden_partition
from anndata import AnnData

OUT = Path(__file__).resolve().parent / "public" / "slideseq_banksy"
N_CELLS = 5000
SEED = 0
LAMBDA_OPTS = [0.1, 0.2, 0.4]
RESOLUTION = 0.6
N_NEIGHBORS_OPTS = [5, 15, 30, 50]
MIN_DIST_OPTS = [0.1, 0.3, 0.5]
PCA_DIMS = 20


def _tab20_palette(n: int) -> dict[str, str]:
    cmap = matplotlib.colormaps.get_cmap("tab20")
    return {
        str(i): "#%02x%02x%02x" % tuple(int(255 * c) for c in cmap(i % 20)[:3])
        for i in range(n)
    }


def run_banksy(adata: AnnData, lam: float):
    banksy_dict = initialize_banksy(
        adata,
        coord_keys=("x", "y", "spatial"),
        num_neighbours=15,
        nbr_weight_decay="scaled_gaussian",
        max_m=1,
        plt_edge_hist=False,
        plt_nbr_weights=False,
        plt_agf_angles=False,
        plt_theta=False,
    )
    banksy_dict, _ = generate_banksy_matrix(adata, banksy_dict, [lam], max_m=1)
    pca_umap(banksy_dict, pca_dims=[PCA_DIMS], plt_remaining_var=False)
    results_df, _ = run_Leiden_partition(
        banksy_dict=banksy_dict,
        resolutions=[RESOLUTION],
        num_nn=50,
        num_iterations=-1,
        partition_seed=1234,
        match_labels=False,
    )
    row = results_df.iloc[0]
    labels = np.asarray(row["labels"].dense, dtype=np.int32)
    pca = np.asarray(
        banksy_dict["scaled_gaussian"][lam]["adata"].obsm[f"reduced_pc_{PCA_DIMS}"],
        dtype=np.float64,
    )
    return labels, pca


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "umap").mkdir(exist_ok=True)
    (OUT / "banksy").mkdir(exist_ok=True)

    print("Loading squidpy slideseqv2 (mouse hippocampus)...")
    adata = sq.datasets.slideseqv2()
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(adata.n_obs, size=min(N_CELLS, adata.n_obs), replace=False))
    adata = adata[idx].copy()
    print(f"subset {adata.n_obs} × {adata.n_vars}")

    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    pub_labels = np.asarray(adata.obs["cluster"].astype(str).to_list())
    pub_categories = sorted(set(pub_labels))
    pub_palette = {
        cat: str(adata.uns["cluster_colors"][i % len(adata.uns["cluster_colors"])])
        for i, cat in enumerate(list(adata.obs["cluster"].astype("category").cat.categories))
    }
    # keep only categories present in subset
    pub_palette = {c: pub_palette[c] for c in pub_categories if c in pub_palette}
    for c in pub_categories:
        pub_palette.setdefault(c, "#999999")

    maxlen = max(len(s) for s in pub_labels)
    np.save(OUT / "spatial.npy", spatial)
    np.save(OUT / "published_labels.npy", np.array(pub_labels, dtype=f"<U{maxlen}"))

    sc.pp.scale(adata, max_value=10)

    banksy_info = {}
    default_lam = 0.2
    default_pca = None
    for lam in LAMBDA_OPTS:
        print(f"Running BANKSY λ={lam} ...")
        labels, pca = run_banksy(adata, lam)
        np.save(OUT / "banksy" / f"labels_lambda{lam}.npy", labels)
        np.save(OUT / "banksy" / f"pca_lambda{lam}.npy", pca)
        n_clust = int(len(np.unique(labels)))
        banksy_info[str(lam)] = {
            "n_clusters": n_clust,
            "palette": _tab20_palette(n_clust),
        }
        if lam == default_lam:
            default_pca = pca
        print(f"  clusters={n_clust}")

    assert default_pca is not None
    print("Computing UMAP grids on BANKSY PCA for each λ...")
    for lam in LAMBDA_OPTS:
        pca = np.load(OUT / "banksy" / f"pca_lambda{lam}.npy")
        udir = OUT / "umap" / f"lambda{lam}"
        udir.mkdir(parents=True, exist_ok=True)
        for nn in N_NEIGHBORS_OPTS:
            for md in MIN_DIST_OPTS:
                ad = AnnData(np.zeros((pca.shape[0], 1)))
                ad.obsm["X_pca"] = pca
                sc.pp.neighbors(ad, n_neighbors=nn, use_rep="X_pca")
                sc.tl.umap(ad, min_dist=md)
                np.save(
                    udir / f"nn{nn}_md{md}.npy",
                    np.asarray(ad.obsm["X_umap"], dtype=np.float64),
                )
                print(f"  umap λ={lam} nn={nn} min_dist={md}")

    meta = {
        "dataset": "squidpy.datasets.slideseqv2",
        "dataset_title": "Slide-seq V2 mouse hippocampus",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_obs_full": 41786,
        "cluster_key_published": "cluster",
        "cluster_key_banksy": "banksy",
        "published_categories": pub_categories,
        "published_palette": pub_palette,
        "lambda_opts": LAMBDA_OPTS,
        "default_lambda": default_lam,
        "banksy_resolution": RESOLUTION,
        "banksy": banksy_info,
        "n_neighbors_opts": N_NEIGHBORS_OPTS,
        "min_dist_opts": MIN_DIST_OPTS,
        "gene_example": "Snap25"
        if "Snap25" in adata.var_names
        else str(adata.var_names[0]),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
