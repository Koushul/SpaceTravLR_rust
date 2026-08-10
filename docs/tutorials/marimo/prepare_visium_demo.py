#!/usr/bin/env python3
"""Prepare public/visium_demo assets for the Squidpy marimo tutorial."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scanpy as sc
import squidpy as sq
from anndata import AnnData

OUT = Path(__file__).resolve().parent / "public" / "visium_demo"
N_NEIGHBORS_OPTS = [5, 15, 30, 50]
MIN_DIST_OPTS = [0.05, 0.1, 0.3, 0.5, 0.8]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    adata = sq.datasets.visium_hne_adata()
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata)

    cluster_key = "cluster" if "cluster" in adata.obs else "leiden"
    labels = np.asarray(
        [str(x) for x in adata.obs[cluster_key].astype(str).to_list()]
    )
    categories = [
        str(c) for c in adata.obs[cluster_key].astype("category").cat.categories
    ]
    color_key = f"{cluster_key}_colors"
    palette = {
        cat: str(adata.uns[color_key][i % len(adata.uns[color_key])])
        for i, cat in enumerate(categories)
    }

    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    pca = np.asarray(adata.obsm["X_pca"][:, :30], dtype=np.float64)
    np.save(OUT / "spatial.npy", spatial)
    np.save(OUT / "pca.npy", pca)
    maxlen = max(len(s) for s in labels.tolist())
    np.save(OUT / "labels.npy", np.array(labels.tolist(), dtype=f"<U{maxlen}"))
    (OUT / "labels.json").write_text(json.dumps(labels.tolist()))

    meta = {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "cluster_key": cluster_key,
        "categories": categories,
        "palette": palette,
        "gene_example": "Sox2"
        if "Sox2" in adata.var_names
        else str(adata.var_names[0]),
        "n_neighbors_opts": N_NEIGHBORS_OPTS,
        "min_dist_opts": MIN_DIST_OPTS,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))

    umap_dir = OUT / "umap"
    umap_dir.mkdir(exist_ok=True)
    for nn in N_NEIGHBORS_OPTS:
        for md in MIN_DIST_OPTS:
            ad = AnnData(np.zeros((pca.shape[0], 1)))
            ad.obsm["X_pca"] = pca
            sc.pp.neighbors(ad, n_neighbors=nn, use_rep="X_pca")
            sc.tl.umap(ad, min_dist=md)
            np.save(
                umap_dir / f"nn{nn}_md{md}.npy",
                np.asarray(ad.obsm["X_umap"], dtype=np.float64),
            )
            print(f"saved nn={nn} min_dist={md}")

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
