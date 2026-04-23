"""Leiden clustering, Moran's I spatial coherence filter, niche signatures."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc

log = logging.getLogger(__name__)


def cluster_embeddings(
    z: np.ndarray,
    resolutions: list[float] = [0.3, 0.5, 1.0, 1.5],
    n_neighbors: int = 15,
) -> dict[float, np.ndarray]:
    """
    Run Leiden clustering on the cell embeddings at multiple resolutions.

    Parameters
    ----------
    z : [N, D] embedding matrix
    resolutions : list of Leiden resolution values
    n_neighbors : number of neighbors for the kNN graph

    Returns
    -------
    {resolution: array of cluster labels (str)}
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata_emb = sc.AnnData(z)
        sc.pp.neighbors(adata_emb, use_rep="X", n_neighbors=n_neighbors)
        results = {}
        for r in resolutions:
            sc.tl.leiden(adata_emb, resolution=r, key_added=f"leiden_{r}")
            results[r] = adata_emb.obs[f"leiden_{r}"].values.astype(str)
    return results


def morans_i(
    labels: np.ndarray,
    spatial_coords: np.ndarray,
    k: int = 6,
) -> float:
    """
    Compute Moran's I for a categorical labelling mapped to integer codes.
    Uses a binary kNN spatial weight matrix.
    """
    from sklearn.neighbors import NearestNeighbors

    codes = pd.Categorical(labels).codes.astype(float)
    n = len(codes)

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)
    indices = indices[:, 1:]  # drop self

    W = np.zeros((n, n))
    for i, row in enumerate(indices):
        W[i, row] = 1.0
    W_sum = W.sum()

    z_centered = codes - codes.mean()
    numerator = n / W_sum * (W * np.outer(z_centered, z_centered)).sum()
    denominator = (z_centered ** 2).sum()
    return float(numerator / (denominator + 1e-12))


def filter_spatial_clusters(
    labels: np.ndarray,
    spatial_coords: np.ndarray,
    min_morans_i: float = 0.1,
    k: int = 6,
) -> np.ndarray:
    """
    Remap clusters with Moran's I < min_morans_i to "non_spatial".
    Returns updated label array.
    """
    unique_labels = np.unique(labels)
    keep = set()
    for lbl in unique_labels:
        mask = (labels == lbl).astype(float)
        I = morans_i(mask, spatial_coords, k=k)
        if I >= min_morans_i:
            keep.add(lbl)
        else:
            log.debug(f"Cluster {lbl}: Moran's I={I:.3f} < {min_morans_i}, dropping")

    updated = np.where(np.isin(labels, list(keep)), labels, "non_spatial")
    return updated


def niche_signatures(
    labels: np.ndarray,
    gene_betas: list,
    mod_vocab: dict[str, int],
    top_k: int = 20,
) -> dict:
    """
    For each niche cluster, rank modulators by mean absolute beta across all genes.

    Parameters
    ----------
    labels : [N] cluster labels
    gene_betas : list of GeneBetadata
    mod_vocab : modulator name → index
    top_k : number of top modulators to report

    Returns
    -------
    dict: {cluster_id: {"top_modulators": [...], "gene_breakdown": {...}}}
    """
    idx_to_mod = {v: k for k, v in mod_vocab.items()}
    n_mods = len(mod_vocab)
    unique_clusters = sorted(set(labels))

    n_cells = len(labels)
    all_abs_betas = np.zeros((n_cells, n_mods), dtype=np.float32)

    for gb in gene_betas:
        mod_idx = gb.mod_indices[0].numpy()    # [M_g]
        abs_betas = gb.beta_values.abs().numpy()  # [N, M_g]
        all_abs_betas[:, mod_idx] += abs_betas

    signatures = {}
    for cluster in unique_clusters:
        mask = labels == cluster
        mean_betas = all_abs_betas[mask].mean(axis=0)  # [n_mods]
        top_indices = np.argsort(mean_betas)[::-1][:top_k]
        top_mods = [
            {"modulator": idx_to_mod[i], "mean_abs_beta": float(mean_betas[i])}
            for i in top_indices
        ]

        gene_breakdown = {}
        for gb in gene_betas:
            mod_idx = gb.mod_indices[0].numpy()
            abs_betas = gb.beta_values.abs().numpy()
            gene_mean = abs_betas[mask].mean(axis=0)  # [M_g]
            top_gene_idx = np.argsort(gene_mean)[::-1][:5]
            gene_breakdown[gb.gene_name] = [
                {"modulator": idx_to_mod[mod_idx[i]], "mean_abs_beta": float(gene_mean[i])}
                for i in top_gene_idx
            ]

        signatures[str(cluster)] = {
            "n_cells": int(mask.sum()),
            "top_modulators": top_mods,
            "gene_breakdown": gene_breakdown,
        }

    return signatures


def run_clustering(
    z: np.ndarray,
    cell_ids: list[str],
    gene_betas: list,
    mod_vocab: dict[str, int],
    spatial_coords: Optional[np.ndarray] = None,
    resolutions: list[float] = [0.3, 0.5, 1.0, 1.5],
    min_morans_i: float = 0.1,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Full clustering pipeline: Leiden → optional Moran's I filter → signatures.

    Returns
    -------
    DataFrame with CellID and niche_id columns for each resolution
    """
    cluster_results = cluster_embeddings(z, resolutions=resolutions)

    label_df = pd.DataFrame({"CellID": cell_ids})
    for r, labels in cluster_results.items():
        if spatial_coords is not None:
            labels = filter_spatial_clusters(labels, spatial_coords, min_morans_i)
        label_df[f"niche_{r}"] = labels

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        label_df.to_parquet(out / "niche_labels.parquet", index=False)

        for r, labels in cluster_results.items():
            sigs = niche_signatures(labels, gene_betas, mod_vocab)
            with open(out / f"niche_signatures_r{r}.json", "w") as f:
                json.dump(sigs, f, indent=2)

        log.info(f"Clustering results saved to {output_dir}")

    return label_df
