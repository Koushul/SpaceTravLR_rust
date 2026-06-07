"""AnnData loading, preprocessing, and spatial tensor cache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

from spatial_maps import create_spatial_features, spatial_maps_for_cluster, xyc2spatial_fast


@dataclass
class SpatialCache:
    obs_names: list[str]
    xy: np.ndarray
    clusters: np.ndarray
    cluster_labels: list[str]
    num_clusters: int
    spatial_maps: np.ndarray
    spatial_features: np.ndarray
    modulators: np.ndarray
    expr_log1p: np.ndarray
    gene_names: list[str]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            obs_names=np.array(self.obs_names, dtype=object),
            xy=self.xy,
            clusters=self.clusters,
            cluster_labels=np.array(self.cluster_labels, dtype=object),
            num_clusters=np.array([self.num_clusters]),
            spatial_maps=self.spatial_maps,
            spatial_features=self.spatial_features,
            modulators=self.modulators,
            expr_log1p=self.expr_log1p,
            gene_names=np.array(self.gene_names, dtype=object),
        )

    @classmethod
    def load(cls, path: Path) -> SpatialCache:
        z = np.load(path, allow_pickle=True)
        return cls(
            obs_names=list(z["obs_names"]),
            xy=z["xy"],
            clusters=z["clusters"].astype(np.int64),
            cluster_labels=list(z["cluster_labels"]),
            num_clusters=int(z["num_clusters"][0]),
            spatial_maps=z["spatial_maps"],
            spatial_features=z["spatial_features"],
            modulators=z["modulators"],
            expr_log1p=z["expr_log1p"],
            gene_names=list(z["gene_names"]),
        )


def preprocess_adata(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    force_genes: list[str] | None = None,
) -> ad.AnnData:
    adata = adata.copy()
    if sparse.issparse(adata.X):
        adata.X = adata.X.copy()
    sc.pp.filter_cells(adata, min_genes=50)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
    if force_genes:
        for g in force_genes:
            if g in adata.var_names:
                adata.var.loc[g, "highly_variable"] = True
    adata = adata[:, adata.var["highly_variable"]].copy()
    return adata


def encode_clusters(adata: ad.AnnData, cluster_col: str = "cell_type") -> tuple[np.ndarray, list[str]]:
    labels = adata.obs[cluster_col].astype(str).tolist()
    uniq = sorted(set(labels))
    label_to_id = {lab: i for i, lab in enumerate(uniq)}
    clusters = np.array([label_to_id[lab] for lab in labels], dtype=np.int64)
    return clusters, uniq


def build_spatial_cache(
    h5ad_path: Path,
    spatial_dim: int = 16,
    radius: float = 300.0,
    ego_center: bool = False,
    cluster_col: str = "cell_type",
    n_top_genes: int = 2000,
    force_genes: list[str] | None = None,
) -> SpatialCache:
    adata = ad.read_h5ad(h5ad_path)
    adata = preprocess_adata(adata, n_top_genes=n_top_genes, force_genes=force_genes)
    xy = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    clusters, cluster_labels = encode_clusters(adata, cluster_col=cluster_col)
    num_clusters = len(cluster_labels)

    spatial_maps = xyc2spatial_fast(
        xy, clusters, num_clusters, spatial_dim, spatial_dim, ego_center=ego_center
    )
    spatial_features = create_spatial_features(xy, clusters, num_clusters, radius=radius)
    modulators = spatial_features.copy()

    if sparse.issparse(adata.X):
        expr = adata.X.toarray()
    else:
        expr = np.asarray(adata.X)
    expr = expr.astype(np.float32)

    return SpatialCache(
        obs_names=list(adata.obs_names.astype(str)),
        xy=xy,
        clusters=clusters,
        cluster_labels=cluster_labels,
        num_clusters=num_clusters,
        spatial_maps=spatial_maps,
        spatial_features=spatial_features.astype(np.float32),
        modulators=modulators.astype(np.float32),
        expr_log1p=expr,
        gene_names=list(adata.var_names.astype(str)),
    )


def cluster_maps_for_cells(cache: SpatialCache) -> np.ndarray:
    """Own-cluster [N,1,H,W] maps for pretrain / inference."""
    n = len(cache.obs_names)
    h, w = cache.spatial_maps.shape[2], cache.spatial_maps.shape[3]
    out = np.zeros((n, 1, h, w), dtype=np.float32)
    for i in range(n):
        c = int(cache.clusters[i])
        out[i, 0] = cache.spatial_maps[i, c]
    return out


def gene_index(cache: SpatialCache, gene: str) -> int:
    try:
        return cache.gene_names.index(gene)
    except ValueError as exc:
        raise KeyError(f"gene {gene} not in cache") from exc


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
