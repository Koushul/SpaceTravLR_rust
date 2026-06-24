"""Spatial niche DEG utilities adapted for SPAC-seq subQ validation.

Adapted from spatial perturbation neighbor analysis: finds bystander cells near
sgP vs NTC sources, runs Wilcoxon DE, and compares experimental log2FC with
SpaceTravLR predicted pseudobulk deltas.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import sparse, stats
from scipy.spatial import KDTree, cKDTree

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None


CCC_PATHWAYS = {
    "MHC-II presentation": ["H2-Aa", "H2-Ab1", "Cd74", "Ciita", "Cd74"],
    "MHC-I / cross-presentation": ["H2-K1", "H2-D1", "B2m", "Tap1", "Tap2"],
    "T cell recruitment (CXCL)": ["Cxcl9", "Cxcl10", "Cxcl11", "Ccl5", "Cxcl2"],
    "T cell costimulation": ["Cd80", "Cd86", "Cd83", "Icosl", "Cd40"],
    "Immune checkpoint / exhaustion": ["Pdcd1", "Cd274", "Lag3", "Havcr2", "Tigit"],
    "Il4 / Th2 signaling": ["Il4ra", "Stat6", "Arg1", "Mrc1"],
    "Macrophage M1": ["Tnf", "Il1b", "Nos2", "Cxcl9", "Stat1"],
    "Macrophage M2": ["Mrc1", "Cd163", "Il10", "Tgfb1", "Vegfa"],
}

TCELL_STATE = {
    "Effector / cytotoxic": ["Cd8a", "Gzmb", "Prf1", "Ifng", "Nkg7", "Cd3e"],
    "Exhaustion / checkpoint": ["Pdcd1", "Lag3", "Tigit", "Havcr2", "Ctla4"],
    "Treg / suppressive": ["Foxp3", "Il2ra", "Ikzf2", "Ctla4", "Tnfrsf18"],
}

SPP1_AXIS = {
    "Spp1 / osteopontin": ["Spp1", "Cd44", "Itgav", "Itgb1", "Fn1", "Mmp9"],
    "Macrophage recruitment": ["Spp1", "Cd44", "Ccr2", "Ccl2", "Mrc1", "Arg1"],
    "ECM remodeling": ["Spp1", "Mmp2", "Mmp9", "Col1a2", "Postn", "Tnc"],
    "Tumor-immune crosstalk": ["Spp1", "Cd44", "Il6", "Tnf", "Cxcl12", "Vegfa"],
}


def prep_barcode(slice_id: str, pool_barcode: str) -> str:
    """Map slice pool barcode to pooled prep / prediction CellID."""
    return f"{slice_id}__{pool_barcode}@{slice_id}"


def pool_pred_names(pool: ad.AnnData, slice_id: str) -> pd.Index:
    return pd.Index([prep_barcode(slice_id, b) for b in pool.obs_names], name="pred_id")


def load_pred_feather(path: str | Path) -> pd.DataFrame:
    pred = pd.read_feather(path)
    if "CellID" in pred.columns:
        pred = pred.set_index("CellID")
    return pred


def align_pool_pred(
    pool: ad.AnnData,
    pred: pd.DataFrame,
    slice_id: str,
    genes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return pred rows aligned to pool.obs_names via prep barcodes."""
    names = pool_pred_names(pool, slice_id)
    if genes:
        cols = [g for g in genes if g in pred.columns]
        aligned = pred.reindex(names)[cols] if cols else pred.reindex(names).iloc[:, :0]
    else:
        aligned = pred.reindex(names)
    aligned.index = pool.obs_names
    if aligned.shape[1] == 0:
        ok = pd.Series(False, index=pool.obs_names)
        return aligned, ok
    ok = aligned.notna().all(axis=1) if genes else aligned.notna().any(axis=1)
    return aligned, ok


def log1p_pred_minus_expr(pred_df: pd.DataFrame, expr_df: pd.DataFrame) -> pd.DataFrame:
    """Element-wise log1p(pred) − expr on matched log1p expression scale."""
    pr = np.log1p(np.maximum(pred_df.to_numpy(dtype=float), 0.0))
    ex = expr_df.to_numpy(dtype=float)
    return pd.DataFrame(pr - ex, index=pred_df.index, columns=pred_df.columns)


def module_score_delta(
    adata_near: sc.AnnData,
    adata_far: sc.AnnData,
    genes: list[str],
) -> float:
    near = module_score(adata_near, genes, "_")
    far = module_score(adata_far, genes, "_")
    if np.isnan(near).all() or np.isnan(far).all():
        return float("nan")
    return float(np.nanmean(near) - np.nanmean(far))


def predicted_module_score_autonomous(
    pool: sc.AnnData,
    neighbor_ct: str,
    pred: pd.DataFrame,
    slice_id: str,
    genes: list[str],
) -> float:
    """Global autonomous module Δ on NTC neighbor cells (not spatially split)."""
    ntc = pool[(pool.obs["target_gene"].astype(str) == "non-targeting") &
               (pool.obs["cell_type"].astype(str) == neighbor_ct)]
    common = _unique_genes(genes, ntc.var_names)
    common = [g for g in common if g in pred.columns]
    if len(common) < 2 or ntc.n_obs < 10:
        return float("nan")
    pred_aligned, ok = align_pool_pred(ntc, pred, slice_id, genes=common)
    if ok.sum() < 10:
        return float("nan")
    sub = ntc[ok.values]
    use = _unique_genes(common, sub.var_names)
    if len(use) < 2:
        return float("nan")
    expr = sub[:, use].X.toarray() if sparse.issparse(sub.X) else np.asarray(sub[:, use].X)
    pr = pred_aligned.loc[sub.obs_names, use].to_numpy(dtype=float)
    delta = np.log1p(np.maximum(pr, 0.0)) - expr
    return float(np.nanmean(delta))


def predicted_module_score_delta(
    pool_near: sc.AnnData,
    pool_far: sc.AnnData,
    pred: pd.DataFrame,
    slice_id: str,
    genes: list[str],
) -> float:
    common = _unique_genes(genes, pool_near.var_names)
    common = [g for g in common if g in pred.columns]
    if len(common) < 2:
        return float("nan")

    def arm_delta(sub: sc.AnnData) -> float:
        use = _unique_genes(common, sub.var_names)
        if len(use) < 2:
            return float("nan")
        pred_aligned, ok = align_pool_pred(sub, pred, slice_id, genes=use)
        if ok.sum() < 3:
            return float("nan")
        sub = sub[ok.values]
        expr = sub[:, use].X.toarray() if sparse.issparse(sub.X) else np.asarray(sub[:, use].X)
        pr = pred_aligned.loc[sub.obs_names, use].to_numpy(dtype=float)
        delta = np.log1p(np.maximum(pr, 0.0)) - expr
        return float(np.nanmean(delta))

    d_near, d_far = arm_delta(pool_near), arm_delta(pool_far)
    if not np.isfinite(d_near) or not np.isfinite(d_far):
        return float("nan")
    return d_near - d_far


def _dense_mean(adata: sc.AnnData, genes: list[str]) -> pd.Series:
    keep = [g for g in genes if g in adata.var_names]
    if not keep:
        return pd.Series(dtype=float)
    sub = adata[:, keep]
    arr = sub.X.toarray() if sparse.issparse(sub.X) else np.asarray(sub.X)
    return pd.Series(arr.mean(axis=0), index=keep)


def _unique_genes(genes: list[str], var_names) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for g in genes:
        if g in seen:
            continue
        if g in var_names:
            seen.add(g)
            out.append(g)
    return out


def module_score(adata: sc.AnnData, genes: list[str], name: str) -> np.ndarray:
    keep = _unique_genes(genes, adata.var_names)
    if len(keep) < 2:
        return np.full(adata.n_obs, np.nan)
    sub = adata[:, keep]
    arr = sub.X.toarray() if sparse.issparse(sub.X) else np.asarray(sub.X)
    return np.asarray(arr.mean(axis=1))


def get_spatial_perturbation_degs(
    adata: sc.AnnData,
    gene: str,
    k_neighbors: int = 25,
    exclude_perturbed: bool = True,
    cell_type: str | None = "immune",
    cell_type_col: str = "cell_type",
    target_col: str = "target_gene",
    ntc_label: str = "non-targeting",
) -> pd.DataFrame:
    """Wilcoxon DE in spatial neighbors of sgP vs NTC source cells."""
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found.")
    if target_col not in adata.obs.columns:
        raise ValueError(f"Column '{target_col}' not found.")

    perturbed_source_mask = adata.obs[target_col].astype(str) == gene
    perturbed_source_indices = np.where(perturbed_source_mask)[0]
    if len(perturbed_source_indices) == 0:
        raise ValueError(f"No cells with target_gene == '{gene}'.")

    unperturbed_mask = adata.obs[target_col].astype(str) == ntc_label
    control_source_indices = np.where(unperturbed_mask)[0]
    if len(control_source_indices) == 0:
        raise ValueError("No NTC cells found.")

    spatial_coords = adata.obsm["spatial"]
    tree = KDTree(spatial_coords)
    actual_k = min(k_neighbors, adata.n_obs - 1)

    _, neighbor_idx_p = tree.query(spatial_coords[perturbed_source_indices], k=actual_k + 1)
    _, neighbor_idx_c = tree.query(spatial_coords[control_source_indices], k=actual_k + 1)

    niche_perturbed_idx = np.unique(neighbor_idx_p.flatten())
    niche_control_idx = np.unique(neighbor_idx_c.flatten())

    niche_perturbed_idx = np.setdiff1d(niche_perturbed_idx, perturbed_source_indices)
    niche_control_idx = np.setdiff1d(niche_control_idx, control_source_indices)

    if exclude_perturbed:
        all_perturbed_indices = np.where(~unperturbed_mask)[0]
        niche_perturbed_idx = np.setdiff1d(niche_perturbed_idx, all_perturbed_indices)
        niche_control_idx = np.setdiff1d(niche_control_idx, all_perturbed_indices)

    if cell_type:
        type_mask = adata.obs[cell_type_col].astype(str) == cell_type
        type_indices = np.where(type_mask)[0]
        niche_perturbed_idx = np.intersect1d(niche_perturbed_idx, type_indices)
        niche_control_idx = np.intersect1d(niche_control_idx, type_indices)

    label_col = f"_spatial_cond_{gene}"
    adata.obs[label_col] = "none"
    adata.obs.iloc[niche_control_idx, adata.obs.columns.get_loc(label_col)] = "control_neighbor"
    adata.obs.iloc[niche_perturbed_idx, adata.obs.columns.get_loc(label_col)] = "perturbed_neighbor"

    n_p, n_c = len(niche_perturbed_idx), len(niche_control_idx)
    if n_p < 3 or n_c < 3:
        adata.obs.drop(columns=[label_col], inplace=True)
        raise ValueError(f"Insufficient niche cells (P: {n_p}, C: {n_c}).")

    key = f"rank_genes_{gene}_{cell_type or 'all'}"
    sc.tl.rank_genes_groups(
        adata,
        groupby=label_col,
        groups=["perturbed_neighbor"],
        reference="control_neighbor",
        method="wilcoxon",
        key_added=key,
    )
    deg_df = sc.get.rank_genes_groups_df(adata, group="perturbed_neighbor", key=key)
    deg_df = deg_df.rename(columns={"names": "gene", "logfoldchanges": "log2fc", "pvals": "pval", "pvals_adj": "pval_adj"})
    deg_df["abs_log2fc"] = deg_df["log2fc"].abs()
    adata.obs.drop(columns=[label_col], inplace=True)
    deg_df.attrs["n_perturbed_neighbors"] = n_p
    deg_df.attrs["n_control_neighbors"] = n_c
    return deg_df


def spatial_neighbor_indices(
    adata: sc.AnnData,
    gene: str,
    k_neighbors: int = 25,
    exclude_perturbed: bool = True,
    cell_type: str | None = "immune",
    source_cell_type: str | None = None,
    restrict_to_ntc: bool = False,
    cell_type_col: str = "cell_type",
    target_col: str = "target_gene",
    ntc_label: str = "non-targeting",
) -> tuple[np.ndarray, np.ndarray]:
    """Return bystander neighbor indices near sgP vs NTC sources."""
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found.")
    src_ct = source_cell_type or cell_type
    perturbed_source_mask = adata.obs[target_col].astype(str) == gene
    if src_ct:
        perturbed_source_mask &= adata.obs[cell_type_col].astype(str) == src_ct
    perturbed_source_indices = np.where(perturbed_source_mask)[0]
    if len(perturbed_source_indices) == 0:
        raise ValueError(f"No source cells with target_gene == '{gene}'.")

    unperturbed_mask = adata.obs[target_col].astype(str) == ntc_label
    if src_ct:
        unperturbed_mask &= adata.obs[cell_type_col].astype(str) == src_ct
    control_source_indices = np.where(unperturbed_mask)[0]
    if len(control_source_indices) == 0:
        raise ValueError("No matched NTC source cells found.")

    spatial_coords = adata.obsm["spatial"]
    tree = KDTree(spatial_coords)
    actual_k = min(k_neighbors, adata.n_obs - 1)
    _, neighbor_idx_p = tree.query(spatial_coords[perturbed_source_indices], k=actual_k + 1)
    _, neighbor_idx_c = tree.query(spatial_coords[control_source_indices], k=actual_k + 1)
    niche_perturbed_idx = np.unique(neighbor_idx_p.flatten())
    niche_control_idx = np.unique(neighbor_idx_c.flatten())
    niche_perturbed_idx = np.setdiff1d(niche_perturbed_idx, perturbed_source_indices)
    niche_control_idx = np.setdiff1d(niche_control_idx, control_source_indices)
    if exclude_perturbed:
        global_ntc = adata.obs[target_col].astype(str) == ntc_label
        all_perturbed_indices = np.where(~global_ntc)[0]
        niche_perturbed_idx = np.setdiff1d(niche_perturbed_idx, all_perturbed_indices)
        niche_control_idx = np.setdiff1d(niche_control_idx, perturbed_source_indices)
    if cell_type:
        type_indices = np.where(adata.obs[cell_type_col].astype(str) == cell_type)[0]
        niche_perturbed_idx = np.intersect1d(niche_perturbed_idx, type_indices)
        niche_control_idx = np.intersect1d(niche_control_idx, type_indices)
    if restrict_to_ntc:
        ntc_indices = np.where(adata.obs[target_col].astype(str) == ntc_label)[0]
        niche_perturbed_idx = np.intersect1d(niche_perturbed_idx, ntc_indices)
        niche_control_idx = np.intersect1d(niche_control_idx, ntc_indices)
    return niche_perturbed_idx, niche_control_idx


def spatial_ntc_niche_indices(
    adata: sc.AnnData,
    gene: str,
    k_neighbors: int = 25,
    cell_type: str | None = "immune",
    source_cell_type: str | None = None,
    cell_type_col: str = "cell_type",
    target_col: str = "target_gene",
    ntc_label: str = "non-targeting",
) -> tuple[np.ndarray, np.ndarray]:
    """NTC bystanders near sgP sources vs matched NTC cells outside that niche.

    SpaceTravLR predictions exist only on NTC substrate cells. In pooled CRISPR
    sections, NTC source neighborhoods rarely contain other NTC bystanders, so
    the sgP-vs-NTC-source kNN contrast has no predicted control arm.
    """
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found.")
    src_ct = source_cell_type or cell_type
    perturbed_source_mask = adata.obs[target_col].astype(str) == gene
    if src_ct:
        perturbed_source_mask &= adata.obs[cell_type_col].astype(str) == src_ct
    perturbed_source_indices = np.where(perturbed_source_mask)[0]
    if len(perturbed_source_indices) == 0:
        raise ValueError(f"No source cells with target_gene == '{gene}'.")

    global_ntc = adata.obs[target_col].astype(str) == ntc_label
    spatial_coords = adata.obsm["spatial"]
    tree = KDTree(spatial_coords)
    actual_k = min(k_neighbors, adata.n_obs - 1)
    _, neighbor_idx_p = tree.query(spatial_coords[perturbed_source_indices], k=actual_k + 1)
    near_idx = np.unique(neighbor_idx_p.flatten())
    near_idx = np.setdiff1d(near_idx, perturbed_source_indices)
    all_sgP = np.where(~global_ntc)[0]
    near_idx = np.setdiff1d(near_idx, all_sgP)
    near_idx = np.intersect1d(near_idx, np.where(global_ntc)[0])
    if cell_type:
        type_indices = np.where(adata.obs[cell_type_col].astype(str) == cell_type)[0]
        near_idx = np.intersect1d(near_idx, type_indices)

    ntc_type = np.where(global_ntc)[0]
    if cell_type:
        ntc_type = np.intersect1d(ntc_type, np.where(adata.obs[cell_type_col].astype(str) == cell_type)[0])
    far_idx = np.setdiff1d(ntc_type, near_idx)
    return near_idx, far_idx


def spatial_ntc_niche_pseudobulk(
    adata: sc.AnnData,
    gene: str,
    k_neighbors: int = 25,
    cell_type: str | None = "immune",
    source_cell_type: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    near_idx, far_idx = spatial_ntc_niche_indices(
        adata, gene, k_neighbors=k_neighbors, cell_type=cell_type,
        source_cell_type=source_cell_type, **kwargs,
    )
    if len(near_idx) < 3 or len(far_idx) < 3:
        raise ValueError(f"Insufficient NTC niche cells (near: {len(near_idx)}, far: {len(far_idx)}).")
    df = pseudobulk_delta_df(adata[near_idx], adata[far_idx])
    df.attrs["n_perturbed_neighbors"] = len(near_idx)
    df.attrs["n_control_neighbors"] = len(far_idx)
    return df


def spatial_neighbor_pseudobulk(
    adata: sc.AnnData,
    gene: str,
    k_neighbors: int = 25,
    exclude_perturbed: bool = True,
    cell_type: str | None = "immune",
    source_cell_type: str | None = None,
    restrict_to_ntc: bool = False,
    cell_type_col: str = "cell_type",
    target_col: str = "target_gene",
    ntc_label: str = "non-targeting",
) -> pd.DataFrame:
    """Mean expression difference in spatial kNN neighbor niches (fast pseudobulk)."""
    niche_perturbed_idx, niche_control_idx = spatial_neighbor_indices(
        adata, gene, k_neighbors=k_neighbors, exclude_perturbed=exclude_perturbed,
        cell_type=cell_type, source_cell_type=source_cell_type,
        restrict_to_ntc=restrict_to_ntc,
        cell_type_col=cell_type_col, target_col=target_col, ntc_label=ntc_label,
    )
    if len(niche_perturbed_idx) < 3 or len(niche_control_idx) < 3:
        raise ValueError(
            f"Insufficient niche cells (P: {len(niche_perturbed_idx)}, C: {len(niche_control_idx)})."
        )
    df = pseudobulk_delta_df(adata[niche_perturbed_idx], adata[niche_control_idx])
    df.attrs["n_perturbed_neighbors"] = len(niche_perturbed_idx)
    df.attrs["n_control_neighbors"] = len(niche_control_idx)
    return df


def direct_cell_pseudobulk(
    adata: sc.AnnData,
    gene: str,
    cell_type: str | None = None,
    cell_type_col: str = "cell_type",
    target_col: str = "target_gene",
    ntc_label: str = "non-targeting",
) -> pd.DataFrame:
    """Mean expression sgP vs NTC within cell type (direct perturbed cells)."""
    pert_mask = adata.obs[target_col].astype(str) == gene
    ntc_mask = adata.obs[target_col].astype(str) == ntc_label
    if cell_type:
        ct_mask = adata.obs[cell_type_col].astype(str) == cell_type
        pert_mask &= ct_mask
        ntc_mask &= ct_mask
    n_p, n_c = int(pert_mask.sum()), int(ntc_mask.sum())
    if n_p < 10 or n_c < 10:
        raise ValueError(f"Insufficient direct cells (P: {n_p}, C: {n_c}).")
    df = pseudobulk_delta_df(adata[pert_mask], adata[ntc_mask])
    df.attrs["n_perturbed_cells"] = n_p
    df.attrs["n_control_cells"] = n_c
    return df


def find_neighbors(
    adata: sc.AnnData,
    cell_target: str,
    cell_indices: Iterable,
    n_neighbors: int,
    max_distance: float,
    cell_type_col: str = "cell_type",
) -> pd.DataFrame:
    coords = adata.obsm["spatial"]
    tree = cKDTree(coords)
    neighbors_df = []
    for cell_id in cell_indices:
        query_idx = adata.obs_names.get_loc(cell_id) if isinstance(cell_id, str) else int(cell_id)
        query_coord = coords[query_idx]
        distances, indices = tree.query(query_coord, k=min(n_neighbors + 1, adata.n_obs))
        if np.isscalar(distances):
            distances, indices = np.array([distances]), np.array([indices])
        neighbor_indices = indices[1:]
        neighbor_distances = distances[1:]
        df = pd.DataFrame({
            "cell_barcode": adata.obs_names[neighbor_indices],
            "cell_index": neighbor_indices,
            "distance": neighbor_distances,
            "cell_type": adata.obs[cell_type_col].iloc[neighbor_indices].values,
        })
        if "slice_id" in adata.obs.columns:
            df["slice_id"] = adata.obs["slice_id"].iloc[neighbor_indices].values
        df = df.query(f'cell_type == "{cell_target}" and distance < {max_distance}')
        neighbors_df.append(df)
    if not neighbors_df:
        return pd.DataFrame(columns=["cell_barcode", "cell_index", "distance", "cell_type"])
    return pd.concat(neighbors_df, ignore_index=True)


def find_de_genes(
    adata1: sc.AnnData,
    adata2: sc.AnnData,
    label1: str = "Experimental",
    label2: str = "SpaceTravLR",
    method: str = "t-test_overestim_var",
) -> pd.DataFrame:
    if adata1.n_obs == 0 or adata2.n_obs == 0:
        return pd.DataFrame(columns=["gene", "log2fc", "pval", "pval_adj", "score", "abs_log2fc"])

    condition_key = "_de_comparison_group"
    a1 = adata1.copy()
    a2 = adata2.copy()
    a1.obs[condition_key] = label1
    a2.obs[condition_key] = label2
    combined = ad.concat([a1, a2], join="outer", merge="same")
    sc.tl.rank_genes_groups(combined, groupby=condition_key, groups=[label1], reference=label2, method=method)
    de_df = sc.get.rank_genes_groups_df(combined, group=label1)
    de_df = de_df[["names", "logfoldchanges", "pvals", "pvals_adj", "scores"]]
    de_df.columns = ["gene", "log2fc", "pval", "pval_adj", "score"]
    de_df["abs_log2fc"] = de_df["log2fc"].abs()
    return de_df


def pseudobulk_delta_df(
    adata_a: sc.AnnData,
    adata_b: sc.AnnData,
    genes: list[str] | None = None,
) -> pd.DataFrame:
    """Mean expression difference (log1p scale) as log2fc proxy."""
    if genes is None:
        genes = sorted(set(adata_a.var_names) & set(adata_b.var_names))
    ma = _dense_mean(adata_a, genes)
    mb = _dense_mean(adata_b, genes)
    common = ma.index.intersection(mb.index)
    delta = ma.loc[common] - mb.loc[common]
    return pd.DataFrame({"gene": delta.index, "log2fc": delta.values, "abs_log2fc": delta.abs().values})


def predicted_niche_delta_df(
    baseline: sc.AnnData,
    pred: pd.DataFrame,
    niche_mask: pd.Series,
    genes: list[str],
) -> pd.DataFrame:
    sub = baseline[niche_mask.values].copy()
    sub = sub[sub.obs_names.isin(pred.index)]
    if sub.n_obs < 3:
        return pd.DataFrame(columns=["gene", "log2fc", "abs_log2fc"])
    common = [g for g in genes if g in sub.var_names and g in pred.columns]
    base_expr = pred.loc[sub.obs_names, common] - (
        sub[:, common].X.toarray() if sparse.issparse(sub.X) else np.asarray(sub[:, common].X)
    )
    delta = base_expr.mean(0)
    return pd.DataFrame({"gene": common, "log2fc": delta.values, "abs_log2fc": np.abs(delta.values)})


def _annotate_texts(ax, texts, merged, x_col, y_col):
    if not texts:
        return
    if adjust_text is not None:
        xp = merged[x_col].to_numpy()
        yp = merged[y_col].to_numpy()
        scatter_pc = ax.collections[0] if ax.collections else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adjust_text(
                texts, x=xp, y=yp, objects=scatter_pc, ax=ax,
                force_text=(0.35, 0.55), force_static=(0.55, 0.9),
                max_move=(22, 22), iter_lim=400,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            )
    else:
        for t in texts:
            t.set_fontsize(8)


def plot_gene_comparison_advanced(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    label1: str = "SPAC-seq",
    label2: str = "SpaceTravLR",
    highlight_genes: list[str] | None = None,
    top_n_labels: int = 15,
    figsize=(8, 8),
    target_ko: str = "",
    alpha: float = 0.85,
    dpi: int = 200,
    palette: dict | None = None,
    size: int = 55,
    save_path: str | None = None,
    show: bool = False,
    neighbor_ct: str = "",
    source_ct: str = "",
    include_zero_log2fc: bool = True,
    zero_atol: float = 1e-12,
    axis_lim: float | None = None,
    ax=None,
    use_size: bool = True,
    title_suffix: str = "",
):
    merged = pd.merge(
        df1[["gene", "log2fc"] + (["pval_adj"] if "pval_adj" in df1.columns else [])],
        df2[["gene", "log2fc"]],
        on="gene",
        suffixes=(f"_{label1}", f"_{label2}"),
    )
    x_col = f"log2fc_{label1}"
    y_col = f"log2fc_{label2}"
    merged = merged.dropna(subset=[x_col, y_col])
    if target_ko:
        merged = merged[merged.gene.astype(str).str.casefold() != str(target_ko).casefold()]

    conditions = [
        (merged[x_col] > 0) & (merged[y_col] > 0),
        (merged[x_col] < 0) & (merged[y_col] < 0),
        (merged[x_col] > 0) & (merged[y_col] < 0),
        (merged[x_col] < 0) & (merged[y_col] > 0),
    ]
    choices = ["Concordant Up", "Concordant Down", "Discordant", "Discordant"]
    merged["Category"] = np.select(conditions, choices, default="Neutral")
    merged["magnitude"] = np.sqrt(merged[x_col] ** 2 + merged[y_col] ** 2)

    if use_size and f"pval_adj_{label1}" in merged.columns:
        p = merged[f"pval_adj_{label1}"].fillna(1.0).clip(lower=1e-300)
        merged["_neg_log10p"] = -np.log10(p)
        pmax = merged["_neg_log10p"].max()
        merged["_size"] = 20 + (size - 20) * (merged["_neg_log10p"] / pmax if pmax > 0 else 0)
    else:
        merged["_size"] = size

    pearson_r = spearman_r = float("nan")
    if len(merged) >= 2:
        pearson_r, _ = stats.pearsonr(merged[x_col], merged[y_col])
        spearman_r, _ = stats.spearmanr(merged[x_col], merged[y_col])

    ax_provided = ax is not None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    if palette is None:
        palette = {
            "Concordant Up": "#16E0BD",
            "Concordant Down": "#FC6471",
            "Discordant": "#8d8dec",
            "Neutral": "#DDDDDD",
        }

    sns.scatterplot(
        data=merged, x=x_col, y=y_col, hue="Category", palette=palette,
        alpha=alpha, size="_size", sizes=(20, size), edgecolor="black",
        linewidth=0.4, ax=ax, legend=False,
    )

    if axis_lim is not None:
        limit = float(axis_lim)
    elif len(merged) == 0:
        limit = 1.0
    else:
        mx = max(merged[x_col].abs().max(), merged[y_col].abs().max())
        limit = float(mx * 1.12) if mx > 0 else 1.0
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0

    ax.plot([-limit, limit], [-limit, limit], ls="--", color="gray", alpha=0.35, lw=1)
    ax.axhline(0, color="black", lw=0.8, alpha=0.15)
    ax.axvline(0, color="black", lw=0.8, alpha=0.15)

    texts, labeled = [], set()
    if highlight_genes:
        for _, row in merged[merged.gene.isin(highlight_genes)].iterrows():
            texts.append(ax.text(row[x_col], row[y_col], row["gene"], fontsize=10, fontweight="bold"))
            labeled.add(row["gene"])

    if top_n_labels > 0 and len(merged) > 0:
        for _, row in merged.nlargest(top_n_labels + len(labeled), "magnitude").iterrows():
            if row.gene not in labeled:
                texts.append(ax.text(row[x_col], row[y_col], row["gene"], fontsize=8, color="#333"))
                labeled.add(row.gene)

    _annotate_texts(ax, texts, merged, x_col, y_col)

    title = f"DEGs in {neighbor_ct} around sg{target_ko} ({source_ct})"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.text(
        0.5, 1.01, f"Spearman r = {spearman_r:.2f}  |  Pearson r = {pearson_r:.2f}  |  n = {len(merged)}",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#444",
    )
    ax.set_xlabel(f"Log2FC ({label1})", fontsize=11)
    ax.set_ylabel(f"Log2FC ({label2})", fontsize=11)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.grid(True, alpha=0.12)

    stats_out = {"pearson_r": pearson_r, "spearman_r": spearman_r, "n_genes": len(merged),
                 "concordant_up": int((merged.Category == "Concordant Up").sum()),
                 "concordant_down": int((merged.Category == "Concordant Down").sum())}

    if not ax_provided:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    elif not ax_provided:
        plt.close(fig)

    return stats_out, merged
