#!/usr/bin/env python3
"""Validate tissue-structure received-ligand inference against spatial ground truth.

Mirrors SpaceTravLR's Gaussian received-ligand formula:

    received[i,l] = (1/N) Σ_j scale · exp(-d(i,j)² / 2r²) · expr[j,l]

and the structure approximation:

    received̂[i,l] = Σ_t Ŝ[type(i), t] · μ[t, l]

where Ŝ is the type-conditional mean Gaussian weight mass learned from a
spatial reference and μ are query type-mean ligand expression profiles.

Error decomposition
-------------------
1. type_mean_oracle  — true per-cell S[i,t] × μ  (expression heterogeneity only)
2. structure_pooled  — Ŝ[type(i),t] × μ from the same sample
3. abundance_baseline — ignore spatial structure; use type frequencies × total mass
4. cross_sample      — Ŝ from a matched tissue / replicate

Outputs metrics CSV + JSON summary under --outdir.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr


def _as_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.toarray(), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def load_xy(adata: ad.AnnData) -> np.ndarray:
    for key in ("spatial", "X_spatial", "spatial_loc"):
        if key in adata.obsm:
            xy = np.asarray(adata.obsm[key], dtype=np.float64)
            if xy.ndim == 2 and xy.shape[1] >= 2:
                return xy[:, :2]
    raise KeyError("No spatial coordinates in obsm")


def gaussian_received(
    xy: np.ndarray,
    lig: np.ndarray,
    radius: float,
    scale: float = 1.0,
    chunk: int = 512,
) -> np.ndarray:
    """Exact SpaceTravLR received ligands with O(N·L) memory."""
    n = xy.shape[0]
    n_lig = lig.shape[1]
    inv_2r2 = -1.0 / (2.0 * radius * radius)
    n_inv = 1.0 / n
    out = np.zeros((n, n_lig), dtype=np.float64)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        d2 = (
            (xy[i0:i1, 0:1] - xy[:, 0]) ** 2
            + (xy[i0:i1, 1:2] - xy[:, 1]) ** 2
        )
        w = scale * np.exp(d2 * inv_2r2)  # (chunk, n)
        out[i0:i1] = (w @ lig) * n_inv
    return out


def type_maps(labels: Sequence[str]) -> Tuple[List[str], np.ndarray]:
    names = sorted(set(labels))
    idx = {t: i for i, t in enumerate(names)}
    codes = np.array([idx[t] for t in labels], dtype=np.int64)
    return names, codes


def cell_structure_weights(
    xy: np.ndarray,
    type_codes: np.ndarray,
    n_types: int,
    radius: float,
    scale: float = 1.0,
    hard_radius: Optional[float] = None,
    chunk: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = xy.shape[0]
    hard_r = radius if hard_radius is None else hard_radius
    hard_r2 = hard_r * hard_r
    inv_2r2 = -1.0 / (2.0 * radius * radius)
    n_inv = 1.0 / n
    weight = np.zeros((n, n_types), dtype=np.float64)
    soft = np.zeros((n, n_types), dtype=np.float64)
    hard = np.zeros((n, n_types), dtype=np.float64)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        d2 = (
            (xy[i0:i1, 0:1] - xy[:, 0]) ** 2
            + (xy[i0:i1, 1:2] - xy[:, 1]) ** 2
        )
        w = scale * np.exp(d2 * inv_2r2)
        for local, i in enumerate(range(i0, i1)):
            for t in range(n_types):
                mask = type_codes == t
                wt = w[local, mask].sum()
                soft[i, t] = wt
                weight[i, t] = wt * n_inv
                if hard_r2 > 0:
                    hard[i, t] = np.sum((d2[local, mask] <= hard_r2) & (np.arange(n)[mask] != i))
    return weight, soft, hard


def build_structure_ref(
    xy: np.ndarray,
    labels: Sequence[str],
    radius: float,
    scale: float = 1.0,
    hard_radius: Optional[float] = None,
) -> Dict:
    names, codes = type_maps(labels)
    n_types = len(names)
    weight, soft, hard = cell_structure_weights(
        xy, codes, n_types, radius, scale, hard_radius
    )
    mean_w = np.zeros((n_types, n_types))
    mean_s = np.zeros((n_types, n_types))
    mean_h = np.zeros((n_types, n_types))
    counts = np.bincount(codes, minlength=n_types)
    for t in range(n_types):
        sel = codes == t
        if not np.any(sel):
            continue
        mean_w[t] = weight[sel].mean(axis=0)
        mean_s[t] = soft[sel].mean(axis=0)
        mean_h[t] = hard[sel].mean(axis=0)
    return {
        "cell_types": names,
        "mean_weight_mass": mean_w,
        "mean_soft_counts": mean_s,
        "mean_hard_counts": mean_h,
        "radius": radius,
        "scale_factor": scale,
        "hard_radius": radius if hard_radius is None else hard_radius,
        "n_ref_cells": xy.shape[0],
        "ref_type_counts": counts.tolist(),
        "_cell_weight": weight,
        "_cell_soft": soft,
        "_cell_hard": hard,
        "_codes": codes,
    }


def type_mean_expr(expr: np.ndarray, codes: np.ndarray, n_types: int) -> np.ndarray:
    means = np.zeros((n_types, expr.shape[1]), dtype=np.float64)
    for t in range(n_types):
        sel = codes == t
        if np.any(sel):
            means[t] = expr[sel].mean(axis=0)
    return means


def infer_from_structure(
    mean_weight: np.ndarray, recv_codes: np.ndarray, type_means: np.ndarray
) -> np.ndarray:
    n = recv_codes.shape[0]
    out = np.zeros((n, type_means.shape[1]), dtype=np.float64)
    for i, r in enumerate(recv_codes):
        out[i] = mean_weight[r] @ type_means
    return out


def abundance_baseline(mean_weight: np.ndarray, type_counts: Sequence[int]) -> np.ndarray:
    freqs = np.asarray(type_counts, dtype=np.float64)
    freqs = freqs / max(freqs.sum(), 1.0)
    out = np.zeros_like(mean_weight)
    for r in range(mean_weight.shape[0]):
        out[r] = mean_weight[r].sum() * freqs
    return out


def restrict_types(ref: Dict, keep: Sequence[str]) -> Dict:
    keep = [t for t in sorted(set(keep)) if t in ref["cell_types"]]
    old = {t: i for i, t in enumerate(ref["cell_types"])}
    idx = [old[t] for t in keep]
    return {
        **ref,
        "cell_types": keep,
        "mean_weight_mass": ref["mean_weight_mass"][np.ix_(idx, idx)],
        "mean_soft_counts": ref["mean_soft_counts"][np.ix_(idx, idx)],
        "mean_hard_counts": ref["mean_hard_counts"][np.ix_(idx, idx)],
        "ref_type_counts": [ref["ref_type_counts"][i] for i in idx],
    }


def expression_matched_weights(
    ref_weight: np.ndarray,
    ref_codes: np.ndarray,
    ref_fp: np.ndarray,
    query_fp: np.ndarray,
    query_codes: np.ndarray,
    k: int = 15,
    exclude_self: bool = False,
) -> np.ndarray:
    """Average reference S rows of kNN cells in cosine fingerprint space (same-type preferred)."""
    def _norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, 1e-12)

    rn = _norm(ref_fp)
    qn = _norm(query_fp)
    sims = qn @ rn.T  # (nq, nref)
    same = (query_codes[:, None] == ref_codes[None, :]).astype(np.float64)
    sims = sims + same
    if exclude_self and ref_fp.shape[0] == query_fp.shape[0]:
        np.fill_diagonal(sims, -np.inf)
    k = max(1, min(k, ref_weight.shape[0] - (1 if exclude_self else 0)))
    nn = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    out = np.zeros((query_fp.shape[0], ref_weight.shape[1]), dtype=np.float64)
    for i in range(query_fp.shape[0]):
        out[i] = ref_weight[nn[i]].mean(axis=0)
    return out


def type_level_metrics(pred: np.ndarray, truth: np.ndarray, codes: np.ndarray) -> Dict:
    """Aggregate to receiver-type means before scoring (natural non-spatial target)."""
    n_types = int(codes.max()) + 1 if len(codes) else 0
    rows_p, rows_t = [], []
    for t in range(n_types):
        sel = codes == t
        if not np.any(sel):
            continue
        rows_p.append(pred[sel].mean(axis=0))
        rows_t.append(truth[sel].mean(axis=0))
    if not rows_p:
        return {"pearson_mean": float("nan"), "mae": float("nan"), "n_types": 0}
    p = np.vstack(rows_p)
    t = np.vstack(rows_t)
    pears, _ = col_corrs(p, t)
    mae, rmse, rel = matrix_metrics(p, t)
    return {
        "pearson_mean": float(np.nanmean(pears)),
        "mae": mae,
        "rmse": rmse,
        "rel_mae": rel,
        "n_types": p.shape[0],
    }


def structure_matrix_cosine(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.ravel(), b.ravel()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den == 0:
        return float("nan")
    return float(x.dot(y) / den)


@dataclass
class Metrics:
    pearson_mean: float
    pearson_median: float
    spearman_mean: float
    mae: float
    rmse: float
    rel_mae: float
    soft_cosine: float
    hard_cosine: float
    slope: float
    n_ligands: int
    n_cells: int


def matrix_metrics(pred: np.ndarray, truth: np.ndarray) -> Tuple[float, float, float]:
    err = np.abs(pred - truth)
    mae = float(err.mean())
    rmse = float(np.sqrt(((pred - truth) ** 2).mean()))
    rel = float((err / np.maximum(np.abs(truth), 1e-8)).mean())
    return mae, rmse, rel


def col_corrs(pred: np.ndarray, truth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pears, spears = [], []
    for k in range(pred.shape[1]):
        a, b = pred[:, k], truth[:, k]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            pears.append(np.nan)
            spears.append(np.nan)
            continue
        pears.append(pearsonr(a, b)[0])
        spears.append(spearmanr(a, b).correlation)
    return np.asarray(pears, float), np.asarray(spears, float)


def composition_cosine(pred: np.ndarray, truth: np.ndarray) -> float:
    num = (pred * truth).sum(axis=1)
    den = np.linalg.norm(pred, axis=1) * np.linalg.norm(truth, axis=1)
    ok = den > 0
    if not np.any(ok):
        return float("nan")
    return float((num[ok] / den[ok]).mean())


def calib_slope(pred: np.ndarray, truth: np.ndarray) -> float:
    x = pred.ravel()
    y = truth.ravel()
    if np.std(x) < 1e-12:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def score(
    pred: np.ndarray,
    truth: np.ndarray,
    soft_pred: Optional[np.ndarray] = None,
    soft_truth: Optional[np.ndarray] = None,
    hard_pred: Optional[np.ndarray] = None,
    hard_truth: Optional[np.ndarray] = None,
) -> Metrics:
    pears, spears = col_corrs(pred, truth)
    mae, rmse, rel = matrix_metrics(pred, truth)
    return Metrics(
        pearson_mean=float(np.nanmean(pears)),
        pearson_median=float(np.nanmedian(pears)),
        spearman_mean=float(np.nanmean(spears)),
        mae=mae,
        rmse=rmse,
        rel_mae=rel,
        soft_cosine=composition_cosine(soft_pred, soft_truth)
        if soft_pred is not None and soft_truth is not None
        else float("nan"),
        hard_cosine=composition_cosine(hard_pred, hard_truth)
        if hard_pred is not None and hard_truth is not None
        else float("nan"),
        slope=calib_slope(pred, truth),
        n_ligands=pred.shape[1],
        n_cells=pred.shape[0],
    )


def pick_ligands(adata: ad.AnnData, n: int, seed: int = 0) -> List[str]:
    rng = np.random.default_rng(seed)
    x = adata.X
    if sparse.issparse(x):
        means = np.asarray(x.mean(axis=0)).ravel()
    else:
        means = np.asarray(x, dtype=np.float64).mean(axis=0)
    order = np.argsort(-means)
    genes = np.asarray(adata.var_names)
    top = genes[order[: max(n * 5, n)]].tolist()
    if len(top) <= n:
        return top
    # Prefer variable ligands among highly expressed genes.
    idx = order[: max(n * 5, n)]
    if sparse.issparse(x):
        sub = _as_dense(x[:, idx])
    else:
        sub = np.asarray(x[:, idx], dtype=np.float64)
    var = sub.var(axis=0)
    pick = np.argsort(-var)[:n]
    return [top[i] for i in pick]


def load_expr(adata: ad.AnnData, genes: Sequence[str]) -> np.ndarray:
    idx = [adata.var_names.get_loc(g) for g in genes]
    return _as_dense(adata[:, idx].X)


def metrics_to_row(method: str, experiment: str, dataset: str, m: Metrics, extra: Optional[Dict] = None) -> Dict:
    row = {
        "experiment": experiment,
        "dataset": dataset,
        "method": method,
        "pearson_mean": m.pearson_mean,
        "pearson_median": m.pearson_median,
        "spearman_mean": m.spearman_mean,
        "mae": m.mae,
        "rmse": m.rmse,
        "rel_mae": m.rel_mae,
        "soft_cosine": m.soft_cosine,
        "hard_cosine": m.hard_cosine,
        "slope": m.slope,
        "n_ligands": m.n_ligands,
        "n_cells": m.n_cells,
        "type_pearson_mean": np.nan,
        "type_mae": np.nan,
        "structure_matrix_cosine": np.nan,
    }
    if extra:
        row.update(extra)
    return row


def evaluate_spatial_holdout(
    path: Path,
    radius: float,
    n_ligands: int,
    max_cells: Optional[int],
    seed: int,
) -> List[Dict]:
    """Split one tissue by x-coordinate: learn structure on left, test on right (coords withheld)."""
    a = ad.read_h5ad(path)
    if max_cells and a.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        a = a[np.sort(rng.choice(a.n_obs, max_cells, replace=False))].copy()
    xy = load_xy(a)
    mid = np.median(xy[:, 0])
    ref_mask = xy[:, 0] <= mid
    qry_mask = ~ref_mask
    if ref_mask.sum() < 50 or qry_mask.sum() < 50:
        return []
    labels = a.obs["cell_type"].astype(str).to_numpy()
    shared = sorted(set(labels[ref_mask]) & set(labels[qry_mask]))
    if len(shared) < 2:
        return []
    keep_ref = ref_mask & np.isin(labels, shared)
    keep_qry = qry_mask & np.isin(labels, shared)
    ref_a = a[keep_ref].copy()
    qry_a = a[keep_qry].copy()
    genes = pick_ligands(qry_a, n_ligands, seed)
    ref = restrict_types(
        build_structure_ref(load_xy(ref_a), ref_a.obs["cell_type"].astype(str).tolist(), radius),
        shared,
    )
    qry_xy = load_xy(qry_a)
    qry_labels = qry_a.obs["cell_type"].astype(str).tolist()
    qry_expr = load_expr(qry_a, genes)
    truth = gaussian_received(qry_xy, qry_expr, radius)
    mapped = np.array([ref["cell_types"].index(t) for t in qry_labels], dtype=np.int64)
    means = type_mean_expr(qry_expr, mapped, len(ref["cell_types"]))
    pooled = infer_from_structure(ref["mean_weight_mass"], mapped, means)
    abund = infer_from_structure(
        abundance_baseline(ref["mean_weight_mass"], ref["ref_type_counts"]), mapped, means
    )
    qry_ref = build_structure_ref(qry_xy, qry_labels, radius)
    qry_names, qry_codes = type_maps(qry_labels)
    self_pooled = infer_from_structure(
        qry_ref["mean_weight_mass"], qry_codes, type_mean_expr(qry_expr, qry_codes, len(qry_names))
    )
    sm_cos = structure_matrix_cosine(
        ref["mean_weight_mass"], restrict_types(qry_ref, shared)["mean_weight_mass"]
    )
    rows = []
    for method, pred in [
        ("structure_transfer", pooled),
        ("abundance_baseline", abund),
        ("query_self_structure", self_pooled),
    ]:
        m = score(pred, truth)
        tl = type_level_metrics(pred, truth, qry_codes)
        rows.append(
            metrics_to_row(
                method,
                "spatial_holdout_transfer",
                path.name,
                m,
                {
                    "type_pearson_mean": tl["pearson_mean"],
                    "type_mae": tl["mae"],
                    "structure_matrix_cosine": sm_cos,
                },
            )
        )
    return rows


def evaluate_same_sample(
    path: Path,
    radius: float,
    n_ligands: int,
    max_cells: Optional[int],
    seed: int,
) -> List[Dict]:
    a = ad.read_h5ad(path)
    if max_cells and a.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(a.n_obs, size=max_cells, replace=False))
        a = a[sel].copy()
    labels = a.obs["cell_type"].astype(str).tolist()
    xy = load_xy(a)
    genes = pick_ligands(a, n_ligands, seed)
    expr = load_expr(a, genes)
    truth = gaussian_received(xy, expr, radius)
    ref = build_structure_ref(xy, labels, radius)
    names, codes = type_maps(labels)
    assert names == ref["cell_types"]
    means = type_mean_expr(expr, codes, len(names))
    oracle = ref["_cell_weight"] @ means
    pooled = infer_from_structure(ref["mean_weight_mass"], codes, means)
    abund_w = abundance_baseline(ref["mean_weight_mass"], ref["ref_type_counts"])
    abund = infer_from_structure(abund_w, codes, means)
    # Leave-one-out style expression match within the same sample (drop self by kNN from others).
    matched_w = expression_matched_weights(
        ref["_cell_weight"], codes, expr, expr, codes, k=15, exclude_self=True
    )
    # Zero out trivial self-match dominance: recompute with diagonal blocked via large negative sim
    # by using a simple leave-one-out: for each cell average k others.
    matched = matched_w @ means
    soft_pred = ref["mean_soft_counts"][codes]
    hard_pred = ref["mean_hard_counts"][codes]
    rows = []
    for method, pred, soft_p, hard_p in [
        ("type_mean_oracle", oracle, soft_pred, hard_pred),
        ("structure_pooled", pooled, soft_pred, hard_pred),
        ("expression_matched", matched, soft_pred, hard_pred),
        ("abundance_baseline", abund, soft_pred, hard_pred),
    ]:
        m = score(
            pred,
            truth,
            soft_pred=soft_p,
            soft_truth=ref["_cell_soft"],
            hard_pred=hard_p,
            hard_truth=ref["_cell_hard"],
        )
        tl = type_level_metrics(pred, truth, codes)
        rows.append(
            metrics_to_row(
                method,
                "same_sample",
                path.name,
                m,
                {
                    "type_pearson_mean": tl["pearson_mean"],
                    "type_mae": tl["mae"],
                    "structure_matrix_cosine": 1.0 if method != "abundance_baseline" else structure_matrix_cosine(abund_w, ref["mean_weight_mass"]),
                },
            )
        )
    return rows


def evaluate_transfer(
    ref_path: Path,
    query_path: Path,
    radius: float,
    n_ligands: int,
    max_cells: Optional[int],
    seed: int,
    experiment: str,
) -> List[Dict]:
    ref_a = ad.read_h5ad(ref_path)
    qry_a = ad.read_h5ad(query_path)
    if max_cells:
        rng = np.random.default_rng(seed)
        if ref_a.n_obs > max_cells:
            ref_a = ref_a[np.sort(rng.choice(ref_a.n_obs, max_cells, replace=False))].copy()
        if qry_a.n_obs > max_cells:
            qry_a = qry_a[np.sort(rng.choice(qry_a.n_obs, max_cells, replace=False))].copy()

    shared_genes = [g for g in pick_ligands(qry_a, n_ligands * 2, seed) if g in ref_a.var_names]
    if len(shared_genes) < 5:
        shared = sorted(set(ref_a.var_names) & set(qry_a.var_names))
        # fall back: high mean on query among shared
        tmp = qry_a[:, shared]
        means = np.asarray(tmp.X.mean(axis=0)).ravel() if sparse.issparse(tmp.X) else tmp.X.mean(0)
        shared_genes = [shared[i] for i in np.argsort(-means)[:n_ligands]]
    else:
        shared_genes = shared_genes[:n_ligands]

    ref_labels = ref_a.obs["cell_type"].astype(str).tolist()
    qry_labels = qry_a.obs["cell_type"].astype(str).tolist()
    shared_types = sorted(set(ref_labels) & set(qry_labels))
    if len(shared_types) < 2:
        return [
            {
                "experiment": experiment,
                "dataset": f"{ref_path.stem}→{query_path.stem}",
                "method": "structure_transfer",
                "error": f"insufficient overlapping cell types ({len(shared_types)})",
            }
        ]

    ref_mask = np.isin(ref_labels, shared_types)
    qry_mask = np.isin(qry_labels, shared_types)
    ref_a = ref_a[ref_mask].copy()
    qry_a = qry_a[qry_mask].copy()
    ref_labels = ref_a.obs["cell_type"].astype(str).tolist()
    qry_labels = qry_a.obs["cell_type"].astype(str).tolist()

    ref = restrict_types(
        build_structure_ref(load_xy(ref_a), ref_labels, radius), shared_types
    )
    qry_xy = load_xy(qry_a)
    qry_expr = load_expr(qry_a, shared_genes)
    truth = gaussian_received(qry_xy, qry_expr, radius)
    qry_names, qry_codes = type_maps(qry_labels)
    # Remap query codes onto ref type order
    ref_idx = {t: i for i, t in enumerate(ref["cell_types"])}
    mapped = np.array([ref_idx[t] for t in qry_labels], dtype=np.int64)
    means = type_mean_expr(qry_expr, mapped, len(ref["cell_types"]))
    pooled = infer_from_structure(ref["mean_weight_mass"], mapped, means)
    abund_w = abundance_baseline(ref["mean_weight_mass"], ref["ref_type_counts"])
    abund = infer_from_structure(abund_w, mapped, means)

    ref_expr = load_expr(ref_a, shared_genes)
    ref_names, ref_codes = type_maps(ref_labels)
    # Map ref codes onto shared ref["cell_types"] order
    ref_type_pos = {t: i for i, t in enumerate(ref["cell_types"])}
    ref_mapped = np.array([ref_type_pos[t] for t in ref_labels], dtype=np.int64)
    # Rebuild cell weights on shared type vocabulary
    ref_full = build_structure_ref(load_xy(ref_a), ref_labels, radius)
    # Restrict cell weight columns to shared types
    old_to_new = [ref_full["cell_types"].index(t) for t in ref["cell_types"]]
    ref_cell_w = ref_full["_cell_weight"][:, old_to_new]
    matched_w = expression_matched_weights(
        ref_cell_w, ref_mapped, ref_expr, qry_expr, mapped, k=15, exclude_self=False
    )
    matched = matched_w @ means

    # Same-sample structure on query for ceiling comparison
    qry_ref = build_structure_ref(qry_xy, qry_labels, radius)
    qry_means = type_mean_expr(qry_expr, qry_codes, len(qry_names))
    self_pooled = infer_from_structure(qry_ref["mean_weight_mass"], qry_codes, qry_means)

    soft_truth = qry_ref["_cell_soft"]
    soft_pred = np.zeros_like(soft_truth)
    hard_pred = np.zeros_like(qry_ref["_cell_hard"])
    for i, t in enumerate(qry_names):
        j = ref["cell_types"].index(t)
        soft_pred[:, i] = ref["mean_soft_counts"][mapped, j]
        hard_pred[:, i] = ref["mean_hard_counts"][mapped, j]

    # Structure matrix agreement (type×type)
    qry_st_shared = restrict_types(qry_ref, shared_types)
    sm_cos = structure_matrix_cosine(ref["mean_weight_mass"], qry_st_shared["mean_weight_mass"])

    rows = []
    for method, pred in [
        ("structure_transfer", pooled),
        ("expression_matched_transfer", matched),
        ("abundance_baseline", abund),
        ("query_self_structure", self_pooled),
    ]:
        m = score(
            pred,
            truth,
            soft_pred=soft_pred if "transfer" in method else qry_ref["mean_soft_counts"][qry_codes],
            soft_truth=soft_truth,
            hard_pred=hard_pred if "transfer" in method else qry_ref["mean_hard_counts"][qry_codes],
            hard_truth=qry_ref["_cell_hard"],
        )
        tl = type_level_metrics(pred, truth, qry_codes)
        rows.append(
            metrics_to_row(
                method,
                experiment,
                f"{ref_path.stem}→{query_path.stem}",
                m,
                {
                    "type_pearson_mean": tl["pearson_mean"],
                    "type_mae": tl["mae"],
                    "structure_matrix_cosine": sm_cos,
                },
            )
        )
    return rows


def write_report(df: pd.DataFrame, outdir: Path, radius: float, n_ligands: int) -> None:
    lines = [
        "# Tissue-structure received-ligand validation",
        "",
        f"- Gaussian radius: **{radius}**",
        f"- Ligands per dataset: **{n_ligands}** (highly expressed / variable genes)",
        "",
        "## Method",
        "",
        "Spatial ground truth uses SpaceTravLR's received-ligand kernel",
        "`(1/N) Σ_j scale·exp(-d²/2r²)·expr[j,l]`.",
        "Structure inference replaces per-cell Gaussian weights with type-conditional",
        "expectations `Ŝ[receiver,sender]` learned from a spatial reference, then",
        "multiplies by query type-mean ligand expression.",
        "",
        "### Error decomposition",
        "",
        "| Method | What it tests |",
        "|---|---|",
        "| `type_mean_oracle` | True per-cell structure × type-mean expression (heterogeneity ceiling) |",
        "| `structure_pooled` | Same-sample type-averaged neighborhoods |",
        "| `expression_matched` | Expression-kNN niche matching to transfer per-cell S |",
        "| `abundance_baseline` | Type frequencies only (no spatial architecture) |",
        "| `structure_transfer` | Cross-sample / cross-replicate structure reuse |",
        "| `expression_matched_transfer` | Cross-sample expression-matched niche transfer |",
        "| `query_self_structure` | Upper bound using the query's own spatial structure |",
        "",
        "## Results",
        "",
    ]
    same = df[df["experiment"] == "same_sample"].copy()
    if not same.empty:
        lines.append("### Same-sample recovery")
        lines.append("")
        pivot = same.pivot_table(
            index=["dataset", "method"],
            values=["pearson_mean", "type_pearson_mean", "spearman_mean", "mae", "rel_mae", "soft_cosine", "slope"],
            aggfunc="first",
        ).reset_index()
        lines.append(pivot.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        lines.append("")
        lines.append("```")
        lines.append(pivot.to_csv(index=False))
        lines.append("```")
        lines.append("")
        lines.append(
            "Note: `type_pearson_mean` scores type-averaged received ligands "
            "(the natural estimand without coordinates). Cell-level Pearson is "
            "limited by irreducible within-type niche variation."
        )
        lines.append("")
    xfer = df[df["experiment"].str.contains("transfer", na=False)].copy()
    if not xfer.empty:
        lines.append("### Cross-sample structure transfer")
        lines.append("")
        pivot = xfer.pivot_table(
            index=["experiment", "dataset", "method"],
            values=["pearson_mean", "type_pearson_mean", "spearman_mean", "mae", "rel_mae", "soft_cosine", "structure_matrix_cosine", "slope"],
            aggfunc="first",
        ).reset_index()
        lines.append(pivot.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        lines.append("")
        lines.append("```")
        lines.append(pivot.to_csv(index=False))
        lines.append("```")
        lines.append("")
    lines.extend(
        [
            "## Interpretation notes",
            "",
            "- Pearson / Spearman near the `type_mean_oracle` means structure pooling loses little.",
            "- Gains over `abundance_baseline` show that tissue architecture (not just composition) matters.",
            "- Cross-replicate transfer should approach `query_self_structure` when tissues match.",
            "- `soft_cosine` / `hard_cosine` score inferred neighbor-type composition vs spatial truth.",
            "",
        ]
    )
    (outdir / "VALIDATION_REPORT.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/ix1/ylee/kor11/tools/SpaceTravLR/data"),
    )
    ap.add_argument("--outdir", type=Path, default=Path("results/structure_validation"))
    ap.add_argument("--radius", type=float, default=200.0)
    ap.add_argument("--n-ligands", type=int, default=40)
    ap.add_argument("--max-cells", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "kidney_rep1": args.data_dir / "XYZeqV2_mouse_kidney_replicate_1.h5ad",
        "kidney_rep2": args.data_dir / "XYZeqV2_mouse_kidney_replicate_2.h5ad",
        "tonsil": args.data_dir / "Slidetags_human_tonsil.h5ad",
        "melanoma": args.data_dir / "Slidetags_human_melanoma.h5ad",
        "lymphnode": args.data_dir / "SlideSeqV2_mouse_lymphnode.h5ad",
        "gc": args.data_dir / "snrna_germinal_center.h5ad",
    }

    rows: List[Dict] = []
    t0 = time.time()
    for name, path in datasets.items():
        if not path.is_file():
            continue
        print(f"[same] {name} ...", flush=True)
        rows.extend(
            evaluate_same_sample(path, args.radius, args.n_ligands, args.max_cells, args.seed)
        )
        print(f"[holdout] {name} ...", flush=True)
        rows.extend(
            evaluate_spatial_holdout(path, args.radius, args.n_ligands, args.max_cells, args.seed)
        )

    transfers = [
        ("replicate_transfer", datasets["kidney_rep1"], datasets["kidney_rep2"]),
        ("replicate_transfer", datasets["kidney_rep2"], datasets["kidney_rep1"]),
        ("related_lymphoid_transfer", datasets["lymphnode"], datasets["gc"]),
        ("related_lymphoid_transfer", datasets["tonsil"], datasets["gc"]),
        ("mismatched_transfer", datasets["tonsil"], datasets["melanoma"]),
        ("mismatched_transfer", datasets["kidney_rep1"], datasets["tonsil"]),
    ]
    for exp, ref_p, qry_p in transfers:
        if not (ref_p.is_file() and qry_p.is_file()):
            continue
        print(f"[{exp}] {ref_p.stem} → {qry_p.stem} ...", flush=True)
        rows.extend(
            evaluate_transfer(
                ref_p,
                qry_p,
                args.radius,
                args.n_ligands,
                args.max_cells,
                args.seed,
                exp,
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "metrics.csv", index=False)
    summary = {
        "radius": args.radius,
        "n_ligands": args.n_ligands,
        "max_cells": args.max_cells,
        "elapsed_sec": time.time() - t0,
        "n_rows": len(df),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(df, args.outdir, args.radius, args.n_ligands)
    print(df.to_string(index=False))
    print(f"\nWrote {args.outdir}/metrics.csv and VALIDATION_REPORT.md")


if __name__ == "__main__":
    main()
