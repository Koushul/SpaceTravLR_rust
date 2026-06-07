"""Per-cluster group lasso anchors (simplified SpaceTravLR [lasso] section)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import ElasticNet


@dataclass
class ClusterLassoFit:
    cluster_id: int
    intercept: float
    coef: np.ndarray
    r2: float
    n_cells: int


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    if ss_tot <= 0:
        return 0.0
    return max(1.0 - ss_res / ss_tot, -1e6)


def scale_modulators(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    std = x.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return x / std, std


def fit_cluster_lasso(
    y: np.ndarray,
    x_mod: np.ndarray,
    clusters: np.ndarray,
    cluster_id: int,
    l1_reg: float = 1e-4,
    min_cells: int = 16,
) -> ClusterLassoFit | None:
    mask = clusters == cluster_id
    n = int(mask.sum())
    if n < min_cells:
        return None
    x_c = x_mod[mask].astype(np.float64)
    y_c = y[mask].astype(np.float64)
    x_scaled, std = scale_modulators(x_c)
    enet = ElasticNet(
        alpha=l1_reg,
        l1_ratio=0.95,
        fit_intercept=True,
        max_iter=2000,
        random_state=0,
    )
    enet.fit(x_scaled, y_c)
    coef = enet.coef_ / std
    pred = enet.intercept_ + x_scaled @ enet.coef_
    return ClusterLassoFit(
        cluster_id=cluster_id,
        intercept=float(enet.intercept_),
        coef=coef.astype(np.float64),
        r2=r2_score(y_c, pred),
        n_cells=n,
    )


def fit_gene_lasso_anchors(
    y: np.ndarray,
    x_mod: np.ndarray,
    clusters: np.ndarray,
    num_clusters: int,
    l1_reg: float = 1e-4,
    score_threshold: float = 0.1,
    min_cells: int = 16,
) -> dict[int, ClusterLassoFit]:
    fits: dict[int, ClusterLassoFit] = {}
    for c in range(num_clusters):
        fit = fit_cluster_lasso(y, x_mod, clusters, c, l1_reg=l1_reg, min_cells=min_cells)
        if fit is not None and fit.r2 >= score_threshold:
            fits[c] = fit
    return fits


def lasso_predict(
    x_mod: np.ndarray,
    intercept: float,
    coef: np.ndarray,
) -> np.ndarray:
    return intercept + x_mod @ coef


def mean_lasso_anchors(fits: dict[int, ClusterLassoFit], n_modulators: int) -> tuple[float, np.ndarray]:
    if not fits:
        return 0.0, np.zeros(n_modulators, dtype=np.float64)
    intercepts = [f.intercept for f in fits.values()]
    coefs = np.stack([f.coef for f in fits.values()], axis=0)
    return float(np.mean(intercepts)), coefs.mean(axis=0)
