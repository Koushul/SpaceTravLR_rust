#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scipy",
#     "numba",
#     "scikit-learn",
#     "torch",
#     "group-lasso",
#     "psutil",
# ]
# ///
"""
Per-step microbenchmarks for the **Python** SpaceTravLR equivalents of the
Rust hot paths exercised by `src/bin/bench_steps.rs`. Each invocation runs
exactly one step on a synthetic dataset of the requested size and prints a
single JSON line so the driver script can aggregate Rust and Python timings
side-by-side.

Steps mirror the Rust binary:
  - received_ligands   : numba `_gaussian_kernel_2d_batch` + `_weighted_mean`.
  - spatial_features   : `scipy.spatial.distance.cdist` neighbor counts.
  - xyc2spatial        : numba `xyc2spatial_fast` (distance maps).
  - group_lasso        : `group_lasso.GroupLasso` fit per cluster.
  - train_one_gene     : tiny PyTorch CNN on a single-cluster minibatch loop.

The reference Python implementations are imported from the upstream
``jishnu-lab/SpaceTravLR`` package whenever available, otherwise local copies
of the same code paths are used so that the script is self-contained.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np


def peak_rss_mb() -> float | None:
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            return ru.ru_maxrss / (1024.0 * 1024.0)
        return ru.ru_maxrss / 1024.0
    except Exception:
        return None


def make_xy(rng: np.random.Generator, n: int, extent: float) -> np.ndarray:
    half = extent * 0.5
    return rng.uniform(-half, half, size=(n, 2)).astype(np.float64)


def make_clusters(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    return rng.integers(low=0, high=max(k, 1), size=n).astype(np.int64)


def make_dense(rng: np.random.Generator, n: int, p: int) -> np.ndarray:
    return np.abs(rng.standard_normal(size=(n, p))).astype(np.float64)


# ── Reference implementations (local fallbacks of the upstream code) ──────────


def _ref_received_ligands_impl(xy: np.ndarray, lig: np.ndarray, radius: float, scale_factor: float) -> np.ndarray:
    try:
        from SpaceTravLR.models.parallel_estimators import (
            _gaussian_kernel_2d_batch,
            _weighted_mean,
        )
    except Exception:
        from numba import njit, prange

        @njit(parallel=True)
        def _gaussian_kernel_2d_batch(xy, radius):
            n = xy.shape[0]
            W = np.empty((n, n), dtype=np.float64)
            inv_2r2 = -1.0 / (2.0 * radius * radius)
            for i in prange(n):
                xi, yi = xy[i, 0], xy[i, 1]
                for j in range(n):
                    dx = xi - xy[j, 0]
                    dy = yi - xy[j, 1]
                    W[i, j] = np.exp((dx * dx + dy * dy) * inv_2r2)
            return W

        @njit(parallel=True)
        def _weighted_mean(W, lig_values):
            n = W.shape[0]
            n_lig = lig_values.shape[1]
            out = np.zeros((n, n_lig), dtype=np.float64)
            for i in prange(n):
                for k in range(n):
                    w = W[i, k]
                    for j in range(n_lig):
                        out[i, j] += w * lig_values[k, j]
                for j in range(n_lig):
                    out[i, j] /= n
            return out

    W = scale_factor * _gaussian_kernel_2d_batch(np.ascontiguousarray(xy, dtype=np.float64), float(radius))
    return _weighted_mean(W, np.ascontiguousarray(lig, dtype=np.float64))


def _ref_spatial_features_impl(xy: np.ndarray, clusters: np.ndarray, radius: float) -> np.ndarray:
    from scipy.spatial.distance import cdist

    unique_celltypes = np.unique(clusters)
    coords = xy
    distances = cdist(coords, coords)
    result = np.zeros((len(coords), len(unique_celltypes)))
    for i, celltype in enumerate(unique_celltypes):
        mask = clusters == celltype
        neighbors = (distances <= radius)[:, mask]
        result[:, i] = np.sum(neighbors, axis=1)
    return result


def _ref_xyc2spatial_impl(xy: np.ndarray, clusters: np.ndarray, m: int, n: int) -> np.ndarray:
    try:
        from SpaceTravLR.models.spatial_map import xyc2spatial_fast
    except Exception:
        from numba import njit, prange

        @njit
        def generate_grid_centers(m, n, xmin, xmax, ymin, ymax):
            centers = []
            cell_width = (xmax - xmin) / n
            cell_height = (ymax - ymin) / m
            for i in range(m):
                for j in range(n):
                    x = xmin + (j + 0.5) * cell_width
                    y = ymax - (i + 0.5) * cell_height
                    centers.append((x, y))
            return centers

        @njit
        def distance(p1, p2):
            return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        @njit(parallel=True)
        def xyc2spatial_fast(xyc, m, n):
            x, y = xyc[:, 0], xyc[:, 1]
            xmin, xmax, ymin, ymax = np.min(x), np.max(x), np.min(y), np.max(y)
            centers = generate_grid_centers(m, n, xmin, xmax, ymin, ymax)
            clusters = np.unique(xyc[:, 2]).astype(np.int32)
            num_clusters = len(clusters)
            spatial_maps = np.zeros((len(xyc), num_clusters, m, n), dtype=np.float32)
            mask = np.ones((num_clusters, m, n), dtype=np.float32)
            for s in prange(len(xyc)):
                x_, y_, _cluster = xyc[s]
                dist_map = np.array([distance((x_, y_), c) for c in centers]).reshape(m, n)
                for i in range(num_clusters):
                    spatial_maps[s, i] = dist_map
            channel_wise_maps = np.empty_like(spatial_maps)
            for s in prange(len(xyc)):
                for i in range(num_clusters):
                    for a in range(m):
                        for b in range(n):
                            channel_wise_maps[s, i, a, b] = spatial_maps[s, i, a, b] * mask[i, a, b]
            min_vals = np.zeros((len(xyc), num_clusters, 1, 1), dtype=np.float32)
            max_vals = np.zeros((len(xyc), num_clusters, 1, 1), dtype=np.float32)
            for s in prange(len(xyc)):
                for i in range(num_clusters):
                    min_vals[s, i, 0, 0] = np.min(channel_wise_maps[s, i])
                    max_vals[s, i, 0, 0] = np.max(channel_wise_maps[s, i])
            denom = np.maximum(max_vals - min_vals, 1e-15)
            out = (channel_wise_maps - min_vals) / denom
            return out

    xyc = np.column_stack([xy[:, 0], xy[:, 1], clusters.astype(np.float64)]).astype(np.float64)
    return xyc2spatial_fast(xyc, m, n)


# ── Step runners ──────────────────────────────────────────────────────────────


def _warmup_received_ligands() -> None:
    rng = np.random.default_rng(0)
    xy = make_xy(rng, 32, 100.0)
    lig = make_dense(rng, 32, 4)
    _ref_received_ligands_impl(xy, lig, 50.0, 1.0)


def step_received_ligands(args: argparse.Namespace) -> float:
    rng = np.random.default_rng(args.seed)
    xy = make_xy(rng, args.n_cells, args.extent)
    lig = make_dense(rng, args.n_cells, args.n_ligands)
    _warmup_received_ligands()
    t0 = time.perf_counter()
    _ref_received_ligands_impl(xy, lig, args.radius, 1.0)
    return time.perf_counter() - t0


def step_spatial_features(args: argparse.Namespace) -> float:
    rng = np.random.default_rng(args.seed)
    xy = make_xy(rng, args.n_cells, args.extent)
    clusters = make_clusters(rng, args.n_cells, args.n_clusters)
    t0 = time.perf_counter()
    _ref_spatial_features_impl(xy, clusters, args.radius)
    return time.perf_counter() - t0


def _warmup_xyc2spatial() -> None:
    rng = np.random.default_rng(0)
    xy = make_xy(rng, 16, 100.0)
    clusters = make_clusters(rng, 16, 4)
    _ref_xyc2spatial_impl(xy, clusters, 8, 8)


def step_xyc2spatial(args: argparse.Namespace) -> float:
    rng = np.random.default_rng(args.seed)
    xy = make_xy(rng, args.n_cells, args.extent)
    clusters = make_clusters(rng, args.n_cells, args.n_clusters)
    _warmup_xyc2spatial()
    t0 = time.perf_counter()
    _ref_xyc2spatial_impl(xy, clusters, args.spatial_dim, args.spatial_dim)
    return time.perf_counter() - t0


def step_group_lasso(args: argparse.Namespace) -> float:
    """
    Per-cluster sparse-group lasso fit, mirroring the Rust ClusteredGroupLasso.
    """
    from group_lasso import GroupLasso

    rng = np.random.default_rng(args.seed)
    n = args.n_cells
    p = args.n_features
    x = make_dense(rng, n, p)
    beta = np.zeros(p)
    beta[: p // 4] = 0.5 * rng.standard_normal(p // 4)
    y = (x @ beta + 0.01 * rng.standard_normal(n)).reshape(-1, 1)
    clusters = rng.integers(0, args.n_clusters, size=n)

    groups = np.arange(p, dtype=np.int64)
    t0 = time.perf_counter()
    for cid in range(args.n_clusters):
        mask = clusters == cid
        if mask.sum() < 4:
            continue
        model = GroupLasso(
            groups=groups,
            group_reg=1e-4,
            l1_reg=1e-4,
            n_iter=args.n_iter,
            tol=1e-4,
            scale_reg=None,
            supress_warning=True,
            fit_intercept=True,
        )
        model.fit(x[mask], y[mask])
    return time.perf_counter() - t0


def step_train_one_gene(args: argparse.Namespace) -> float:
    """
    Tiny end-to-end CNN fit on synthetic data. Mirrors the Rust
    `train_cluster_cnn_epochs` shape and minibatch behaviour.
    """
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(args.seed)
    modulators = min(args.n_features, 32)
    n = args.n_cells
    x = make_dense(rng, n, modulators)
    beta_true = np.zeros(modulators)
    beta_true[: modulators // 2] = 0.3 * rng.standard_normal(modulators // 2)
    y = x @ beta_true + 0.05 * rng.standard_normal(n)

    clusters = make_clusters(rng, n, args.n_clusters)
    cluster_id = 0
    idx = np.where(clusters == cluster_id)[0]
    if idx.size == 0:
        return 0.0
    n_c = idx.size
    h = w = args.spatial_dim
    sm_c = np.full((n_c, 1, h, w), 0.05, dtype=np.float32)
    x_c = x[idx].astype(np.float32)
    y_c = y[idx].astype(np.float32)
    sf_c = np.zeros((n_c, args.n_clusters), dtype=np.float32)
    sf_c[:, cluster_id] = 1.0

    device = torch.device("cpu")
    sm_t = torch.from_numpy(sm_c).to(device)
    x_t = torch.from_numpy(x_c).to(device)
    y_t = torch.from_numpy(y_c).to(device)
    sf_t = torch.from_numpy(sf_c).to(device)

    class TinyCNN(nn.Module):
        def __init__(self, n_modulators, n_clusters, h, w):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.head = nn.Sequential(
                nn.Linear(8 * 4 * 4 + n_modulators + n_clusters, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, sm, mods, sf):
            v = self.conv(sm).flatten(1)
            return self.head(torch.cat([v, mods, sf], dim=1)).squeeze(-1)

    model = TinyCNN(modulators, args.n_clusters, h, w).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=4e-4)
    loss_fn = nn.MSELoss()

    bs = min(64, n_c)
    n_batches = max(1, n_c // bs)
    t0 = time.perf_counter()
    for _ in range(args.epochs):
        perm = torch.randperm(n_c)
        for b in range(n_batches):
            sl = perm[b * bs : (b + 1) * bs]
            opt.zero_grad()
            pred = model(sm_t[sl], x_t[sl], sf_t[sl])
            loss = loss_fn(pred, y_t[sl])
            loss.backward()
            opt.step()
    return time.perf_counter() - t0


STEPS = {
    "received_ligands": step_received_ligands,
    "spatial_features": step_spatial_features,
    "xyc2spatial": step_xyc2spatial,
    "group_lasso": step_group_lasso,
    "train_one_gene": step_train_one_gene,
}


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--step", required=True, choices=sorted(STEPS.keys()))
    p.add_argument("--n-cells", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-ligands", type=int, default=32)
    p.add_argument("--n-features", type=int, default=64)
    p.add_argument("--n-clusters", type=int, default=8)
    p.add_argument("--extent", type=float, default=5_000.0)
    p.add_argument("--radius", type=float, default=300.0)
    p.add_argument("--spatial-dim", type=int, default=24)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--repeats", type=int, default=1)
    args = p.parse_args()

    fn = STEPS[args.step]
    times: list[float] = []
    for _ in range(max(1, args.repeats)):
        times.append(fn(args))

    out: dict[str, Any] = {
        "impl": "python",
        "step": args.step,
        "n_cells": args.n_cells,
        "n_features": args.n_features,
        "n_ligands": args.n_ligands,
        "n_clusters": args.n_clusters,
        "spatial_dim": args.spatial_dim,
        "epochs": args.epochs,
        "n_iter": args.n_iter,
        "seed": args.seed,
        "repeats": args.repeats,
        "wall_s": median(times),
        "wall_s_runs": times,
        "peak_rss_mb": peak_rss_mb(),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
