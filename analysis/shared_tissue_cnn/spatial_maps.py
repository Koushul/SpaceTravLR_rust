"""Mirror SpaceTravLR spatial map generation (xyc2spatial_fast + create_spatial_features)."""

from __future__ import annotations

import numpy as np


def min_max_finite_col(col: np.ndarray) -> tuple[float, float]:
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(finite.min()), float(finite.max())


def xyc2spatial_fast(
    xy: np.ndarray,
    clusters: np.ndarray,
    num_clusters: int,
    m: int,
    n: int,
    ego_center: bool = False,
) -> np.ndarray:
    """Inverse-distance spatial maps, shape (n_cells, num_clusters, m, n)."""
    num_cells = xy.shape[0]
    x_col = xy[:, 0]
    y_col = xy[:, 1]
    xmin, xmax = min_max_finite_col(x_col)
    ymin, ymax = min_max_finite_col(y_col)

    span_x = max(xmax - xmin, 1e-6)
    span_y = max(ymax - ymin, 1e-6)
    cell_width = span_x / n
    cell_height = span_y / m

    cx_global = xmin + (np.arange(n) + 0.5) * cell_width
    cy_global = ymax - (np.arange(m) + 0.5) * cell_height

    spatial_maps = np.zeros((num_cells, num_clusters, m, n), dtype=np.float32)
    for s in range(num_cells):
        cluster_s = int(clusters[s])
        if cluster_s >= num_clusters:
            continue
        x_s = float(xy[s, 0])
        y_s = float(xy[s, 1])
        if not np.isfinite(x_s) or not np.isfinite(y_s):
            continue

        if ego_center:
            half_x = span_x * 0.5
            half_y = span_y * 0.5
            cx_grid = x_s - half_x + (np.arange(n) + 0.5) * cell_width
            top_y = y_s + half_y
            cy_grid = top_y - (np.arange(m) + 0.5) * cell_height
        else:
            cx_grid = cx_global
            cy_grid = cy_global

        channel_map = spatial_maps[s, cluster_s]
        for i in range(m):
            gy = cy_grid[i]
            if not np.isfinite(gy):
                continue
            dy2 = (y_s - gy) ** 2
            for j in range(n):
                gx = cx_grid[j]
                if not np.isfinite(gx):
                    continue
                dx2 = (x_s - gx) ** 2
                d = max(float(np.sqrt(dx2 + dy2)), 1e-6)
                channel_map[i, j] = 1.0 / d
    return spatial_maps


def create_spatial_features(
    xy: np.ndarray,
    clusters: np.ndarray,
    num_clusters: int,
    radius: float,
) -> np.ndarray:
    """Neighbor counts per cluster within radius, shape (n_cells, num_clusters)."""
    from scipy.spatial import cKDTree

    n = xy.shape[0]
    result = np.zeros((n, num_clusters), dtype=np.float64)
    r2 = radius * radius
    if r2 > 0 and np.isfinite(r2):
        r2 = np.nextafter(r2, np.inf)

    valid_idx: list[int] = []
    points: list[list[float]] = []
    for i in range(n):
        x, y = float(xy[i, 0]), float(xy[i, 1])
        if np.isfinite(x) and np.isfinite(y):
            valid_idx.append(i)
            points.append([x, y])

    if not points:
        return result

    tree = cKDTree(np.asarray(points, dtype=np.float64))
    for i in range(n):
        xi, yi = float(xy[i, 0]), float(xy[i, 1])
        if not np.isfinite(xi) or not np.isfinite(yi):
            continue
        neighbors = tree.query_ball_point([xi, yi], np.sqrt(r2))
        for nb in neighbors:
            j = valid_idx[nb]
            c = int(clusters[j])
            if c < num_clusters:
                result[i, c] += 1.0
    return result


def spatial_maps_for_cluster(
    spatial_maps: np.ndarray,
    row_indices: np.ndarray,
    cluster_id: int,
) -> np.ndarray:
    """Extract [batch, 1, H, W] maps for one cluster (SpaceTravLR convention)."""
    k = len(row_indices)
    if k == 0:
        h, w = spatial_maps.shape[2], spatial_maps.shape[3]
        return np.zeros((0, 1, h, w), dtype=np.float32)
    h, w = spatial_maps.shape[2], spatial_maps.shape[3]
    out = np.zeros((k, 1, h, w), dtype=np.float32)
    for out_i, src_i in enumerate(row_indices):
        out[out_i, 0] = spatial_maps[src_i, cluster_id]
    return out
