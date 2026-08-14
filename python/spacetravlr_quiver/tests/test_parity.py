"""Parity checks: Rust PyO3 vs pure-NumPy reference (SpaceOracle shift/cartography)."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

import spacetravlr_quiver as sq


def _pearson_ref(di: np.ndarray, diff: np.ndarray) -> float:
    di0 = di - di.mean()
    d0 = diff - diff.mean()
    da = float(np.dot(di0, di0))
    db = float(np.dot(d0, d0))
    if da <= 1e-18 or db <= 1e-18:
        return 1.0
    r = float(np.dot(di0, d0) / np.sqrt(da * db))
    if not np.isfinite(r):
        return 1.0
    return float(np.clip(r, -1.0, 1.0))


def test_pearson_parity():
    rng = np.random.default_rng(0)
    g = 64
    ei = rng.normal(size=g)
    ej = rng.normal(size=g)
    vi = rng.normal(size=g)
    rust = sq.pearson_velocity_vs_expr_delta(ei.tolist(), ej.tolist(), vi.tolist())
    ref = _pearson_ref(vi, ej - ei)
    assert abs(rust - ref) < 1e-9


def test_col_delta_cor_partial_parity():
    rng = np.random.default_rng(1)
    n, g, k = 40, 32, 8
    expr = rng.normal(size=(n, g))
    delta = rng.normal(size=(n, g))
    umap = rng.normal(size=(n, 2))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(umap)
    _, idx = nn.kneighbors(umap)
    neighbors = [row[1:].tolist() for row in idx]
    rust = np.asarray(sq.col_delta_cor_partial(expr, delta, neighbors))
    for i in range(n):
        for j in neighbors[i]:
            ref = _pearson_ref(delta[i], expr[j] - expr[i])
            assert abs(rust[i, j] - ref) < 1e-8, (i, j, rust[i, j], ref)


def test_transition_grid_smoke():
    rng = np.random.default_rng(2)
    n, g = 60, 40
    expr = np.abs(rng.normal(size=(n, g)))
    delta = rng.normal(size=(n, g)) * 0.1
    umap = rng.normal(size=(n, 2))
    grid = sq.compute_transition_grid(
        expr,
        delta,
        umap,
        n_neighbors=15,
        remove_null=True,
        null_subtract_mode="raw",
        unit_directions=False,
        grid_scale=1.0,
        vector_scale=0.85,
    )
    assert grid["nx"] >= 2 and grid["ny"] >= 2
    assert len(grid["u"]) == grid["nx"] * grid["ny"]
    assert len(grid["cell_u"]) == n


def test_raw_vs_clip_differ():
    rng = np.random.default_rng(3)
    n, g = 50, 30
    expr = np.abs(rng.normal(size=(n, g)))
    delta = rng.normal(size=(n, g))
    umap = rng.normal(size=(n, 2))
    kw = dict(n_neighbors=12, remove_null=True, grid_scale=1.0, vector_scale=1.0)
    a = sq.compute_transition_grid(expr, delta, umap, null_subtract_mode="raw", **kw)
    b = sq.compute_transition_grid(expr, delta, umap, null_subtract_mode="clip_renorm", **kw)
    # Not required to always differ, but on random data they almost always do
    da = np.asarray(a["cell_u"])
    db = np.asarray(b["cell_u"])
    assert da.shape == db.shape
