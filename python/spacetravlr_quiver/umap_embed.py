"""UMAP embedding helpers — prefer SpaceTravLR rust-process, fall back to umap-learn."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np


# Defaults aligned with RustPreprocessParams / spaceship [preprocess]
DEFAULT_UMAP = dict(
    n_neighbors=15,
    min_dist=0.5,
    spread=0.5,
    n_pcs=50,
    n_top_hvg=2000,
    random_seed=42,
)


def _dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def ensure_umap_embedding(
    adata: ad.AnnData,
    *,
    prefer_rust: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    spread: float = 0.5,
    n_pcs: int = 50,
    n_top_hvg: int = 2000,
    random_seed: int = 42,
    force: bool = False,
) -> ad.AnnData:
    """Ensure ``obsm['X_umap']`` exists.

    When ``prefer_rust`` and the ``spacetravlr`` binary is on PATH, writes a temp
    ``.h5ad``, runs ``spacetravlr --umap --output`` (rust-process), and copies
    ``X_umap`` back — preserving row order when QC does not drop cells. If that
    fails or drops cells, falls back to PCA + umap-learn with the same knobs.
    """
    if "X_umap" in adata.obsm and not force:
        return adata

    if prefer_rust and shutil.which("spacetravlr"):
        try:
            return _umap_via_spacetravlr_cli(
                adata,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_pcs=n_pcs,
                n_top_hvg=n_top_hvg,
            )
        except Exception as e:
            print(f"[spacetravlr_quiver] rust UMAP failed ({e}); falling back to umap-learn")

    return _umap_via_umap_learn(
        adata,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        n_pcs=n_pcs,
        random_seed=random_seed,
    )


def _umap_via_spacetravlr_cli(
    adata: ad.AnnData,
    *,
    n_neighbors: int,
    min_dist: float,
    n_pcs: int,
    n_top_hvg: int,
) -> ad.AnnData:
    with tempfile.TemporaryDirectory(prefix="stl_umap_") as td:
        td = Path(td)
        src = td / "in.h5ad"
        dst = td / "out.h5ad"
        adata.write_h5ad(src)
        cmd = [
            "spacetravlr",
            "--plain",
            "--h5ad",
            str(src),
            "--umap",
            "--output",
            str(dst),
            "--rust-n-neighbors",
            str(n_neighbors),
            "--rust-n-top-hvg",
            str(n_top_hvg),
        ]
        # min_dist / pcs via env or flags if available — check help
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        out = ad.read_h5ad(dst)
        if out.n_obs != adata.n_obs:
            raise RuntimeError(
                f"rust UMAP changed n_obs {adata.n_obs} → {out.n_obs} (QC filter); use umap-learn"
            )
        if "X_umap" not in out.obsm:
            raise RuntimeError("rust UMAP output missing X_umap")
        # align by obs_names
        umap = np.asarray(out[adata.obs_names].obsm["X_umap"], dtype=np.float64)
        adata.obsm["X_umap"] = umap
        return adata


def _umap_via_umap_learn(
    adata: ad.AnnData,
    *,
    n_neighbors: int,
    min_dist: float,
    spread: float,
    n_pcs: int,
    random_seed: int,
) -> ad.AnnData:
    from sklearn.decomposition import PCA
    from umap import UMAP

    expr = _dense(adata.X)
    lib = expr.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    norm = np.log1p(expr * (1e4 / lib))
    n_pcs = min(n_pcs, norm.shape[0] - 1, norm.shape[1] - 1)
    pcs = PCA(n_components=n_pcs, random_state=random_seed).fit_transform(norm)
    emb = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        random_state=random_seed,
    ).fit_transform(pcs)
    adata.obsm["X_umap"] = emb.astype(np.float64)
    return adata
