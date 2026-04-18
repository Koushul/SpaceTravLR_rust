"""Shared helpers for the NicheCompass / Banksy microniche benchmark."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "h5ad"
EXP_DIR = REPO_ROOT / "experiments" / "niche_benchmark"
LABELS_DIR = EXP_DIR / "labels"
RESULTS_DIR = EXP_DIR / "results"
FIG_DIR = EXP_DIR / "figures"

DEFAULT_DATASET = "SlideTags_human_tonsil.h5ad"
GROUND_TRUTH_KEY = "cell_type_2"
SPATIAL_KEY = "spatial"


def load_dataset(name: str = DEFAULT_DATASET, *, ground_truth_key: str = GROUND_TRUTH_KEY) -> ad.AnnData:
    """Load AnnData and ensure it has a spatial coord matrix and the GT label column."""
    path = DATA_DIR / name
    adata = sc.read_h5ad(path)
    if SPATIAL_KEY not in adata.obsm:
        raise ValueError(f"{name} has no obsm['{SPATIAL_KEY}']")
    if ground_truth_key not in adata.obs.columns:
        raise ValueError(
            f"{name} missing ground-truth obs '{ground_truth_key}'. "
            f"Have: {list(adata.obs.columns)}"
        )
    adata.obs[ground_truth_key] = adata.obs[ground_truth_key].astype("category")
    adata.obs["microniche_gt"] = adata.obs[ground_truth_key]
    coords = np.asarray(adata.obsm[SPATIAL_KEY], dtype=float)
    adata.obs["x"] = coords[:, 0]
    adata.obs["y"] = coords[:, 1]
    return adata


def basic_qc(adata: ad.AnnData, *, min_counts: int = 200, min_genes_per_cell: int = 50,
             min_cells_per_gene: int = 5) -> ad.AnnData:
    """Light QC; spatial datasets are sparse so thresholds are mild."""
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, min_genes=min_genes_per_cell)
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
    return adata


def add_normalized_layers(adata: ad.AnnData) -> ad.AnnData:
    """Store raw counts in layers['counts'] and produce log-normalized X.

    NicheCompass expects raw counts in `.layers['counts']`.
    Banksy operates on the log-normalized X.
    """
    X = adata.X
    if sp.issparse(X):
        adata.layers["counts"] = X.copy().astype(np.float32)
    else:
        adata.layers["counts"] = np.asarray(X, dtype=np.float32).copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def save_labels(name: str, obs_index: Iterable[str], labels: Iterable, *, runtime_sec: float | None = None,
                extra: dict | None = None) -> Path:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"cell_id": list(obs_index), "label": list(labels)})
    out = LABELS_DIR / f"{name}.csv"
    df.to_csv(out, index=False)
    meta = {"name": name, "n_cells": int(len(df)), "n_labels": int(df["label"].nunique())}
    if runtime_sec is not None:
        meta["runtime_sec"] = runtime_sec
    if extra:
        meta.update(extra)
    (LABELS_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return out


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def stop(self) -> float:
        return time.perf_counter() - self.t0
