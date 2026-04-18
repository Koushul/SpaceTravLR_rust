"""Compute clustering metrics + spatial coherence for the saved label CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)
from sklearn.neighbors import kneighbors_graph

from _common import LABELS_DIR, RESULTS_DIR


def _load_predictions(name: str) -> pd.DataFrame:
    csv = LABELS_DIR / f"{name}.csv"
    df = pd.read_csv(csv)
    df["label"] = df["label"].astype(str)
    return df.set_index("cell_id")


def _spatial_purity(coords: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Mean fraction of k nearest spatial neighbors that share the cell's label.

    Higher = niches are more spatially contiguous (1.0 means every neighbor
    has the same label).
    """
    g = kneighbors_graph(coords, n_neighbors=k + 1, mode="connectivity", include_self=True)
    g = g.tocsr()
    purities = np.empty(coords.shape[0], dtype=float)
    for i in range(coords.shape[0]):
        nb = g.indices[g.indptr[i]:g.indptr[i + 1]]
        nb = nb[nb != i][:k]
        if len(nb) == 0:
            purities[i] = np.nan
            continue
        purities[i] = float(np.mean(labels[nb] == labels[i]))
    return float(np.nanmean(purities))


def _eval(name: str, gt: np.ndarray, coords: np.ndarray, pred: np.ndarray) -> dict:
    h, c, v = homogeneity_completeness_v_measure(gt, pred)
    return {
        "method": name,
        "n_cells": int(len(gt)),
        "n_clusters_pred": int(len(np.unique(pred))),
        "n_clusters_gt": int(len(np.unique(gt))),
        "ari": float(adjusted_rand_score(gt, pred)),
        "nmi": float(normalized_mutual_info_score(gt, pred)),
        "ami": float(adjusted_mutual_info_score(gt, pred)),
        "fmi": float(fowlkes_mallows_score(gt, pred)),
        "homogeneity": float(h),
        "completeness": float(c),
        "v_measure": float(v),
        "spatial_purity_k10": _spatial_purity(coords, pred, k=10),
    }


def main() -> Path:
    adata = sc.read_h5ad(RESULTS_DIR / "tonsil_prepared.h5ad")
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    gt = adata.obs["microniche_gt"].astype(str).to_numpy()
    print(f"GT cells: {len(gt)}, GT clusters: {len(np.unique(gt))}")

    rows = []
    for name in ["banksy", "nichecompass"]:
        path = LABELS_DIR / f"{name}.csv"
        if not path.exists():
            print(f"skip {name}: no labels file")
            continue
        df = _load_predictions(name)
        df = df.reindex(adata.obs_names)
        if df["label"].isna().any():
            print(f"warning: {name} has missing predictions for some cells; dropping")
        mask = ~df["label"].isna()
        pred = df.loc[mask, "label"].to_numpy()
        coords_m = coords[mask.to_numpy()]
        gt_m = gt[mask.to_numpy()]
        meta_path = LABELS_DIR / f"{name}.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        scores = _eval(name, gt_m, coords_m, pred)
        scores["runtime_sec"] = meta.get("runtime_sec")
        rows.append(scores)
        print(json.dumps(scores, indent=2))

    spatial_purity_gt = _spatial_purity(coords, gt, k=10)
    rows.insert(0, {
        "method": "ground_truth",
        "n_cells": int(len(gt)),
        "n_clusters_pred": int(len(np.unique(gt))),
        "n_clusters_gt": int(len(np.unique(gt))),
        "ari": 1.0,
        "nmi": 1.0,
        "ami": 1.0,
        "fmi": 1.0,
        "homogeneity": 1.0,
        "completeness": 1.0,
        "v_measure": 1.0,
        "spatial_purity_k10": spatial_purity_gt,
        "runtime_sec": None,
    })
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "metrics.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(df.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
