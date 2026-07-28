#!/usr/bin/env python3
"""Smoke tests for within-label host/bact imputation invariants."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parent / "06_add_imputed_layer.py"
spec = importlib.util.spec_from_file_location("impute06", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_neighbors_stay_in_label():
    rng = np.random.default_rng(0)
    xy = rng.normal(size=(200, 2))
    labels = np.array(["A"] * 100 + ["B"] * 100)
    xy[:100] += 0
    xy[100:] += 10
    by = mod.neighbor_index_by_label(xy, labels, k=8)
    for lab, (idx, ind) in by.items():
        assert ind.shape[1] == 8
        assert np.all(labels[idx][ind] == lab)
    print("ok neighbors_stay_in_label")


def test_fill_does_not_cross_labels():
    # Two well-separated clusters; only label A has signal on feature 0
    xy = np.vstack(
        [
            np.column_stack([np.linspace(0, 1, 40), np.zeros(40)]),
            np.column_stack([np.linspace(0, 1, 40) + 100, np.zeros(40)]),
        ]
    )
    labels = np.array(["A"] * 40 + ["B"] * 40)
    X = np.zeros((80, 2), dtype=np.float32)
    X[0:20, 0] = 5.0  # A has feature0
    X[40:60, 1] = 7.0  # B has feature1
    by = mod.neighbor_index_by_label(xy, labels, k=5)
    out, fills = mod.fill_zeros_with_precomputed(X, by)
    assert fills > 0
    # B cells must not receive feature0 from A
    assert float(out[40:, 0].max()) == 0.0
    # A cells must not receive feature1 from B
    assert float(out[:40, 1].max()) == 0.0
    # A zeros near signal get filled
    assert float(out[20:40, 0].max()) > 0.0
    print("ok fill_does_not_cross_labels fills=", fills)


def test_outputs_exist_schema():
    import anndata as ad
    import pandas as pd

    host = Path("/ix1/ylee/kor11/tools/spacetravlr_microbiome/processed/GSM9456850_tumor_cells_imputed.h5ad")
    bact = Path("/ix1/ylee/kor11/tools/spacetravlr_microbiome/processed/GSM9456850_bact_senders_colony25um.parquet")
    if host.exists():
        a = ad.read_h5ad(host, backed="r")
        assert "imputed_count" in a.layers
        meta = dict(a.uns.get("imputation", {}))
        print("host imputation meta:", meta or "(legacy file, rerun 06)")
    if bact.exists():
        df = pd.read_parquet(bact)
        print("bact columns:", df.columns.tolist()[:10], "n=", len(df))
        if "bact_label" in df.columns:
            print("bact_label nunique", df["bact_label"].nunique())


if __name__ == "__main__":
    test_neighbors_stay_in_label()
    test_fill_does_not_cross_labels()
    test_outputs_exist_schema()
    print("all smoke tests passed")
