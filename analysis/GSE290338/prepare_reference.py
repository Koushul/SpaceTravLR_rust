#!/usr/bin/env python3
"""Prepare GSE179936 MC38 tumor immune scRNA-seq reference for MALT map-labels."""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

import anndata as ad
import scanpy as sc


def pick_label_column(adata: ad.AnnData) -> str:
    for col in (
        "singler_label",
        "cell_type",
        "celltype",
        "annotation",
        "CellType",
        "cell_types",
        "predicted_cell_type",
    ):
        if col in adata.obs and adata.obs[col].nunique() > 1:
            return col
    for col in adata.obs.columns:
        if adata.obs[col].dtype.name in ("category", "object"):
            n = adata.obs[col].nunique()
            if 3 <= n <= 80:
                return col
    raise ValueError("No suitable cell-type column found in reference obs")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gz",
        type=Path,
        default=Path(__file__).resolve().parent / "reference" / "GSE179936_LFD_samples.h5ad.gz",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "GSE179936_MC38_reference.h5ad",
    )
    p.add_argument("--max-cells", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    h5ad_path = args.gz.with_suffix("")
    if not h5ad_path.exists():
        with gzip.open(args.gz, "rb") as f_in, open(h5ad_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    adata = ad.read_h5ad(h5ad_path)
    adata.var_names_make_unique()
    if "singler_label" in adata.obs:
        label_col = "singler_label"
        adata = adata[adata.obs["singler_label"].notna()].copy()
        adata.obs["singler_label"] = adata.obs["singler_label"].astype(str)
        adata = adata[~adata.obs["singler_label"].isin({"nan", ""})].copy()
    else:
        label_col = pick_label_column(adata)
    print(f"Using obs[{label_col!r}] ({adata.obs[label_col].nunique()} types)")
    adata.obs["cell_type"] = adata.obs[label_col].astype(str)

    if adata.n_obs > args.max_cells:
        sc.pp.subsample(adata, n_obs=args.max_cells, random_state=args.seed)

    xmax = float(adata.X.max()) if adata.n_vars else 0.0
    if xmax > 50:
        if "counts" not in adata.layers:
            adata.layers["counts"] = adata.X.copy()
    else:
        for key in list(adata.layers.keys()):
            if key == "counts":
                del adata.layers[key]

    adata.var = adata.var[[c for c in adata.var.columns if c != "highly_variable"]]
    for col in list(adata.var.columns):
        if adata.var[col].dtype == bool:
            adata.var = adata.var.drop(columns=[col])

    adata.write_h5ad(args.out)
    print(f"Wrote {args.out}: {adata.n_obs} cells x {adata.n_vars} genes")
    print(adata.obs["cell_type"].value_counts().head(15))


if __name__ == "__main__":
    main()
