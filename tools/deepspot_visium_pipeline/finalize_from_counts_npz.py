#!/usr/bin/env python3
"""Rebuild correlation tables + .h5ad from `eval_paired_zen38.py` cached `.counts.npz` (skip ViT inference)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--counts-npz", type=Path, required=True)
    p.add_argument("--measured-h5ad", type=Path, required=True)
    p.add_argument("--weights-dir", type=Path, required=True)
    p.add_argument("--out-h5ad", type=Path, required=True)
    p.add_argument("--out-corr-csv", type=Path, default=None)
    p.add_argument("--out-markers-csv", type=Path, default=None)
    args = p.parse_args()

    z = np.load(args.counts_npz, allow_pickle=True)
    counts = z["counts"]
    intersect_genes = z["intersect_genes"].tolist()
    obs_names = z["obs_names"].astype(str)

    raw = sc.read_h5ad(args.measured_h5ad)
    measured = raw[:, intersect_genes].copy()
    sc.pp.normalize_total(measured, target_sum=1e4)
    sc.pp.log1p(measured)
    measured = measured[obs_names].copy()

    gene_csv = args.weights_dir / "info_highly_variable_genes.csv"
    genes_manifest = pd.read_csv(gene_csv)
    pred_mask = genes_manifest["isPredicted"].values.astype(bool)
    pred_var_names = genes_manifest.loc[pred_mask, "gene_name"].astype(str).values
    pred_col = {g: i for i, g in enumerate(pred_var_names)}
    pred_mat = counts[:, [pred_col[g] for g in intersect_genes]]

    meas_mat = measured.X
    if hasattr(meas_mat, "toarray"):
        meas_mat = meas_mat.toarray()
    meas_mat = np.asarray(meas_mat, dtype=np.float64)

    rows = []
    for i, g in enumerate(intersect_genes):
        x, y = meas_mat[:, i], pred_mat[:, i].astype(np.float64)
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            r = np.nan
        else:
            r, _ = pearsonr(x, y)
        rows.append({"gene": g, "pearson_r": r, "n_spots": meas_mat.shape[0]})
    corr_df = pd.DataFrame(rows).sort_values("pearson_r", ascending=False, na_position="last")

    markers = [
        "MUC2",
        "ITLN1",
        "CLCA1",
        "FCGBP",
        "EPCAM",
        "KRT20",
        "CDX2",
        "CD3D",
        "CD68",
        "COL1A1",
        "VIM",
        "MS4A1",
    ]
    mk = [g for g in markers if g in set(intersect_genes)]
    marker_df = corr_df[corr_df["gene"].isin(mk)].copy()

    out = ad.AnnData(
        meas_mat,
        obs=measured.obs.copy(),
        var=pd.DataFrame(index=intersect_genes),
    )
    out.layers["measured_log1p"] = meas_mat.copy()
    out.layers["imputed_count"] = pred_mat.astype(np.float32)
    out.obsm["spatial"] = np.asarray(measured.obsm["spatial"])
    out.uns["marker_pearson_json"] = marker_df.to_json(orient="records")
    out.uns["pearson_median_all_genes"] = float(np.nanmedian(corr_df["pearson_r"].values))
    top500 = corr_df.head(500)["pearson_r"].dropna().values
    out.uns["pearson_mean_top500"] = float(np.mean(top500)) if len(top500) else float("nan")
    out.uns["source_counts_npz"] = str(args.counts_npz.resolve())

    args.out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out_h5ad)
    if args.out_corr_csv:
        corr_df.to_csv(args.out_corr_csv, index=False)
    if args.out_markers_csv:
        marker_df.to_csv(args.out_markers_csv, index=False)
    print("Wrote", args.out_h5ad)
    print("Median r:", out.uns["pearson_median_all_genes"])
    print(marker_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
