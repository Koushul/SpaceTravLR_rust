#!/usr/bin/env python3
"""Build Bonsai input from individually retained signed β features."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import scanpy as sc

BETA_OUT = Path("/ix1/ylee/kor11/djishnu_kor11/tonsil_ablation/snrna_human_tonsil_2026-08-10")
H5AD = Path("/ix1/ylee/kor11/djishnu_kor11/tonsil_ablation/snrna_human_tonsil_processed.h5ad")
FEATURES = Path(__file__).parent / "public/tonsil_niche_benchmark/beta_features_kept.csv"
OUT = Path(os.environ.get("BONSAI_SIGNED_OUT", "/tmp/tonsil_bonsai_signed"))
SIGMA_FRACTION = float(os.environ.get("BONSAI_SIGMA_FRACTION", "0.1"))
SIGMA_FLOOR = 1e-6


def main() -> None:
    data_dir = OUT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(H5AD)
    cell_ids = list(adata.obs_names.astype(str))
    cell_index = pd.Index(cell_ids)
    selected = pd.read_csv(FEATURES).sort_values("rank").drop_duplicates("col")
    if not selected["feature"].str.startswith("beta_").all():
        raise ValueError("Selected feature table contains non-beta columns")

    matrix = np.zeros((len(cell_ids), len(selected)), dtype=np.float32)
    for gene, rows in selected.groupby("gene", sort=False):
        path = BETA_OUT / f"{gene}_betadata.feather"
        with pa.memory_map(str(path), "r") as source:
            table = ipc.open_file(source).read_all().select(["CellID", *rows["feature"]])
        ids = pd.Index(table["CellID"].to_pylist()).astype(str)
        locations = cell_index.get_indexer(ids)
        if (locations < 0).any() or len(ids) != len(cell_ids):
            raise ValueError(f"Cell IDs do not align for {gene}")
        for row in rows.itertuples():
            column = selected.index.get_loc(row.Index)
            matrix[locations, column] = table[row.feature].to_numpy()

    scales = np.maximum(selected["mad"].to_numpy(dtype=np.float64), SIGMA_FLOOR)
    feat = matrix.T.astype(np.float64)
    std = np.broadcast_to(
        np.maximum(SIGMA_FRACTION * scales, SIGMA_FLOOR)[:, None],
        feat.shape,
    )
    print("Writing Bonsai TSV input...")
    np.savetxt(data_dir / "features.txt", feat, delimiter="\t", fmt="%.6g")
    np.savetxt(data_dir / "standard_deviations.txt", std, delimiter="\t", fmt="%.6g")
    (data_dir / "cellID.txt").write_text("\n".join(cell_ids) + "\n")
    (data_dir / "geneID.txt").write_text("\n".join(selected["col"]) + "\n")
    np.save(OUT / "cell_feature_beta.npy", matrix)
    (OUT / "feature_names.json").write_text(json.dumps(selected["col"].tolist()))
    (OUT / "cell_ids.json").write_text(json.dumps(cell_ids))

    ann = adata.obs[["cell_type"]].copy()
    if "cell_type_2" in adata.obs.columns:
        ann["cell_type_2"] = adata.obs["cell_type_2"].astype(str)
    ann.insert(0, "cellID", cell_ids)
    ann.to_csv(data_dir / "cell_annotation.csv", index=False)

    meta = {
        "n_cells": len(cell_ids),
        "n_features": len(selected),
        "n_target_genes": int(selected["gene"].nunique()),
        "sigma_fraction_of_feature_mad": SIGMA_FRACTION,
        "nonzero_fraction": float(np.count_nonzero(matrix) / matrix.size),
        "positive_fraction_nonzero": float((matrix > 0).sum() / max(np.count_nonzero(matrix), 1)),
        "feature_source": str(FEATURES),
        "representation": "individual signed target::beta_modulator coefficients",
    }
    (OUT / "input_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
