#!/usr/bin/env python3
"""Cell-type annotation for a SPAC-seq Visium HD subQ slice.

Writes processed/<slice>_cells_annotated.h5ad with graphclust, cell_type,
and spatial coordinates (centroids from segmentation polygons).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import scanpy as sc


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent.parent

MARKERS = {
    "tumor": ["Epcam", "Krt8", "Krt18", "Msln", "Pecam1"],
    "immune": ["Ptprc", "Cd3d", "Cd3e", "Cd8a", "Cd4", "Nkg7"],
    "myeloid": ["Adgre1", "Cd68", "Itgam", "Ly6c2"],
    "fibroblast": ["Col1a1", "Col1a2", "Dcn", "Pdgfra"],
}


def cellid_to_barcode(cell_id: int) -> str:
    return f"cellid_{int(cell_id):09d}-1"


def load_cell_adata(data_dir: Path) -> sc.AnnData:
    h5 = data_dir / "segmentation/extracted/segmentation/filtered_feature_cell_matrix.h5"
    geo = data_dir / "segmentation/extracted/segmentation/graphclust_annotated_cell_segmentations.geojson"
    adata = sc.read_10x_h5(h5)
    adata.var_names_make_unique()

    gdf = gpd.read_file(geo)
    gdf["barcode"] = gdf["cell_id"].map(cellid_to_barcode)
    gdf["graphclust"] = gdf["classification"].apply(
        lambda x: x["name"] if isinstance(x, dict) else str(x)
    )
    gdf_indexed = gdf.set_index("barcode")
    adata = adata[adata.obs_names.isin(gdf_indexed.index)].copy()
    adata.obs["graphclust"] = gdf_indexed.loc[adata.obs_names, "graphclust"].astype("category").values
    centroid_map = {
        barcode: (geom.centroid.x, geom.centroid.y)
        for barcode, geom in zip(gdf["barcode"], gdf.geometry)
    }
    adata.obsm["spatial"] = np.array([centroid_map[b] for b in adata.obs_names])
    return adata


def score_cell_types(adata: sc.AnnData) -> sc.AnnData:
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    for ct, genes in MARKERS.items():
        present = [g for g in genes if g in tmp.var_names]
        if not present:
            adata.obs[f"score_{ct}"] = 0.0
            continue
        sc.tl.score_genes(tmp, present, score_name=f"score_{ct}")
        adata.obs[f"score_{ct}"] = tmp.obs[f"score_{ct}"].values
    scores = adata.obs[[f"score_{k}" for k in MARKERS]]
    adata.obs["cell_type"] = scores.idxmax(axis=1).str.replace("score_", "")
    adata.obs.loc[adata.obs["score_immune"] < 0.1, "cell_type"] = np.where(
        adata.obs.loc[adata.obs["score_immune"] < 0.1, "score_tumor"]
        > adata.obs.loc[adata.obs["score_immune"] < 0.1, "score_fibroblast"],
        "tumor",
        "fibroblast",
    )
    adata.obs.loc[adata.obs["score_immune"] >= 0.25, "cell_type"] = "immune"
    adata.obs.loc[adata.obs["score_myeloid"] >= 0.3, "cell_type"] = "myeloid"
    return adata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--slice-name", type=str, required=True)
    p.add_argument(
        "--out-h5ad",
        type=Path,
        default=None,
        help="Default: <data-dir>/processed/<slice>_cells_annotated.h5ad",
    )
    args = p.parse_args()

    out = args.out_h5ad or (args.data_dir / "processed" / f"{args.slice_name}_cells_annotated.h5ad")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data_dir} ...")
    adata = load_cell_adata(args.data_dir)
    adata = score_cell_types(adata)
    adata.uns["slice_name"] = args.slice_name
    adata.write_h5ad(out)
    print(f"Wrote {out} ({adata.n_obs:,} cells, {adata.n_vars:,} genes)")
    print(adata.obs["cell_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
