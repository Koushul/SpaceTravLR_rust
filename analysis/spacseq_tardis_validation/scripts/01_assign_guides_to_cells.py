#!/usr/bin/env python3
"""Assign sgRNAs (and thus target genes) to segmented cells in SPAC-seq subQ-1.

Inputs
------
- bin-level guide matrix: subQ-1/perturbation/filtered_guide_bc_matrix.h5
  (1520 sgRNAs x 632k 8 um bins, sparse counts)
- bin-level tissue positions: subQ-1/raw/extracted/tissue_positions.parquet
- cell segmentations (polygons + cellid -> barcode): subQ-1/segmentation/extracted/segmentation/graphclust_annotated_cell_segmentations.geojson

Output
------
- analysis/spacseq_tardis_validation/data/cell_guide_assignments.parquet
  Columns: cell_barcode, top_guide, top_count, total_count, n_guides_detected,
           sgrna_target_gene, is_unambiguous, is_ntc
- analysis/spacseq_tardis_validation/data/sgrna_metadata.parquet
  Columns: sgRNA, target_gene, is_ntc
- analysis/spacseq_tardis_validation/data/guide_summary.json

Approach
--------
For each bin with at least one guide UMI, find the segmented cell whose polygon
contains the bin's pixel coordinate (point-in-polygon via geopandas.sjoin). Sum
guide counts within each cell. The "top guide" for a cell is the sgRNA with the
largest UMI count (with optional plurality > second-best). Multi-guide cells are
flagged as ambiguous.

sgRNA naming convention from SPAC: 'sgGene_1', 'sgGene_2', ..., 'sgNon_targeting_*'
We strip the trailing '_<n>' to derive the target gene symbol, and treat any
guide whose symbol contains 'Non' or 'NTC' as a non-targeting control.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from shapely.geometry import Point


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent.parent
DEFAULT_DATA = REPO / "analysis" / "mc38_visiumhd" / "subQ-1"


def cellid_to_barcode(cell_id: int) -> str:
    return f"cellid_{int(cell_id):09d}-1"


def parse_target(sgrna: str) -> tuple[str, bool]:
    """Strip trailing '_<n>' (sgRNA replicate index) -> target symbol.

    Returns (target_gene, is_non_targeting).
    """
    name = sgrna
    if name.lower().startswith("sg"):
        name = name[2:]
    base = re.sub(r"_\d+$", "", name)
    is_ntc = any(tok in base.lower() for tok in ("non_targeting", "non-targeting", "nontargeting", "ntc"))
    return base, is_ntc


def load_bin_guide_matrix(perturb_h5: Path) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Load bin-level guide matrix in cell-major (csr) form: rows = bins, cols = sgRNAs."""
    with h5py.File(perturb_h5, "r") as f:
        # Stored as csr with bins as rows (obs).
        data = f["X/data"][:]
        indices = f["X/indices"][:]
        indptr = f["X/indptr"][:]
        bins = np.asarray([b.decode() if isinstance(b, bytes) else b for b in f["obs/_index"][:]])
        guides = np.asarray([b.decode() if isinstance(b, bytes) else b for b in f["var/_index"][:]])
        n_bins = len(bins)
        n_guides = len(guides)
    mat = sparse.csr_matrix((data, indices, indptr), shape=(n_bins, n_guides))
    return mat, bins, guides


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out-dir", type=Path, default=ROOT / "data")
    p.add_argument("--min-guide-umi", type=int, default=1,
                   help="Per-cell threshold for the top guide UMI count.")
    p.add_argument("--unambig-ratio", type=float, default=0.7,
                   help="top_guide_umi / total_guide_umi must exceed this to call a cell unambiguous.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- load tissue positions for bin coordinates ---
    pos = pd.read_parquet(args.data_dir / "raw/extracted/tissue_positions.parquet")
    pos = pos.set_index("barcode")
    print(f"[+{time.time()-t0:.1f}s] tissue_positions: {len(pos):,} bins, in_tissue={int(pos.in_tissue.sum()):,}")

    # --- load cell polygons (mapping cellid -> polygon) ---
    geo_path = args.data_dir / "segmentation/extracted/segmentation/graphclust_annotated_cell_segmentations.geojson"
    cells = gpd.read_file(geo_path)
    cells["barcode"] = cells["cell_id"].map(cellid_to_barcode)
    cells["graphclust"] = cells["classification"].apply(
        lambda x: x["name"] if isinstance(x, dict) else str(x)
    )
    # Cell polygons are stored in a coord frame whose Y-axis (rows) aligns with
    # tissue_positions.pxl_row_in_fullres but whose X-axis (cols) is offset
    # from pxl_col_in_fullres. Estimate the offset empirically from the bounding
    # boxes so polygons line up with bin pixel coordinates.
    poly_x_min = cells.geometry.bounds["minx"].min()
    bin_x_min = pos.loc[pos.in_tissue == 1, "pxl_col_in_fullres"].min()
    x_offset = float(bin_x_min - poly_x_min)
    print(f"[+{time.time()-t0:.1f}s] segmentation: {len(cells):,} cells | x_offset={x_offset:.1f}")
    cells = cells.set_geometry(cells.geometry.translate(xoff=x_offset, yoff=0.0))
    cells = cells.set_crs(None, allow_override=True)

    # --- load bin-level guide matrix ---
    mat, bin_barcodes, guide_names = load_bin_guide_matrix(args.data_dir / "perturbation/filtered_guide_bc_matrix.h5")
    bins_with_guides = np.unique(mat.nonzero()[0])
    print(f"[+{time.time()-t0:.1f}s] guide matrix: {mat.shape}, bins-with-any-guide: {len(bins_with_guides):,}")

    # restrict to bins with at least one guide UMI for the spatial join (much smaller)
    bin_pos = pos.loc[bin_barcodes[bins_with_guides]].copy().reset_index(drop=True)
    bin_pos["bin_row"] = bins_with_guides
    bin_pos["geometry"] = [
        Point(c, r) for c, r in zip(bin_pos.pxl_col_in_fullres, bin_pos.pxl_row_in_fullres)
    ]
    bin_gdf = gpd.GeoDataFrame(bin_pos[["bin_row", "geometry"]], geometry="geometry", crs=None)
    print(f"[+{time.time()-t0:.1f}s] bin point GeoDataFrame: {len(bin_gdf):,} rows")

    # point-in-polygon via spatial join
    cells_sub = cells[["barcode", "graphclust", "geometry"]]
    joined = gpd.sjoin(bin_gdf, cells_sub, how="inner", predicate="within")
    print(f"[+{time.time()-t0:.1f}s] sjoin: {len(joined):,} bin-in-cell hits")

    if len(joined) == 0:
        raise SystemExit(
            "0 sjoin hits; coordinate alignment failed. Inspect bin pxl ranges vs polygon bounds."
        )

    # explode bin guide counts -> long form
    bin_rows = joined["bin_row"].to_numpy()
    cell_codes, cell_uniques = pd.factorize(joined["barcode"].to_numpy(), sort=False)

    sub = mat[bin_rows]  # rows aligned to joined rows
    coo = sub.tocoo()
    if len(coo.data) == 0:
        raise SystemExit("No guide UMIs in joined bins; check spatial join.")
    long = pd.DataFrame({
        "cell_idx": cell_codes[coo.row],
        "guide_idx": coo.col,
        "umi": coo.data,
    })
    cell_guide = long.groupby(["cell_idx", "guide_idx"], as_index=False)["umi"].sum()
    print(f"[+{time.time()-t0:.1f}s] cell x guide nonzero entries: {len(cell_guide):,}")

    # --- per-cell summaries ---
    total_per_cell = cell_guide.groupby("cell_idx")["umi"].sum()
    top_rows = cell_guide.sort_values(["cell_idx", "umi"], ascending=[True, False]).drop_duplicates(
        "cell_idx", keep="first"
    )
    # second best for ambiguity check
    second = (
        cell_guide.sort_values(["cell_idx", "umi"], ascending=[True, False])
        .groupby("cell_idx")
        .nth(1)
        .reset_index()
    )
    n_guides = cell_guide.groupby("cell_idx").size()
    top_rows = top_rows.merge(second.rename(columns={"umi": "second_umi"})[["cell_idx", "second_umi"]],
                              on="cell_idx", how="left")
    top_rows["second_umi"] = top_rows["second_umi"].fillna(0).astype(int)
    top_rows["total_umi"] = total_per_cell.loc[top_rows["cell_idx"]].values
    top_rows["n_guides_detected"] = n_guides.loc[top_rows["cell_idx"]].values

    top_rows["cell_barcode"] = cell_uniques[top_rows["cell_idx"].values]
    top_rows["top_guide"] = guide_names[top_rows["guide_idx"].values]
    top_rows["target_gene"], top_rows["is_ntc"] = zip(*[parse_target(g) for g in top_rows["top_guide"]])
    top_rows["plurality"] = top_rows["umi"] / top_rows["total_umi"].clip(lower=1)
    top_rows["is_unambiguous"] = (
        (top_rows["umi"] >= args.min_guide_umi)
        & (top_rows["plurality"] >= args.unambig_ratio)
    )

    out_cols = [
        "cell_barcode", "top_guide", "umi", "total_umi", "n_guides_detected",
        "plurality", "target_gene", "is_ntc", "is_unambiguous", "second_umi",
    ]
    table = top_rows[out_cols].rename(columns={"umi": "top_umi"})

    table.to_parquet(args.out_dir / "cell_guide_assignments.parquet", index=False)
    print(f"[+{time.time()-t0:.1f}s] wrote cell_guide_assignments.parquet ({len(table):,} rows)")

    sgrna_meta = pd.DataFrame({"sgRNA": guide_names})
    sgrna_meta["target_gene"], sgrna_meta["is_ntc"] = zip(*[parse_target(g) for g in sgrna_meta["sgRNA"]])
    sgrna_meta.to_parquet(args.out_dir / "sgrna_metadata.parquet", index=False)

    # summary
    unambig = table[table["is_unambiguous"]]
    by_gene = unambig.groupby("target_gene").size().sort_values(ascending=False)
    summary = {
        "n_bins": int(mat.shape[0]),
        "n_sgrnas": int(mat.shape[1]),
        "n_cells_polygons": int(len(cells)),
        "n_cells_with_any_guide": int(table.shape[0]),
        "n_cells_unambiguous": int(unambig.shape[0]),
        "n_cells_ntc": int(unambig[unambig["is_ntc"]].shape[0]),
        "n_unique_targets_unambig": int(unambig["target_gene"].nunique()),
        "top_15_targets": by_gene.head(15).to_dict(),
        "icam1_cells": int(unambig.query("target_gene == 'Icam1'").shape[0]),
        "cd44_cells": int(unambig.query("target_gene == 'Cd44'").shape[0]),
        "spp1_cells": int(unambig.query("target_gene == 'Spp1'").shape[0]),
    }
    (args.out_dir / "guide_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
