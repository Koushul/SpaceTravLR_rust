#!/usr/bin/env python3
"""Run BANKSY spatial clustering on an AnnData .h5ad.

Invoked by `spacetravlr banksy --h5ad PATH` via isolated `uv run` (see src/banksy_cluster.rs).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

NBR_WEIGHT_DECAY_CHOICES = ("scaled_gaussian", "uniform", "reciprocal", "ranked")


def ensure_spatial_coordinates(
    adata: ad.AnnData,
    coord_key: str = "spatial",
    x_col: str | None = None,
    y_col: str | None = None,
) -> tuple[str, str, str]:
    if coord_key in adata.obsm:
        return x_col or "x", y_col or "y", coord_key

    candidates = [
        ("array_col", "array_row"),
        ("x", "y"),
        ("X", "Y"),
        ("col", "row"),
    ]
    if x_col and y_col:
        candidates.insert(0, (x_col, y_col))

    for x_name, y_name in candidates:
        if x_name in adata.obs.columns and y_name in adata.obs.columns:
            adata.obsm[coord_key] = np.column_stack(
                [
                    adata.obs[x_name].to_numpy(dtype=np.float64),
                    adata.obs[y_name].to_numpy(dtype=np.float64),
                ]
            )
            return x_name, y_name, coord_key

    raise ValueError(
        "Could not find spatial coordinates. Provide --coord-key for an existing obsm entry "
        f"or --x-col/--y-col for obs columns. Available obsm keys: {list(adata.obsm.keys())}"
    )


def preprocess_for_banksy(adata: ad.AnnData) -> None:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)


def run_banksy(
    adata: ad.AnnData,
    *,
    lambda_: float,
    num_neighbours: int,
    nbr_weight_decay: str,
    max_m: int,
    resolution: float,
    num_nn: int,
    pca_dims: int,
    partition_seed: int,
    num_iterations: int,
    coord_key: str,
    x_col: str | None,
    y_col: str | None,
    cluster_key: str,
    preprocess: bool,
) -> ad.AnnData:
    from banksy.initialize_banksy import initialize_banksy
    from banksy.run_banksy import generate_banksy_matrix, pca_umap, run_Leiden_partition

    if nbr_weight_decay not in NBR_WEIGHT_DECAY_CHOICES:
        raise ValueError(
            f"Unsupported nbr_weight_decay={nbr_weight_decay!r}. "
            f"Choose one of {NBR_WEIGHT_DECAY_CHOICES}."
        )

    adata = adata.copy()
    x_name, y_name, coord_key = ensure_spatial_coordinates(
        adata, coord_key=coord_key, x_col=x_col, y_col=y_col
    )
    if preprocess:
        preprocess_for_banksy(adata)

    coord_keys = (x_name, y_name, coord_key)
    banksy_dict = initialize_banksy(
        adata,
        coord_keys=coord_keys,
        num_neighbours=num_neighbours,
        nbr_weight_decay=nbr_weight_decay,
        max_m=max_m,
        plt_edge_hist=False,
        plt_nbr_weights=False,
        plt_agf_angles=False,
        plt_theta=False,
    )
    banksy_dict, _ = generate_banksy_matrix(
        adata, banksy_dict, [lambda_], max_m=max_m, verbose=False
    )
    pca_umap(banksy_dict, pca_dims=[pca_dims], plt_remaining_var=False, add_umap=False)
    results_df, _ = run_Leiden_partition(
        banksy_dict=banksy_dict,
        resolutions=[resolution],
        num_nn=num_nn,
        num_iterations=num_iterations,
        partition_seed=partition_seed,
        match_labels=False,
    )
    labels = np.asarray(results_df.iloc[0]["labels"].dense, dtype=np.int32)
    adata.obs[cluster_key] = labels.astype(str)
    adata.obsm[f"X_banksy_pca_{pca_dims}"] = np.asarray(
        banksy_dict[nbr_weight_decay][lambda_]["adata"].obsm[f"reduced_pc_{pca_dims}"],
        dtype=np.float64,
    )
    return adata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run BANKSY spatial clustering on an AnnData .h5ad")
    parser.add_argument("--h5ad", required=True, help="Input .h5ad path")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .h5ad path (default: <input_stem>_banksy.h5ad)",
    )
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.2)
    parser.add_argument("--num-neighbours", type=int, default=15)
    parser.add_argument(
        "--nbr-weight-decay",
        choices=NBR_WEIGHT_DECAY_CHOICES,
        default="scaled_gaussian",
    )
    parser.add_argument("--max-m", type=int, default=1)
    parser.add_argument("--resolution", type=float, default=0.6)
    parser.add_argument("--num-nn", type=int, default=50)
    parser.add_argument("--pca-dims", type=int, default=20)
    parser.add_argument("--partition-seed", type=int, default=1234)
    parser.add_argument("--num-iterations", type=int, default=-1)
    parser.add_argument("--coord-key", default="spatial")
    parser.add_argument("--x-col", default=None)
    parser.add_argument("--y-col", default=None)
    parser.add_argument("--cluster-key", default="banksy_cluster")
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    h5ad_path = Path(args.h5ad)
    output_path = (
        Path(args.output)
        if args.output
        else h5ad_path.with_name(f"{h5ad_path.stem}_banksy.h5ad")
    )

    logging.info("Loading %s", h5ad_path)
    adata = ad.read_h5ad(h5ad_path)
    result = run_banksy(
        adata,
        lambda_=args.lambda_,
        num_neighbours=args.num_neighbours,
        nbr_weight_decay=args.nbr_weight_decay,
        max_m=args.max_m,
        resolution=args.resolution,
        num_nn=args.num_nn,
        pca_dims=args.pca_dims,
        partition_seed=args.partition_seed,
        num_iterations=args.num_iterations,
        coord_key=args.coord_key,
        x_col=args.x_col,
        y_col=args.y_col,
        cluster_key=args.cluster_key,
        preprocess=not args.no_preprocess,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.obs.index = result.obs.index.astype(str)
    result.var.index = result.var.index.astype(str)
    ad.settings.allow_write_nullable_strings = True
    result.write_h5ad(output_path)
    n_clusters = result.obs[args.cluster_key].nunique()
    logging.info(
        "BANKSY finished: %d cells, %d clusters (lambda=%s, resolution=%s)",
        result.n_obs,
        n_clusters,
        args.lambda_,
        args.resolution,
    )
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
