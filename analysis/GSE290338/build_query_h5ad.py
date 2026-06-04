#!/usr/bin/env python3
"""Build spot x gene AnnData from GSE290338 Space Ranger (Visium HD 8um) outputs."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc


def extract_sample(raw_dir: Path, gsm_prefix: str, out_subdir: str) -> Path:
    tar_gz = raw_dir / f"{gsm_prefix}_square_008um.tar.gz"
    dest = raw_dir / "extracted" / out_subdir
    if not tar_gz.exists():
        raise FileNotFoundError(tar_gz)
    if not dest.exists():
        dest.mkdir(parents=True)
        with tarfile.open(tar_gz, "r:gz") as tf:
            tf.extractall(dest)
    children = [p for p in dest.iterdir() if p.is_dir()]
    if len(children) == 1:
        return children[0]
    return dest


def load_visium(counts_dir: Path, sample_id: str) -> ad.AnnData:
    adata = sc.read_visium(counts_dir, library_id=sample_id, load_images=False)
    adata.var_names_make_unique()
    adata.obs["sample"] = sample_id
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    return adata


def filter_and_subsample(
    adata: ad.AnnData,
    min_genes: int,
    min_counts: int,
    max_cells: int | None,
    seed: int,
) -> ad.AnnData:
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    if max_cells is not None and adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=seed)
    return adata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "raw",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "GSE290338_query.h5ad",
    )
    p.add_argument(
        "--samples",
        default="7d",
        help="Comma-separated: 24h, 7d, or both",
    )
    p.add_argument("--min-genes", type=int, default=10)
    p.add_argument("--min-counts", type=int, default=20)
    p.add_argument(
        "--max-cells-per-sample",
        type=int,
        default=40000,
        help="Subsample each sample to this many spots for downstream UMAP/MALT",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    sample_map = {
        "24h": ("GSM8810907_24h", "24h_post_irr"),
        "7d": ("GSM8810908_7d", "7d_post_irr"),
    }
    wanted = [s.strip() for s in args.samples.split(",")]

    parts: list[ad.AnnData] = []
    for key in wanted:
        if key not in sample_map:
            raise ValueError(f"Unknown sample {key!r}; choose from {list(sample_map)}")
        gsm, sid = sample_map[key]
        counts_dir = extract_sample(args.raw_dir, gsm, sid)
        print(f"Loading {counts_dir} ({key})")
        adata = load_visium(counts_dir, sid)
        print(f"  raw {adata.n_obs} spots")
        adata = filter_and_subsample(
            adata,
            args.min_genes,
            args.min_counts,
            args.max_cells_per_sample,
            args.seed,
        )
        print(f"  kept {adata.n_obs} spots after QC/subsample")
        parts.append(adata)

    if len(parts) == 1:
        out = parts[0]
    else:
        out = ad.concat(parts, label="sample", keys=[a.obs["sample"][0] for a in parts])
        out.obs_names_make_unique()

    if "counts" not in out.layers:
        out.layers["counts"] = out.X.copy()
    out.write_h5ad(args.out)
    print(f"Wrote {args.out}: {out.n_obs} spots x {out.n_vars} genes")


if __name__ == "__main__":
    main()
