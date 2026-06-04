#!/usr/bin/env python3
"""Save UMAP colored by Leiden from a preprocessed h5ad."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("h5ad", type=Path)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("umap_leiden.png"),
    )
    p.add_argument("--obs", default="leiden")
    args = p.parse_args()

    adata = sc.read_h5ad(args.h5ad)
    if "X_umap" not in adata.obsm:
        raise KeyError("obsm['X_umap'] missing; run spacetravlr --rust-process-h5ad first")
    if args.obs not in adata.obs:
        raise KeyError(f"obs[{args.obs!r}] missing")

    sc.pl.umap(adata, color=args.obs, show=False, frameon=False, size=2 if adata.n_obs > 20000 else 20)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
