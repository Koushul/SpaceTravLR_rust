#!/usr/bin/env python3
"""UMAP figures for Leiden and MALT labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", type=Path, required=True)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=Path("figures"))
    args = p.parse_args()

    adata = sc.read_h5ad(args.h5ad)
    labels = pd.read_csv(args.labels_csv, index_col=0)
    malt = labels["malt_label"].reindex(adata.obs_names)
    adata.obs["malt_label"] = malt.astype("category")

    args.outdir.mkdir(parents=True, exist_ok=True)
    for col in ("leiden", "malt_label", "sample"):
        if col not in adata.obs:
            continue
        sc.pl.umap(adata, color=col, show=False, frameon=False, size=3)
        plt.tight_layout()
        out = args.outdir / f"umap_{col}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")

    if "malt_label" in adata.obs:
        vc = adata.obs["malt_label"].value_counts()
        immune = vc.reindex(
            ["Macrophages", "Monocytes", "T cells", "NKT", "NK cells", "ILC"],
        ).fillna(0).astype(int)
        print("\nImmune MALT counts:")
        print(immune.to_string())

    for gene in ("Cd8a", "Cd8b1", "Lyz2", "Spp1", "Cxcl10"):
        if gene in adata.var_names:
            sc.pl.umap(adata, color=gene, show=False, frameon=False, size=3, cmap="viridis")
            plt.tight_layout()
            out = args.outdir / f"umap_{gene}.png"
            plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved {out}")


if __name__ == "__main__":
    main()
