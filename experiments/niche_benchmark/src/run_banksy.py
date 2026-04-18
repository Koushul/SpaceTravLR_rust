"""Run BANKSY on the prepared tonsil benchmark.

BANKSY (Singhal & Chou et al., Nat. Genet. 2024) augments each cell's
transcriptome with neighborhood-averaged expression and clusters via
Leiden in PCA space. This script:

  1. Loads the prepared tonsil AnnData.
  2. Initializes the BANKSY spatial graph (k=15 nearest neighbors,
     scaled-Gaussian decay).
  3. Runs BANKSY at lambda in {0.0, 0.2, 0.5, 0.8} and Leiden res in
     {0.5, 1.0, 1.5}, then picks the parameter set whose number of
     clusters is closest to the ground-truth count (19) -- the
     unsupervised choice we lock in for benchmarking.
  4. Writes a CSV of per-cell labels under
     ``experiments/niche_benchmark/labels/banksy.csv``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from _common import LABELS_DIR, RESULTS_DIR, Timer, save_labels

LAMBDAS = [0.2, 0.5, 0.8]
RESOLUTIONS = [0.5, 1.0, 1.5, 2.0]
N_NBRS = 15
PCA_DIMS = 20
SEED = 0


def _hvg(adata: ad.AnnData, n_top_genes: int = 2000) -> ad.AnnData:
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
    return adata[:, adata.var["highly_variable"]].copy()


def main() -> Path:
    from banksy.initialize_banksy import initialize_banksy
    from banksy.run_banksy import run_banksy_multiparam

    src = RESULTS_DIR / "tonsil_prepared.h5ad"
    adata = sc.read_h5ad(src)
    n_gt = int(adata.obs["microniche_gt"].nunique())
    print(f"loaded {adata.shape}; n_gt microniches = {n_gt}")

    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    adata.obs["x"] = coords[:, 0]
    adata.obs["y"] = coords[:, 1]

    adata_banksy = _hvg(adata, n_top_genes=2000)
    sc.pp.scale(adata_banksy, zero_center=True, max_value=10.0)
    print(f"banksy input: {adata_banksy.shape}")

    coord_keys = ("x", "y", "spatial")

    import matplotlib
    palette = list(matplotlib.colormaps["tab20"].colors) + list(matplotlib.colormaps["tab20b"].colors) + list(matplotlib.colormaps["tab20c"].colors)
    color_list = [matplotlib.colors.to_hex(c) for c in palette]

    timer = Timer()
    banksy_dict = initialize_banksy(
        adata_banksy,
        coord_keys=coord_keys,
        num_neighbours=N_NBRS,
        nbr_weight_decay="scaled_gaussian",
        max_m=1,
        plt_edge_hist=False,
        plt_nbr_weights=False,
        plt_agf_angles=False,
        plt_theta=False,
    )
    fig_dir = LABELS_DIR.parent / "figures" / "banksy"
    fig_dir.mkdir(parents=True, exist_ok=True)
    results_df = run_banksy_multiparam(
        adata_banksy,
        banksy_dict,
        lambda_list=LAMBDAS,
        resolutions=RESOLUTIONS,
        color_list=color_list,
        max_m=1,
        filepath=str(fig_dir),
        key=coord_keys,
        pca_dims=[PCA_DIMS],
        savefig=False,
        annotation_key="microniche_gt",
        cluster_algorithm="leiden",
        partition_seed=SEED,
        add_nonspatial=False,
    )
    runtime = timer.stop()
    print(f"banksy ran in {runtime:.1f}s")
    print(results_df.index.tolist())
    print(results_df[[c for c in results_df.columns if c not in {"adata", "labels"}]])

    rows = []
    for idx, row in results_df.iterrows():
        labels = row["labels"]
        if hasattr(labels, "dense"):
            arr = np.asarray(labels.dense)
        else:
            arr = np.asarray(labels)
        rows.append({
            "param_id": str(idx),
            "lambda_param": float(row["lambda_param"]),
            "resolution": float(row["resolution"]),
            "n_clusters": int(np.unique(arr).size),
            "labels": arr,
        })
    summary = pd.DataFrame(rows)
    summary["abs_diff"] = (summary["n_clusters"] - n_gt).abs()
    summary = summary.sort_values(["abs_diff", "n_clusters"]).reset_index(drop=True)
    print(summary[["param_id", "lambda_param", "resolution", "n_clusters", "abs_diff"]].head(10))

    chosen = summary.iloc[0]
    print(f"chosen banksy run: {chosen['param_id']} with {chosen['n_clusters']} clusters")
    out = save_labels(
        "banksy",
        adata_banksy.obs_names,
        chosen["labels"],
        runtime_sec=runtime,
        extra={
            "method": "BANKSY",
            "chosen_param": chosen["param_id"],
            "n_clusters": int(chosen["n_clusters"]),
            "lambdas": LAMBDAS,
            "resolutions": RESOLUTIONS,
            "k_nbrs": N_NBRS,
            "pca_dims": PCA_DIMS,
            "n_hvg": int(adata_banksy.shape[1]),
            "all_runs": [
                {"param_id": r["param_id"], "n_clusters": int(r["n_clusters"])}
                for _, r in summary.iterrows()
            ],
        },
    )
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
