"""Run NicheCompass on the prepared tonsil benchmark.

NicheCompass (Birk et al., Nat. Genet. 2025) learns interpretable gene-program
embeddings of cells in their *spatial* graph context, then niches are obtained
by clustering the latent representation. To stay fully reproducible without
external API calls (no live OmniPath / NicheNet downloads) we build the
prior gene-program dictionary from the repository's own
``data/human_network.parquet``:

  * ligand-receptor pairs (`edge_type == "lr"`)  ->  GP per ligand
    {sources: [ligand], targets: receptor list}
  * NicheNet ligand-target inferences (`edge_type == "nichenet"`)
    -> GP per ligand
    {sources: [ligand], targets: top-N predicted targets by weight}

These programs follow the NicheCompass convention (sources = transmitting
neighbor genes, targets = receiving cell genes).

Pipeline:
  1. Load tonsil_prepared.h5ad (raw counts in `.layers['counts']`).
  2. Build spatial KNN graph (k=8) into ``adata.obsp['spatial_connectivities']``.
  3. Build LR + NicheNet GP dict, attach via ``add_gps_from_gp_dict_to_adata``.
  4. Train a small NicheCompass model (CPU).
  5. Cluster latent representation with Leiden, sweep resolutions and lock in
     the run whose cluster count is closest to the ground-truth count.
  6. Save labels CSV.
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")

from _common import EXP_DIR, RESULTS_DIR, Timer, save_labels

import torch

N_THREADS = int(os.environ.get("BENCH_TORCH_THREADS", str(min(32, os.cpu_count() or 8))))
torch.set_num_threads(N_THREADS)
torch.set_num_interop_threads(max(1, N_THREADS // 2))

NETWORK_PARQUET = EXP_DIR.parents[1] / "data" / "human_network.parquet"

K_SPATIAL_NBRS = 8
NICHENET_TOP_TARGETS = 50
N_HVG = 2000
N_HIDDEN = 64
N_ADDON_GP = 16
N_EPOCHS = 30
N_EPOCHS_ALL_GPS = 5
EDGE_BATCH_SIZE = 1024
NODE_BATCH_SIZE = 2048
LATENT_KEY = "nichecompass_latent"
ACTIVE_GP_THRESH_RATIO = 0.05
SEED = 0


def build_spatial_graph(adata: ad.AnnData, k: int = K_SPATIAL_NBRS) -> None:
    import squidpy as sq

    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=k, delaunay=False)


def build_gp_dict_from_parquet(parquet_path: Path,
                                nichenet_top: int = NICHENET_TOP_TARGETS) -> dict:
    df = pd.read_parquet(parquet_path)
    gp_dict: dict = {}

    lr = df[df["edge_type"] == "lr"].copy()
    lr_groups = lr.groupby("source")["target"].apply(lambda s: sorted(set(s)))
    for ligand, receptors in lr_groups.items():
        if not receptors:
            continue
        gp_dict[f"{ligand}_LR_GP"] = {
            "sources": [ligand],
            "sources_categories": ["ligand"],
            "targets": list(receptors),
            "targets_categories": ["receptor"] * len(receptors),
        }

    nn = df[df["edge_type"] == "nichenet"].copy()
    nn = nn.sort_values(["source", "weight"], ascending=[True, False])
    nn_top = (
        nn.groupby("source", group_keys=False)
        .head(nichenet_top)
        .groupby("source")["target"]
        .apply(lambda s: sorted(set(s)))
    )
    for ligand, targets in nn_top.items():
        if not targets:
            continue
        key = f"{ligand}_nichenet_GP"
        gp_dict[key] = {
            "sources": [ligand],
            "sources_categories": ["ligand"],
            "targets": list(targets),
            "targets_categories": ["target_gene"] * len(targets),
        }
    return gp_dict


def main() -> Path:
    from nichecompass.models import NicheCompass
    from nichecompass.utils import (
        add_gps_from_gp_dict_to_adata,
        filter_and_combine_gp_dict_gps_v2,
    )

    src = RESULTS_DIR / "tonsil_prepared.h5ad"
    adata = sc.read_h5ad(src)
    n_gt = int(adata.obs["microniche_gt"].nunique())
    print(f"loaded {adata.shape}; n_gt = {n_gt}", flush=True)

    if "counts" not in adata.layers:
        raise RuntimeError("Expected raw counts in adata.layers['counts']")

    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor="seurat", inplace=True)
    keep = adata.var["highly_variable"].to_numpy()
    adata = adata[:, keep].copy()
    print(f"subset to {adata.shape[1]} HVGs", flush=True)

    print("building spatial KNN graph...", flush=True)
    build_spatial_graph(adata, k=K_SPATIAL_NBRS)
    deg = np.asarray((adata.obsp["spatial_connectivities"] > 0).sum(axis=1)).ravel()
    print(f"spatial graph: mean deg = {deg.mean():.1f}, n_edges = {int((adata.obsp['spatial_connectivities'] > 0).sum())}")

    print(f"building gene program dict from {NETWORK_PARQUET.name}...")
    raw_gp_dict = build_gp_dict_from_parquet(NETWORK_PARQUET, nichenet_top=NICHENET_TOP_TARGETS)
    print(f"raw GP dict: {len(raw_gp_dict)} programs")

    combined = filter_and_combine_gp_dict_gps_v2(
        gp_dicts=[raw_gp_dict],
        overlap_thresh_target_genes=1.0,
        verbose=False,
    )
    print(f"after dedup: {len(combined)} programs")

    add_gps_from_gp_dict_to_adata(
        gp_dict=combined,
        adata=adata,
        genes_uppercase=True,
        min_genes_per_gp=2,
        min_source_genes_per_gp=1,
        min_target_genes_per_gp=1,
        max_genes_per_gp=200,
        filter_genes_not_in_masks=False,
        plot_gp_gene_count_distributions=False,
    )
    n_gp = len(adata.uns["nichecompass_gp_names"])
    print(f"GP names attached to adata: {n_gp}")

    print("instantiating NicheCompass model...")
    model = NicheCompass(
        adata,
        counts_key="counts",
        adj_key="spatial_connectivities",
        n_hidden_encoder=N_HIDDEN,
        n_addon_gp=N_ADDON_GP,
        active_gp_thresh_ratio=ACTIVE_GP_THRESH_RATIO,
        latent_key=LATENT_KEY,
        conv_layer_encoder="gcnconv",
        seed=SEED,
        use_cuda_if_available=False,
        log_variational=True,
    )
    print("training...")
    timer = Timer()
    model.train(
        n_epochs=N_EPOCHS,
        n_epochs_all_gps=N_EPOCHS_ALL_GPS,
        lr=1e-3,
        edge_batch_size=EDGE_BATCH_SIZE,
        node_batch_size=NODE_BATCH_SIZE,
        use_cuda_if_available=False,
        retrieve_recon_edge_probs=False,
        retrieve_agg_weights=False,
        verbose=False,
    )
    runtime = timer.stop()
    print(f"trained in {runtime:.1f}s")

    Z = adata.obsm[LATENT_KEY]
    print(f"latent shape: {Z.shape}")

    print("clustering latent with Leiden, sweeping resolutions...")
    sc.pp.neighbors(adata, use_rep=LATENT_KEY, n_neighbors=15, key_added="nc_neigh", random_state=SEED)
    candidates = []
    for res in [0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]:
        key = f"leiden_nc_{res:.2f}"
        sc.tl.leiden(adata, resolution=res, neighbors_key="nc_neigh", random_state=SEED, key_added=key)
        n_cl = int(adata.obs[key].nunique())
        candidates.append({"res": res, "key": key, "n_clusters": n_cl})
        print(f"  res={res:>4} -> {n_cl} clusters")
    summary = pd.DataFrame(candidates)
    summary["abs_diff"] = (summary["n_clusters"] - n_gt).abs()
    summary = summary.sort_values(["abs_diff", "n_clusters"]).reset_index(drop=True)
    print(summary)

    chosen = summary.iloc[0]
    labels = adata.obs[chosen["key"]].astype(int).to_numpy()
    print(f"chosen NicheCompass clustering: res={chosen['res']} -> {chosen['n_clusters']} clusters")

    out = save_labels(
        "nichecompass",
        adata.obs_names,
        labels,
        runtime_sec=runtime,
        extra={
            "method": "NicheCompass",
            "n_clusters": int(chosen["n_clusters"]),
            "leiden_resolution": float(chosen["res"]),
            "n_gp": int(n_gp),
            "n_addon_gp": N_ADDON_GP,
            "n_hidden_encoder": N_HIDDEN,
            "n_epochs": N_EPOCHS,
            "k_spatial_nbrs": K_SPATIAL_NBRS,
            "all_runs": [
                {"res": float(r["res"]), "n_clusters": int(r["n_clusters"])}
                for _, r in summary.iterrows()
            ],
        },
    )
    print(f"wrote {out}")

    latent_out = RESULTS_DIR / "nichecompass_latent.npy"
    np.save(latent_out, Z)
    print(f"wrote {latent_out}")
    return out


if __name__ == "__main__":
    main()
