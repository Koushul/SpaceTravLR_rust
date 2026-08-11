#!/usr/bin/env python3
"""Prepare public/niche_benchmark assets: Slide-seq V2 niche method comparison.

Runs NicheCompass, BANKSY, Leiden, COVET, and SpaceFlow on Squidpy's
slideseqv2 mouse hippocampus dataset (NicheCompass paper functional-brain
setting), then computes the paper's single-sample metric suite.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

# OmniPath defaults to an unwritable home cache in this environment.
os.environ.setdefault("HOME", "/tmp/fakehome")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
Path(os.environ["HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import umap
from anndata import AnnData
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

OUT = Path(__file__).resolve().parent / "public" / "niche_benchmark"
GP_ROOT = Path("/tmp/nichecompass_src/data")
NC_WORK = Path("/tmp/nichebench_work")
N_CELLS = 5000
SEED = 0
LEIDEN_RESOLUTION = 0.6
BANKSY_LAMBDA = 0.2
SPACEFLOW_EPOCHS = 250
NICHECOMPASS_EPOCHS = 120
METRICS = ["cas", "mlami", "clisis", "gcs", "cnmi", "nasw"]
METHODS = ["leiden", "banksy", "covet", "spaceflow", "nichecompass"]


def _tab20_palette(n: int) -> dict[str, str]:
    cmap = matplotlib.colormaps.get_cmap("tab20")
    return {
        str(i): "#%02x%02x%02x" % tuple(int(255 * c) for c in cmap(i % 20)[:3])
        for i in range(n)
    }


def _ensure_counts(adata: AnnData) -> AnnData:
    """Squidpy slideseqv2 ships log-normalised X; store it as counts for models."""
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    return adata


def _leiden_on_rep(adata: AnnData, use_rep: str, key: str, resolution: float) -> np.ndarray:
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=15, key_added=key)
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added=key,
        neighbors_key=key,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    return np.asarray(adata.obs[key].astype(str).to_numpy())


def run_leiden(adata: AnnData) -> tuple[np.ndarray, np.ndarray]:
    ad = adata.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=min(2000, ad.n_vars), subset=False)
    sc.pp.pca(ad, n_comps=30, use_highly_variable=True)
    labels = _leiden_on_rep(ad, "X_pca", "leiden_expr", LEIDEN_RESOLUTION)
    return labels, np.asarray(ad.obsm["X_pca"], dtype=np.float64)


def run_banksy(adata: AnnData) -> tuple[np.ndarray, np.ndarray]:
    from banksy.initialize_banksy import initialize_banksy
    from banksy.run_banksy import generate_banksy_matrix, pca_umap, run_Leiden_partition

    ad = adata.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.scale(ad, max_value=10)
    banksy_dict = initialize_banksy(
        ad,
        coord_keys=("x", "y", "spatial"),
        num_neighbours=15,
        nbr_weight_decay="scaled_gaussian",
        max_m=1,
        plt_edge_hist=False,
        plt_nbr_weights=False,
        plt_agf_angles=False,
        plt_theta=False,
    )
    banksy_dict, _ = generate_banksy_matrix(ad, banksy_dict, [BANKSY_LAMBDA], max_m=1)
    pca_umap(banksy_dict, pca_dims=[20], plt_remaining_var=False)
    results_df, _ = run_Leiden_partition(
        banksy_dict=banksy_dict,
        resolutions=[LEIDEN_RESOLUTION],
        num_nn=50,
        num_iterations=-1,
        partition_seed=1234,
        match_labels=False,
    )
    labels = np.asarray(results_df.iloc[0]["labels"].dense, dtype=np.int32)
    pca = np.asarray(
        banksy_dict["scaled_gaussian"][BANKSY_LAMBDA]["adata"].obsm["reduced_pc_20"],
        dtype=np.float64,
    )
    return labels.astype(str), pca


def run_covet(adata: AnnData) -> tuple[np.ndarray, np.ndarray]:
    from scenvi.utils import compute_covet
    from sklearn.decomposition import PCA

    ad = adata.copy()
    # slideseqv2 marks all genes HVG; force a compact gene set for COVET
    if "highly_variable" in ad.var.columns:
        del ad.var["highly_variable"]
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=64, subset=False)
    ad = ad[:, ad.var["highly_variable"]].copy()
    _, covet_sqrt, _ = compute_covet(
        ad,
        k=8,
        g=64,
        spatial_key="spatial",
        batch_size=512,
    )
    flat = np.asarray(covet_sqrt, dtype=np.float64).reshape(ad.n_obs, -1)
    latent = PCA(n_components=min(30, flat.shape[1] - 1), random_state=SEED).fit_transform(flat)
    ad.obsm["X_covet"] = latent
    labels = _leiden_on_rep(ad, "X_covet", "leiden_covet", LEIDEN_RESOLUTION)
    return labels, latent.astype(np.float64)


def run_spaceflow(adata: AnnData) -> tuple[np.ndarray, np.ndarray]:
    import networkx as nx
    from SpaceFlow import SpaceFlow

    # NetworkX >=3 removed to_scipy_sparse_matrix
    if not hasattr(nx, "to_scipy_sparse_matrix"):
        nx.to_scipy_sparse_matrix = lambda G, *a, **k: nx.to_scipy_sparse_array(G, *a, **k)

    ad = adata.copy()
    work = NC_WORK / "spaceflow"
    work.mkdir(parents=True, exist_ok=True)
    sf = SpaceFlow.SpaceFlow(adata=ad)
    sf.preprocessing_data(n_top_genes=min(2000, ad.n_vars), n_neighbors=10)
    emb_path = str(work / "embedding.tsv")
    dom_path = str(work / "domains.tsv")
    sf.train(
        embedding_save_filepath=emb_path,
        spatial_regularization_strength=0.1,
        z_dim=50,
        lr=0.001,
        epochs=SPACEFLOW_EPOCHS,
        max_patience=50,
        min_stop=80,
        random_seed=SEED,
        gpu=0,
    )
    sf.segmentation(
        domain_label_save_filepath=dom_path,
        n_neighbors=50,
        resolution=LEIDEN_RESOLUTION,
    )
    labels = np.asarray(sf.domains).astype(str)
    latent = np.asarray(sf.embedding, dtype=np.float64)
    return labels, latent


def run_nichecompass(adata: AnnData) -> tuple[np.ndarray, np.ndarray]:
    import pickle

    from nichecompass.models import NicheCompass
    from nichecompass.utils import (
        add_gps_from_gp_dict_to_adata,
        extract_gp_dict_from_mebocost_ms_interactions,
        extract_gp_dict_from_omnipath_lr_interactions,
        filter_and_combine_gp_dict_gps_v2,
    )

    ad = adata.copy()
    _ensure_counts(ad)
    species = "mouse"
    spatial_key = "spatial"
    counts_key = "counts"
    adj_key = "spatial_connectivities"
    gp_names_key = "nichecompass_gp_names"
    active_gp_names_key = "nichecompass_active_gp_names"
    gp_targets_mask_key = "nichecompass_gp_targets"
    gp_targets_categories_mask_key = "nichecompass_gp_targets_categories"
    gp_sources_mask_key = "nichecompass_gp_sources"
    gp_sources_categories_mask_key = "nichecompass_gp_sources_categories"
    latent_key = "nichecompass_latent"

    gp_pickle = NC_WORK / "combined_gp_dict.pkl"
    if gp_pickle.is_file():
        with open(gp_pickle, "rb") as f:
            combined = pickle.load(f)
        print(f"  Loaded cached GP dict ({len(combined)} programs)")
    else:
        ga = GP_ROOT / "gene_annotations" / "human_mouse_gene_orthologs.csv"
        gp_dir = GP_ROOT / "gene_programs"
        # Always fetch live OmniPath; on-disk CSV can contain NaNs that break parsing.
        omni = extract_gp_dict_from_omnipath_lr_interactions(
            species=species,
            load_from_disk=False,
            save_to_disk=False,
            lr_network_file_path=str(gp_dir / "omnipath_lr_network.csv"),
            gene_orthologs_mapping_file_path=str(ga),
            plot_gp_gene_count_distributions=False,
        )
        mebo = extract_gp_dict_from_mebocost_ms_interactions(
            dir_path=str(gp_dir / "metabolite_enzyme_sensor_gps"),
            species=species,
            plot_gp_gene_count_distributions=False,
        )
        combined = filter_and_combine_gp_dict_gps_v2([omni, mebo], verbose=False)
        NC_WORK.mkdir(parents=True, exist_ok=True)
        with open(gp_pickle, "wb") as f:
            pickle.dump(combined, f)

    sq.gr.spatial_neighbors(ad, coord_type="generic", spatial_key=spatial_key, n_neighs=4)
    ad.obsp[adj_key] = ad.obsp[adj_key].maximum(ad.obsp[adj_key].T)

    add_gps_from_gp_dict_to_adata(
        gp_dict=combined,
        adata=ad,
        gp_targets_mask_key=gp_targets_mask_key,
        gp_targets_categories_mask_key=gp_targets_categories_mask_key,
        gp_sources_mask_key=gp_sources_mask_key,
        gp_sources_categories_mask_key=gp_sources_categories_mask_key,
        gp_names_key=gp_names_key,
        min_genes_per_gp=2,
        min_source_genes_per_gp=1,
        min_target_genes_per_gp=1,
        max_genes_per_gp=None,
        max_source_genes_per_gp=None,
        max_target_genes_per_gp=None,
    )

    model = NicheCompass(
        ad,
        counts_key=counts_key,
        adj_key=adj_key,
        gp_names_key=gp_names_key,
        active_gp_names_key=active_gp_names_key,
        gp_targets_mask_key=gp_targets_mask_key,
        gp_targets_categories_mask_key=gp_targets_categories_mask_key,
        gp_sources_mask_key=gp_sources_mask_key,
        gp_sources_categories_mask_key=gp_sources_categories_mask_key,
        latent_key=latent_key,
        conv_layer_encoder="gcnconv",
        active_gp_thresh_ratio=0.01,
    )
    model.train(
        n_epochs=NICHECOMPASS_EPOCHS,
        n_epochs_all_gps=20,
        lr=0.001,
        lambda_edge_recon=500000.0,
        lambda_gene_expr_recon=300.0,
        lambda_l1_masked=0.0,
        edge_batch_size=1024,
        n_sampled_neighbors=4,
        use_cuda_if_available=True,
        verbose=False,
    )
    latent = np.asarray(model.adata.obsm[latent_key], dtype=np.float64)
    model.adata.obsm[latent_key] = latent
    sc.pp.neighbors(model.adata, use_rep=latent_key, n_neighbors=15, key_added=latent_key)
    sc.tl.leiden(
        model.adata,
        resolution=0.4,
        key_added="nichecompass_leiden",
        neighbors_key=latent_key,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    labels = np.asarray(model.adata.obs["nichecompass_leiden"].astype(str).to_numpy())
    return labels, latent


def _patch_nichecompass_knn() -> None:
    """NicheCompass 0.3.3 calls removed scanpy.neighbors._compute_connectivities_umap."""
    import scipy.sparse as sp
    from nichecompass.benchmarking import utils as nc_utils
    from scib_metrics.nearest_neighbors import pynndescent
    from scanpy.neighbors._connectivity import umap as umap_connectivities

    def compute_knn_graph_connectivities_and_distances(
        adata: AnnData,
        feature_key: str = "nichecompass_latent",
        knng_key: str = "nichecompass_latent_15knng",
        n_neighbors: int = 15,
        random_state: int = 0,
        n_jobs: int = 1,
    ) -> None:
        neigh_output = pynndescent(
            adata.obsm[feature_key],
            n_neighbors=n_neighbors,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        indices, distances = neigh_output.indices, neigh_output.distances
        row_idx = np.where(distances == 0)[0]
        col_idx = np.where(distances == 0)[1]
        new_row_idx = row_idx[np.where(row_idx != indices[row_idx, col_idx])[0]]
        new_col_idx = col_idx[np.where(row_idx != indices[row_idx, col_idx])[0]]
        distances[new_row_idx, new_col_idx] = distances[new_row_idx, new_col_idx] + np.nextafter(
            0, 1, dtype=np.float32
        )

        knn_indices = indices[:, :n_neighbors]
        knn_dists = distances[:, :n_neighbors]
        sp_conns = umap_connectivities(
            knn_indices,
            knn_dists,
            n_obs=adata.n_obs,
            n_neighbors=n_neighbors,
        )
        n_obs = adata.n_obs
        rows = np.repeat(np.arange(n_obs), n_neighbors)
        cols = knn_indices.ravel()
        vals = knn_dists.ravel()
        sp_distances = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_obs)).tocsr()
        adata.obsp[f"{knng_key}_connectivities"] = sp_conns
        adata.obsp[f"{knng_key}_distances"] = sp_distances
        adata.uns[f"{knng_key}_n_neighbors"] = n_neighbors

    nc_utils.compute_knn_graph_connectivities_and_distances = (
        compute_knn_graph_connectivities_and_distances
    )
    import nichecompass.benchmarking.metrics as nc_metrics
    import nichecompass.benchmarking.cas as nc_cas
    import nichecompass.benchmarking.clisis as nc_clisis
    import nichecompass.benchmarking.gcs as nc_gcs
    import nichecompass.benchmarking.mlami as nc_mlami
    import nichecompass.benchmarking.nasw as nc_nasw

    for mod in (nc_metrics, nc_cas, nc_clisis, nc_gcs, nc_mlami, nc_nasw):
        if hasattr(mod, "compute_knn_graph_connectivities_and_distances"):
            mod.compute_knn_graph_connectivities_and_distances = (
                compute_knn_graph_connectivities_and_distances
            )


def _compute_clisis(
    spatial: np.ndarray,
    latent: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 90,
    seed: int = 0,
) -> float:
    """CLISIS via scib-metrics NeighborsResults (NicheCompass 0.3.3 CSR path is broken)."""
    from scib_metrics.metrics._lisi import lisi_knn
    from scib_metrics.nearest_neighbors import NeighborsResults, pynndescent

    def _fix_zero(nn: NeighborsResults) -> NeighborsResults:
        dists = nn.distances.copy()
        idx = nn.indices
        row_idx, col_idx = np.where(dists == 0)
        keep = row_idx != idx[row_idx, col_idx]
        dists[row_idx[keep], col_idx[keep]] = dists[row_idx[keep], col_idx[keep]] + np.nextafter(
            0, 1, dtype=np.float32
        )
        return NeighborsResults(indices=idx, distances=dists)

    spatial_nn = _fix_zero(
        pynndescent(spatial, n_neighbors=n_neighbors, random_state=seed, n_jobs=1)
    )
    latent_nn = _fix_zero(
        pynndescent(latent, n_neighbors=n_neighbors, random_state=seed, n_jobs=1)
    )
    spatial_clisi = lisi_knn(X=spatial_nn, labels=labels)
    latent_clisi = lisi_knn(X=latent_nn, labels=labels)
    cell_log_rclisi = np.log2(latent_clisi / spatial_clisi)
    max_cell_log_rclisi = np.log2(len(np.unique(labels)) / 1)
    return float(1 - np.nanmedian(np.abs(cell_log_rclisi / max_cell_log_rclisi)))


def compute_metrics_for_method(
    spatial: np.ndarray,
    published: np.ndarray,
    labels: np.ndarray,
    latent: np.ndarray,
    method: str,
) -> dict:
    from nichecompass.benchmarking import compute_benchmarking_metrics

    _patch_nichecompass_knn()

    ad = AnnData(X=np.zeros((len(labels), 1)))
    ad.obs_names = [f"c{i}" for i in range(len(labels))]
    ad.obsm["spatial"] = np.asarray(spatial, dtype=np.float64)
    ad.obsm[f"{method}_latent"] = np.asarray(latent, dtype=np.float64)
    ad.obs["cluster"] = pd.Categorical(published.astype(str))
    ad.obs[f"{method}_niche"] = pd.Categorical(labels.astype(str))

    # Skip broken built-in CLISIS; recompute below with NeighborsResults.
    paper_metrics = [m for m in METRICS if m != "clisis"]
    paper = compute_benchmarking_metrics(
        adata=ad,
        metrics=paper_metrics,
        cell_type_key="cluster",
        batch_key=None,
        spatial_key="spatial",
        latent_key=f"{method}_latent",
        n_jobs=1,
        seed=SEED,
    )
    paper = {k: float(v) for k, v in paper.items() if v is not None}
    paper["clisis"] = _compute_clisis(
        np.asarray(spatial, dtype=np.float64),
        np.asarray(latent, dtype=np.float64),
        published.astype(str),
        seed=SEED,
    )
    paper["ari_vs_published"] = float(
        adjusted_rand_score(published.astype(str), labels.astype(str))
    )
    paper["nmi_vs_published"] = float(
        normalized_mutual_info_score(published.astype(str), labels.astype(str))
    )
    paper["n_clusters"] = int(len(np.unique(labels)))
    return paper


def main() -> None:
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "methods").mkdir(exist_ok=True)
    NC_WORK.mkdir(parents=True, exist_ok=True)

    print("Loading squidpy slideseqv2 (mouse hippocampus / functional brain)...")
    adata = sq.datasets.slideseqv2()
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(adata.n_obs, size=min(N_CELLS, adata.n_obs), replace=False))
    adata = adata[idx].copy()
    print(f"subset {adata.n_obs} × {adata.n_vars}")

    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    published = np.asarray(adata.obs["cluster"].astype(str).to_numpy())
    pub_categories = sorted(set(published))
    full_cats = list(adata.obs["cluster"].astype("category").cat.categories)
    pub_palette = {
        cat: str(adata.uns["cluster_colors"][i % len(adata.uns["cluster_colors"])])
        for i, cat in enumerate(full_cats)
    }
    pub_palette = {c: pub_palette.get(c, "#999999") for c in pub_categories}

    np.save(OUT / "spatial.npy", spatial)
    maxlen = max(len(s) for s in published)
    np.save(OUT / "published_labels.npy", np.array(published, dtype=f"<U{maxlen}"))

    runners = {
        "leiden": run_leiden,
        "banksy": run_banksy,
        "covet": run_covet,
        "spaceflow": run_spaceflow,
        "nichecompass": run_nichecompass,
    }

    metrics_path = OUT / "metrics.csv"
    prior_metrics = {}
    if metrics_path.is_file():
        prior_df = pd.read_csv(metrics_path).set_index("method")
        prior_metrics = prior_df.to_dict(orient="index")

    metrics_rows = []
    method_meta = {}
    for method, runner in runners.items():
        label_path = OUT / "methods" / f"{method}_labels.npy"
        latent_path = OUT / "methods" / f"{method}_latent.npy"
        umap_path = OUT / "methods" / f"{method}_umap.npy"
        if label_path.is_file() and latent_path.is_file() and umap_path.is_file() and method in prior_metrics:
            print(f"\n=== Skipping {method} (cached) ===")
            labels = np.load(label_path, allow_pickle=False).astype(str)
            latent = np.load(latent_path, allow_pickle=False).astype(np.float64)
            m = {k: prior_metrics[method][k] for k in prior_metrics[method]}
            m["method"] = method
            m["n_clusters"] = int(len(np.unique(labels)))
            metrics_rows.append(m)
            method_meta[method] = {
                "n_clusters": m["n_clusters"],
                "palette": _tab20_palette(m["n_clusters"]),
                "latent_dim": int(latent.shape[1]),
            }
            continue

        print(f"\n=== Running {method} ===")
        labels, latent = runner(adata)
        labels = np.asarray(labels).astype(str)
        latent = np.asarray(latent, dtype=np.float64)
        print(f"  labels={len(np.unique(labels))} clusters; latent={latent.shape}")

        print(f"  UMAP for {method}...")
        um = umap.UMAP(
            n_neighbors=15,
            min_dist=0.3,
            random_state=SEED,
            n_jobs=1,
        ).fit_transform(latent)

        np.save(label_path, labels)
        np.save(latent_path, latent.astype(np.float32))
        np.save(umap_path, um.astype(np.float32))

        print(f"  Metrics for {method}...")
        m = compute_metrics_for_method(spatial, published, labels, latent, method)
        m["method"] = method
        metrics_rows.append(m)
        method_meta[method] = {
            "n_clusters": m["n_clusters"],
            "palette": _tab20_palette(m["n_clusters"]),
            "latent_dim": int(latent.shape[1]),
        }
        print(f"  {method}: {m}")
        # incremental save for resume
        pd.DataFrame(metrics_rows).set_index("method").to_csv(metrics_path)

    metrics_df = pd.DataFrame(metrics_rows).set_index("method")
    metrics_df.to_csv(metrics_path)
    metrics_json = {
        method: {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()}
        for method, row in metrics_df.to_dict(orient="index").items()
    }

    meta = {
        "dataset": "squidpy.datasets.slideseqv2",
        "dataset_description": (
            "Slide-seq V2 mouse hippocampus (Stickels et al.), the NicheCompass "
            "paper functional-brain niche analysis setting."
        ),
        "n_obs_full": 41786,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "seed": SEED,
        "cluster_key_published": "cluster",
        "published_palette": pub_palette,
        "methods": METHODS,
        "method_meta": method_meta,
        "metrics": metrics_json,
        "metric_names": METRICS + ["ari_vs_published", "nmi_vs_published"],
        "metric_groups": {
            "spatial_consistency": ["cas", "mlami", "clisis", "gcs"],
            "niche_coherence": ["cnmi", "nasw"],
            "label_agreement": ["ari_vs_published", "nmi_vs_published"],
        },
        "params": {
            "leiden_resolution": LEIDEN_RESOLUTION,
            "banksy_lambda": BANKSY_LAMBDA,
            "spaceflow_epochs": SPACEFLOW_EPOCHS,
            "nichecompass_epochs": NICHECOMPASS_EPOCHS,
            "nichecompass_gps": ["omnipath", "mebocost"],
        },
        "gene_example": "Hpca",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote assets to {OUT}")
    print(metrics_df[METRICS + ["ari_vs_published", "nmi_vs_published", "n_clusters"]])


if __name__ == "__main__":
    main()
