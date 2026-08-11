#!/usr/bin/env python3
"""Compare SpaceTravLR β niches vs BANKSY / COVET / NicheCompass on tonsil GC.

Reuses the NicheCompass single-sample metric suite from prepare_niche_benchmark.py
(CAS, MLAMI, CLISIS, GCS, CNMI, NASW). LZ/DZ (cell_type_2) is reported only as a
confounded expression-derived reference — primary comparison is method-vs-method
and the unsupervised spatial / niche-coherence scores.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("HOME", "/tmp/fakehome")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
Path(os.environ["HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import scanpy as sc
import squidpy as sq
import umap
from anndata import AnnData
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

# Reuse runners / metrics from the hippocampus prep script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_niche_benchmark import (  # noqa: E402
    BANKSY_LAMBDA,
    LEIDEN_RESOLUTION,
    METRICS,
    _leiden_on_rep,
    _tab20_palette,
    compute_metrics_for_method,
    run_banksy,
)

H5AD = Path("/ix1/ylee/kor11/djishnu_kor11/tonsil_ablation/snrna_human_tonsil_processed.h5ad")
BETA_OUT = Path("/ix1/ylee/kor11/djishnu_kor11/tonsil_ablation/snrna_human_tonsil_2026-08-10")
_PUBLIC = Path(__file__).resolve().parent / "public" / "tonsil_niche_benchmark"
_KEPT_CANDIDATES = [
    _PUBLIC / "beta_features_kept.csv",
    Path("/tmp/gc_beta_spatial_filter/data/beta_features_kept.csv"),
]
KEPT_BETAS = next((p for p in _KEPT_CANDIDATES if p.is_file()), _KEPT_CANDIDATES[0])
OUT = Path(os.environ.get("TONSIL_NICHE_OUT", str(_PUBLIC)))
NC_WORK = Path("/tmp/nichebench_work_tonsil")
GP_ROOT = Path("/tmp/nichecompass_src/data")
SEED = 0
NICHECOMPASS_EPOCHS = 80
METHODS = ["spacetravlr_beta", "banksy", "covet", "nichecompass"]


def load_gc_adata() -> AnnData:
    adata = sc.read_h5ad(H5AD)
    gc = adata.obs["cell_type"].astype(str) == "B_germinal_center"
    ad = adata[gc].copy()
    # BANKSY expects x/y obs columns in this codebase's initialize_banksy call.
    xy = np.asarray(ad.obsm["spatial"], dtype=np.float64)
    ad.obs["x"] = xy[:, 0]
    ad.obs["y"] = xy[:, 1]
    # .X is already scaled; restore raw counts for BANKSY / COVET / NicheCompass.
    if "raw_count" in ad.layers:
        ad.layers["counts"] = ad.layers["raw_count"].copy()
        ad.X = ad.layers["raw_count"].copy()
    elif "normalized_count" in ad.layers:
        ad.layers["counts"] = ad.layers["normalized_count"].copy()
        ad.X = ad.layers["normalized_count"].copy()
    else:
        ad.layers["counts"] = ad.X.copy()
    return ad


def run_covet(adata: AnnData) -> tuple[np.ndarray, np.ndarray]:
    from scenvi.utils import compute_covet
    from sklearn.decomposition import PCA

    ad = adata.copy()
    # Ensure non-negative expression for HVG / COVET.
    if "raw_count" in ad.layers:
        ad.X = ad.layers["raw_count"].copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    if "highly_variable" in ad.var.columns:
        del ad.var["highly_variable"]
    sc.pp.highly_variable_genes(ad, n_top_genes=min(64, ad.n_vars - 1), flavor="seurat", subset=False)
    ad = ad[:, ad.var["highly_variable"]].copy()
    g = int(ad.n_vars)
    _, covet_sqrt, _ = compute_covet(
        ad,
        k=8,
        g=g,
        spatial_key="spatial",
        batch_size=512,
    )
    flat = np.asarray(covet_sqrt, dtype=np.float64).reshape(ad.n_obs, -1)
    latent = PCA(n_components=min(30, flat.shape[1] - 1), random_state=SEED).fit_transform(flat)
    ad.obsm["X_covet"] = latent
    labels = _leiden_on_rep(ad, "X_covet", "leiden_covet", LEIDEN_RESOLUTION)
    return labels, latent.astype(np.float64)


def run_banksy_counts(adata: AnnData) -> tuple[np.ndarray, np.ndarray]:
    """BANKSY on raw-count-derived log-normalised matrix (not pre-scaled .X)."""
    ad = adata.copy()
    if "raw_count" in ad.layers:
        ad.X = ad.layers["raw_count"].copy()
    return run_banksy(ad)


def run_spacetravlr_beta(adata: AnnData) -> tuple[np.ndarray, np.ndarray]:
    """Spatially filtered SpaceTravLR βs → z-score → PCA → Leiden."""
    kept = pd.read_csv(KEPT_BETAS)
    # Load only needed gene feathers
    genes = sorted(kept["gene"].unique())
    gene_mats = {}
    for g in genes:
        df = feather.read_feather(BETA_OUT / f"{g}_betadata.feather")
        # adata is GC-subset; feather is full tissue — align by CellID
        full_ids = df["CellID"].astype(str).to_numpy()
        idx = pd.Index(full_ids).get_indexer(adata.obs_names.astype(str))
        if (idx < 0).any():
            raise RuntimeError(f"CellID mismatch loading {g}")
        mat = df.drop(columns=["CellID"])
        cols = {c: j for j, c in enumerate(mat.columns)}
        gene_mats[g] = (mat.to_numpy(dtype=np.float32)[idx], cols)

    blocks = []
    for row in kept.itertuples(index=False):
        X, cols = gene_mats[row.gene]
        feat = row.feature
        if feat not in cols:
            raise KeyError(f"missing {row.gene}::{feat}")
        blocks.append(X[:, cols[feat]])
    Xb = np.column_stack(blocks)
    keep = Xb.std(0) > 1e-8
    Xz = StandardScaler().fit_transform(Xb[:, keep])
    ad = AnnData(Xz)
    ad.obs_names = adata.obs_names.copy()
    n_pcs = min(40, Xz.shape[1] - 1, Xz.shape[0] - 1)
    sc.tl.pca(ad, n_comps=n_pcs, svd_solver="arpack")
    labels = _leiden_on_rep(ad, "X_pca", "leiden_beta", LEIDEN_RESOLUTION)
    return labels, np.asarray(ad.obsm["X_pca"], dtype=np.float64)


def run_nichecompass_human(adata: AnnData) -> tuple[np.ndarray, np.ndarray]:
    """NicheCompass with human OmniPath / MEBOCOST priors."""
    import pickle

    from nichecompass.models import NicheCompass
    from nichecompass.utils import (
        add_gps_from_gp_dict_to_adata,
        extract_gp_dict_from_mebocost_ms_interactions,
        extract_gp_dict_from_omnipath_lr_interactions,
        filter_and_combine_gp_dict_gps_v2,
    )

    ad = adata.copy()
    if "counts" not in ad.layers:
        ad.layers["counts"] = ad.X.copy()
    species = "human"
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

    NC_WORK.mkdir(parents=True, exist_ok=True)
    gp_pickle = NC_WORK / "combined_gp_dict_human.pkl"
    if gp_pickle.is_file():
        with open(gp_pickle, "rb") as f:
            combined = pickle.load(f)
        print(f"  Loaded cached human GP dict ({len(combined)} programs)")
    else:
        ga = GP_ROOT / "gene_annotations" / "human_mouse_gene_orthologs.csv"
        gp_dir = GP_ROOT / "gene_programs"
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
        n_epochs_all_gps=15,
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


def pairwise_ari(labels_by_method: dict[str, np.ndarray]) -> pd.DataFrame:
    methods = list(labels_by_method)
    mat = np.zeros((len(methods), len(methods)))
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            mat[i, j] = adjusted_rand_score(labels_by_method[a], labels_by_method[b])
    return pd.DataFrame(mat, index=methods, columns=methods)


def main() -> None:
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "methods").mkdir(exist_ok=True)

    print("Loading tonsil GC B cells...")
    adata = load_gc_adata()
    print(f"  {adata.n_obs} × {adata.n_vars}")
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    # Confounded expression-derived zones — kept only for optional reporting.
    zone = np.asarray(adata.obs["cell_type_2"].astype(str).to_numpy())

    runners = {
        "spacetravlr_beta": run_spacetravlr_beta,
        "banksy": run_banksy_counts,
        "covet": run_covet,
        "nichecompass": run_nichecompass_human,
    }

    metrics_path = OUT / "metrics.csv"
    prior_metrics = {}
    if metrics_path.is_file():
        prior_df = pd.read_csv(metrics_path).set_index("method")
        prior_metrics = prior_df.to_dict(orient="index")

    metrics_rows = []
    method_meta = {}
    labels_by_method: dict[str, np.ndarray] = {}
    latents: dict[str, np.ndarray] = {}

    for method, runner in runners.items():
        label_path = OUT / "methods" / f"{method}_labels.npy"
        latent_path = OUT / "methods" / f"{method}_latent.npy"
        umap_path = OUT / "methods" / f"{method}_umap.npy"

        if (
            label_path.is_file()
            and latent_path.is_file()
            and umap_path.is_file()
            and method in prior_metrics
        ):
            print(f"\n=== Skipping {method} (cached) ===")
            labels = np.load(label_path, allow_pickle=False).astype(str)
            latent = np.load(latent_path, allow_pickle=False).astype(np.float64)
            m = dict(prior_metrics[method])
            m["method"] = method
            m["n_clusters"] = int(len(np.unique(labels)))
            metrics_rows.append(m)
            labels_by_method[method] = labels
            latents[method] = latent
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
        um = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=SEED, n_jobs=1).fit_transform(
            latent
        )
        np.save(label_path, labels)
        np.save(latent_path, latent.astype(np.float32))
        np.save(umap_path, um.astype(np.float32))

        print(f"  Metrics for {method}...")
        # Use zone labels only inside the metric function for CAS/CNMI machinery;
        # they are marked confounded in the report.
        m = compute_metrics_for_method(spatial, zone, labels, latent, method)
        m["method"] = method
        # Rename confounded agreement metrics for clarity in outputs.
        m["ari_vs_zone_confounded"] = m.pop("ari_vs_published")
        m["nmi_vs_zone_confounded"] = m.pop("nmi_vs_published")
        metrics_rows.append(m)
        labels_by_method[method] = labels
        latents[method] = latent
        method_meta[method] = {
            "n_clusters": m["n_clusters"],
            "palette": _tab20_palette(m["n_clusters"]),
            "latent_dim": int(latent.shape[1]),
        }
        print(f"  {method}: {m}")
        pd.DataFrame(metrics_rows).set_index("method").to_csv(metrics_path)

    metrics_df = pd.DataFrame(metrics_rows).set_index("method")
    metrics_df.to_csv(metrics_path)

    pw = pairwise_ari(labels_by_method)
    pw.to_csv(OUT / "pairwise_ari.csv")
    print("\nPairwise ARI (method agreement):")
    print(pw)

    np.save(OUT / "spatial.npy", spatial)
    maxlen = max(len(s) for s in zone)
    np.save(OUT / "zone_labels_confounded.npy", np.array(zone, dtype=f"<U{maxlen}"))

    metric_names = METRICS + ["ari_vs_zone_confounded", "nmi_vs_zone_confounded"]
    metrics_json = {
        method: {
            k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()
        }
        for method, row in metrics_df.to_dict(orient="index").items()
    }
    zone_cats = sorted(set(map(str, zone)))
    default_zone_palette = {
        "GC Light Zone": "#2A9D8F",
        "GC Dark Zone": "#E76F51",
        "GC Intermediate Zone": "#E9C46A",
        "FDC": "#6D6875",
        "plasma": "#B5838D",
        "B_naive": "#457B9D",
        "B_memory": "#1D3557",
        "T_follicular_helper": "#9B5DE5",
        "T_CD8": "#F15BB5",
    }
    zone_palette = {
        cat: default_zone_palette.get(cat, _tab20_palette(len(zone_cats)).get(str(i), "#999999"))
        for i, cat in enumerate(zone_cats)
    }
    meta = {
        "dataset": "snrna_human_tonsil_processed · B_germinal_center",
        "dataset_description": (
            "Human tonsil snRNA-seq germinal-center B cells (cell_type==B_germinal_center). "
            "SpaceTravLR spatially filtered β niches vs BANKSY / COVET / NicheCompass using "
            "the NicheCompass single-sample metric suite. cell_type_2 GC Light/Dark/Intermediate "
            "Zone labels are expression-derived and treated as a confounded reference only — "
            "primary comparison is unsupervised spatial/niche metrics and pairwise ARI."
        ),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "seed": SEED,
        "methods": METHODS,
        "method_meta": method_meta,
        "metrics": metrics_json,
        "metric_names": metric_names,
        "metric_groups": {
            "spatial_consistency": ["cas", "mlami", "clisis", "gcs"],
            "niche_coherence": ["cnmi", "nasw"],
            "zone_agreement_confounded": ["ari_vs_zone_confounded", "nmi_vs_zone_confounded"],
        },
        "pairwise_ari": pw.to_dict(),
        "zone_key": "cell_type_2",
        "zone_palette": zone_palette,
        "display_names": {
            "spacetravlr_beta": "SpaceTravLR β",
            "banksy": "BANKSY",
            "covet": "COVET",
            "nichecompass": "NicheCompass",
        },
        "params": {
            "leiden_resolution": LEIDEN_RESOLUTION,
            "banksy_lambda": BANKSY_LAMBDA,
            "nichecompass_epochs": NICHECOMPASS_EPOCHS,
            "spacetravlr_beta_features": str(KEPT_BETAS),
            "n_beta_features": int(pd.read_csv(KEPT_BETAS).shape[0]),
        },
        "note": (
            "Primary read-out: spatial_consistency + niche_coherence + pairwise ARI. "
            "Do not treat ari_vs_zone_confounded as ground truth. CAS/CNMI also use "
            "expression-derived zone labels internally."
        ),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote assets to {OUT}")
    cols = [c for c in METRICS + ["ari_vs_zone_confounded", "n_clusters"] if c in metrics_df.columns]
    print(metrics_df[cols])


if __name__ == "__main__":
    main()
