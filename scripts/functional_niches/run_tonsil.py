"""
Run SpatialFunctionalNicheModel on tonsil betadata.

Target genes: BCL6, IL21, IL7, IL7R, PAX5, AICDA, CXCR4
Target biology: distinguish GC B cells by Tfh proximity (DZ / IZ / LZ)
Target metric:  ARI/NMI vs cell_type_2 (which encodes GC zones)

Usage:
    cd SpaceTravLR_rust
    PYTHONPATH=scripts python3 scripts/functional_niches/run_tonsil.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anndata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import seaborn as sns
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from functional_niches.dataset import build_spatial_graph
from functional_niches.functional_model import (
    SpatialFunctionalModel, TripletSpatialLoss, TfhRankingLoss,
    build_spatial_features, train_functional,
)
from functional_niches.cluster import cluster_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
FEATHER_DIR  = "/ix1/ylee/kor11/djishnu_kor11/tonsil_ablation/runs/tf_lr_tfl__full_2"
H5AD_PATH    = "/ix1/ylee/kor11/djishnu_kor11/tonsil_ablation/snrna_human_tonsil.h5ad"
OUT_DIR      = "/tmp/tonsil_func_signed"
TARGET_GENES = {
    # Core GC B cell transcription factors
    "BCL6", "AICDA", "PAX5", "IRF4", "PRDM1", "FOXO1", "BACH2", "MYBL1",
    # DZ vs LZ markers
    "CXCR4", "CXCR5", "SELL", "CD83", "CD86",
    # B-T interaction / Tfh signals
    "IL21", "IL7", "IL7R", "IL4", "IL2RA", "ICOS", "PDCD1",
    "CXCL13", "LTB", "LTB4R",
    # Apoptosis / selection signals
    "FAS",
    # Cytokines / niche signals
    "IL6", "IL6R",
    # MHC / Ag presentation (LZ function)
    "CD74", "HLA-DRA", "HLA-DRB1",
    # Stromal interaction
    "CXCL12",
    # Plasma cell exit
    "CD28",
}
N_WORKERS    = 16
# ──────────────────────────────────────────────────────────────────


# ── Parallel feather loading ───────────────────────────────────────

def _get_schema(p: Path) -> list[str]:
    with pa.memory_map(str(p), "r") as src:
        import pyarrow.ipc as ipc
        return ipc.open_file(src).schema.names


def _load_one(path: Path, cell_ids: list[str], mod_vocab: dict[str, int]):
    tbl       = feather.read_table(path)
    df        = tbl.to_pandas()
    id_col    = "CellID" if "CellID" in df.columns else df.columns[0]
    beta_cols = [c for c in df.columns if c.startswith("beta_")]
    if not beta_cols:
        return None
    df        = df.set_index(id_col).reindex(cell_ids)
    betas     = df[beta_cols].fillna(0).values.astype(np.float32)
    mod_idx   = np.array([mod_vocab[c] for c in beta_cols], dtype=np.int64)
    gene_name = path.stem.replace("_betadata", "")
    return gene_name, mod_idx, betas


def build_vocab_parallel(paths, n_workers=16):
    all_mods: set[str] = set()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for cols in ex.map(_get_schema, paths):
            all_mods.update(c for c in cols if c.startswith("beta_"))
    return {name: i for i, name in enumerate(sorted(all_mods))}


def load_feathers(paths, cell_ids, mod_vocab, n_workers=16):
    """
    Returns
    -------
    beta_X      : [N, G × M]  signed beta values, one block per gene
                  (preserves both magnitude and direction of every regulator)
    gene_activity: [N, G]     mean|β| per gene — used as rec_target
    gene_names  : list[str]   gene order (matches columns of beta_X blocks)
    """
    n_cells   = len(cell_ids)
    n_mods    = len(mod_vocab)
    results: list[tuple[str, np.ndarray, np.ndarray]] = []

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_load_one, p, cell_ids, mod_vocab): p for p in paths}
        for fut in as_completed(futs):
            res = fut.result()
            if res is None:
                continue
            gene_name, mod_idx, betas = res
            results.append((gene_name, mod_idx, betas))

    # Sort by gene name for deterministic column order
    results.sort(key=lambda x: x[0])
    gene_names = [r[0] for r in results]

    # Build signed beta blocks [N, M] per gene, then concatenate → [N, G×M]
    blocks        = []
    activity_cols = []
    for gene_name, mod_idx, betas in results:
        block = np.zeros((n_cells, n_mods), dtype=np.float32)
        block[:, mod_idx] = betas                        # signed betas in place
        blocks.append(block)
        activity_cols.append(np.abs(betas).mean(axis=1)) # mean|β| scalar per gene

    beta_X       = np.concatenate(blocks, axis=1)        # [N, G × M]
    gene_activity = np.stack(activity_cols, axis=1)       # [N, G]

    log.info(f"  beta_X (signed, G×M): {beta_X.shape}   "
             f"gene_activity (|β|): {gene_activity.shape}")
    return beta_X, gene_activity, gene_names


# ── Evaluation ────────────────────────────────────────────────────

def evaluate(z, true_labels, resolutions=(0.15, 0.20, 0.25, 0.30)):
    best = {"ari": -1.0, "nmi": 0.0, "res": None, "n": 0, "labels": None}
    for res in resolutions:
        r    = cluster_embeddings(z, resolutions=[res])
        pred = r[res].astype(str)
        ari  = adjusted_rand_score(true_labels, pred)
        nmi  = normalized_mutual_info_score(true_labels, pred, average_method="arithmetic")
        if ari > best["ari"]:
            best = {"ari": ari, "nmi": nmi, "res": res,
                    "n": len(set(pred)), "labels": pred}
    return best


# ── Plotting ──────────────────────────────────────────────────────

_PAL = (["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4",
          "#f032e6","#bfef45","#469990","#dcbeff","#9a6324","#800000",
          "#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9","#ffe119",
          "#4e9ddb","#c0a040"])


def _pal(labels):
    unique = sorted(set(labels), key=lambda x: (int(x) if x.isdigit() else 999, x))
    cm = {k: _PAL[i % len(_PAL)] for i, k in enumerate(unique)}
    return cm, [cm[l] for l in labels]


def _handles(cm):
    return [plt.Line2D([0],[0], marker="o", color="w",
                        markerfacecolor=v, markersize=7, label=str(k))
            for k, v in cm.items()]


def plot_results(
    spatial_coords, umap_coords, z,
    niche_labels, cell_type, cell_type_2,
    tfh_dist, best_meta, out_dir,
):
    n  = len(niche_labels)
    ari = best_meta["ari"]; nmi = best_meta["nmi"]

    niche_cm, niche_col = _pal(niche_labels)
    ct_cm,    ct_col    = _pal(cell_type)
    ct2_cm,   ct2_col   = _pal(cell_type_2)

    # Continuous Tfh distance coloring (for context)
    tfh_norm = (tfh_dist - tfh_dist.min()) / (tfh_dist.ptp() + 1e-6)

    # ── 1. Spatial: niches / cell_type_2 / Tfh proximity ──────────
    fig, axes = plt.subplots(1, 3, figsize=(30, 9))
    for ax, colors, cm, title in [
        (axes[0], niche_col, niche_cm,
         f"Functional microniches ({best_meta['n']})\nARI={ari:.3f}  NMI={nmi:.3f}"),
        (axes[1], ct2_col,   ct2_cm,   "cell_type_2 (reference)"),
        (axes[2], ct_col,    ct_cm,    "cell_type"),
    ]:
        ax.scatter(spatial_coords[:,0], spatial_coords[:,1],
                   c=colors, s=5, alpha=0.85, rasterized=True)
        ax.legend(handles=_handles(cm), bbox_to_anchor=(1.01,1), loc="upper left",
                  fontsize=7, frameon=True, ncol=max(1, len(cm)//15))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal"); ax.invert_yaxis()
    plt.suptitle("Tonsil — SpatialFunctionalNicheModel", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "spatial.png"), dpi=180, bbox_inches="tight")
    plt.close()
    log.info("  Saved spatial.png")

    # ── 2. Spatial: Tfh proximity heatmap (gold standard visual) ──
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    sc = axes[0].scatter(spatial_coords[:,0], spatial_coords[:,1],
                          c=tfh_norm, cmap="RdYlBu_r", s=5, alpha=0.9, rasterized=True)
    plt.colorbar(sc, ax=axes[0], label="Tfh proximity (norm)")
    axes[0].set_title("Tfh proximity (closer = red)", fontsize=11)
    axes[0].set_aspect("equal"); axes[0].invert_yaxis()
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    axes[1].scatter(spatial_coords[:,0], spatial_coords[:,1],
                    c=niche_col, s=5, alpha=0.85, rasterized=True)
    axes[1].legend(handles=_handles(niche_cm), bbox_to_anchor=(1.01,1),
                   loc="upper left", fontsize=7, frameon=True)
    axes[1].set_title(f"Functional niches (ARI={ari:.3f})", fontsize=11)
    axes[1].set_aspect("equal"); axes[1].invert_yaxis()
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    plt.suptitle("Tfh proximity vs functional niches — Tonsil", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "spatial_tfh_proximity.png"), dpi=180, bbox_inches="tight")
    plt.close()
    log.info("  Saved spatial_tfh_proximity.png")

    # ── 3. UMAP of model embeddings ────────────────────────────────
    log.info("  Computing UMAP …")
    import umap as umap_lib
    coords = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(z)
    np.save(str(Path(out_dir) / "umap_coords.npy"), coords)

    fig, axes = plt.subplots(1, 3, figsize=(28, 8))
    for ax, colors, cm, title in [
        (axes[0], niche_col, niche_cm, f"Functional niches  (ARI={ari:.3f})"),
        (axes[1], ct2_col,   ct2_cm,   "cell_type_2"),
        (axes[2], ct_col,    ct_cm,    "cell_type"),
    ]:
        ax.scatter(coords[:,0], coords[:,1], c=colors, s=4, alpha=0.7, rasterized=True)
        ax.legend(handles=_handles(cm), bbox_to_anchor=(1.01,1), loc="upper left",
                  fontsize=7, ncol=max(1, len(cm)//15))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    plt.suptitle("UMAP of SpatialFunctionalModel — Tonsil", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "umap.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # ── 4. UMAP coloured by Tfh distance (gold standard) ──────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sc = axes[0].scatter(coords[:,0], coords[:,1], c=tfh_norm,
                          cmap="RdYlBu_r", s=3, alpha=0.7, rasterized=True)
    plt.colorbar(sc, ax=axes[0], label="Tfh proximity")
    axes[0].set_title("Tfh proximity on model UMAP", fontsize=11)
    axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")

    axes[1].scatter(umap_coords[:,0], umap_coords[:,1],
                    c=niche_col, s=3, alpha=0.7, rasterized=True)
    axes[1].legend(handles=_handles(niche_cm), bbox_to_anchor=(1.01,1),
                   loc="upper left", fontsize=7)
    axes[1].set_title("Functional niches on scRNA UMAP", fontsize=11)
    axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "umap_tfh.png"), dpi=180, bbox_inches="tight")
    plt.close()
    log.info("  Saved umap.png  umap_tfh.png")

    # ── 5. Composition heatmap ──────────────────────────────────────
    comp = pd.crosstab(
        pd.Series(niche_labels, name="niche"),
        pd.Series(cell_type_2,  name="cell_type_2"),
        normalize="index",
    ).reindex(sorted(set(niche_labels), key=int))

    h = max(5, len(comp) * 0.45)
    w = max(12, len(comp.columns) * 0.65)
    fig, ax = plt.subplots(figsize=(w, h))
    sns.heatmap(comp, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
                linewidths=0.4, cbar_kws={"label": "fraction of niche"})
    ax.set_title(f"Niche composition  ARI={ari:.4f}  NMI={nmi:.4f}", fontsize=12)
    ax.set_xlabel("cell_type_2"); ax.set_ylabel("Functional niche")
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "composition.png"), dpi=180, bbox_inches="tight")
    plt.close()
    log.info("  Saved composition.png")

    # ── 6. Tfh distance distribution per niche ─────────────────────
    df_plot = pd.DataFrame({
        "niche":    niche_labels,
        "tfh_dist": tfh_dist,
        "ct2":      cell_type_2,
    })
    niche_order = sorted(set(niche_labels), key=int)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ax = axes[0]
    ax.boxplot(
        [df_plot.loc[df_plot["niche"] == n, "tfh_dist"].values for n in niche_order],
        labels=niche_order,
    )
    ax.set_xlabel("Functional niche"); ax.set_ylabel("Distance to nearest Tfh")
    ax.set_title("Tfh proximity per niche (↓ = closer to Tfh / Light Zone)", fontsize=11)
    ax.tick_params(axis="x", rotation=45)

    # GC B cells only: Tfh dist per ct2 zone
    gc_df = df_plot[df_plot["ct2"].isin(["GC Dark Zone","GC Light Zone","GC Intermediate Zone"])]
    ax2 = axes[1]
    ax2.boxplot(
        [gc_df.loc[gc_df["ct2"] == z, "tfh_dist"].values
         for z in ["GC Dark Zone","GC Intermediate Zone","GC Light Zone"]],
        labels=["DZ","IZ","LZ"],
    )
    ax2.set_xlabel("GC zone (ground truth)"); ax2.set_ylabel("Distance to nearest Tfh")
    ax2.set_title("Tfh proximity per GC zone (ground truth)", fontsize=11)

    plt.suptitle("Tfh proximity analysis — functional niches vs GC zones", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(Path(out_dir) / "tfh_dist_per_niche.png"), dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved tfh_dist_per_niche.png")


# ── Main ──────────────────────────────────────────────────────────

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Load h5ad
    log.info("Loading h5ad …")
    adata          = anndata.read_h5ad(H5AD_PATH)
    cell_ids       = list(adata.obs_names)
    spatial_coords = adata.obsm["spatial"].astype(np.float32)
    umap_coords    = adata.obsm["X_umap"].astype(np.float32)
    cell_type      = adata.obs["cell_type"].values.astype(str)
    cell_type_2    = adata.obs["cell_type_2"].values.astype(str)
    gc_mask        = (cell_type == "B_germinal_center")
    log.info(f"  {len(cell_ids)} cells  ·  {gc_mask.sum()} GC B cells  "
             f"·  {(cell_type=='T_follicular_helper').sum()} Tfh")

    # 2. Load target feathers
    all_feathers  = sorted(Path(FEATHER_DIR).glob("*_betadata.feather"))
    feather_paths = [p for p in all_feathers
                     if p.stem.replace("_betadata","") in TARGET_GENES]
    found = sorted({p.stem.replace("_betadata","") for p in feather_paths})
    log.info(f"Loading {len(feather_paths)} feathers: {found}")

    mod_vocab = build_vocab_parallel(feather_paths, n_workers=N_WORKERS)
    beta_X, gene_activity, gene_names = load_feathers(
        feather_paths, cell_ids, mod_vocab, n_workers=N_WORKERS
    )
    log.info(f"  genes: {sorted(gene_names)}")

    # 3. Spatial composition features
    log.info("Building spatial composition features …")
    spat_X, tfh_dist = build_spatial_features(
        spatial_coords, cell_type,
        ks=(10, 30, 60), tfh_k=10, rbf_sigma=60.0,
    )

    # 4. Spatial graph (k=6 was best in round 2 grid)
    edge_index, edge_weight = build_spatial_graph(spatial_coords, k=6)

    # 5. Train
    log.info("Training SpatialFunctionalModel …")
    z = train_functional(
        beta_X      = beta_X,          # [N, G×M] signed betas — full regulatory info
        spat_X      = spat_X,
        rec_target  = gene_activity,   # [N, G] mean|β| per gene — reconstruction target
        tfh_dist    = tfh_dist,
        gc_mask     = gc_mask,
        edge_index  = edge_index,
        edge_weight = edge_weight,
        cell_ids    = cell_ids,
        output_dir  = OUT_DIR,
        hidden_dim  = 64,
        mlp_layers  = 2,
        gcn_layers  = 2,
        epochs      = 800,
        lr          = 1e-3,
        w_triplet   = 1.0,
        w_rec       = 0.05,
        w_smooth    = 0.3,
        w_tfh_rank  = 2.0,
        w_nbr_comp  = 0.5,
        device_str  = "auto",
        log_every   = 100,
    )

    # 6. Cluster
    log.info("Clustering …")
    best = evaluate(z, cell_type_2, resolutions=(0.15, 0.20, 0.25, 0.30))
    log.info(f"Best: ARI={best['ari']:.4f}  NMI={best['nmi']:.4f}  "
             f"n={best['n']}  res={best['res']}")

    niche_labels = best["labels"]
    # Sort niches by mean Tfh distance (niche 0 = closest to Tfh = LZ-enriched)
    niche_tfh_means = {n: tfh_dist[niche_labels == n].mean() for n in set(niche_labels)}
    tfh_order = sorted(niche_tfh_means, key=niche_tfh_means.get)
    remap = {old: str(i) for i, old in enumerate(tfh_order)}
    niche_labels = np.array([remap[n] for n in niche_labels])

    # Save
    pd.DataFrame({
        "CellID": cell_ids, "niche": niche_labels,
        "cell_type": cell_type, "cell_type_2": cell_type_2,
        "tfh_dist": tfh_dist,
    }).to_parquet(Path(OUT_DIR) / "niche_labels.parquet", index=False)
    np.save(str(Path(OUT_DIR) / "embeddings.npy"), z)

    # Report GC zone composition per niche
    gc_df = pd.DataFrame({"niche": niche_labels[gc_mask], "ct2": cell_type_2[gc_mask]})
    log.info("\nGC B cells: niche × zone distribution")
    log.info(pd.crosstab(gc_df["niche"], gc_df["ct2"]).to_string())

    # 7. Plots
    log.info("Generating plots …")
    plot_results(
        spatial_coords, umap_coords, z,
        niche_labels, cell_type, cell_type_2,
        tfh_dist, best, OUT_DIR,
    )
    log.info(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
