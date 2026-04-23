"""
Apply SimpleNicheModel to tonsil betadata and produce spatial niche plots.

Usage:
    cd SpaceTravLR_rust
    PYTHONPATH=scripts python3 scripts/functional_niches/run_tonsil.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anndata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from functional_niches.dataset import build_spatial_graph, FunctionalNicheDataset
from functional_niches.simple_model import train_simple
from functional_niches.cluster import cluster_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
FEATHER_DIR  = "/ix1/ylee/kor11/djishnu_kor11/tonsil_ablation/runs/tf_lr_tfl__full_2"
H5AD_PATH    = "/ix1/ylee/kor11/djishnu_kor11/tonsil_ablation/snrna_human_tonsil.h5ad"
OUT_DIR      = "/tmp/tonsil_niches"
HIDDEN_DIM   = 64
MLP_LAYERS   = 2
GCN_LAYERS   = 2
EPOCHS       = 500
LR           = 1e-3
SPATIAL_K    = 6
RESOLUTION   = 0.5
N_WORKERS    = 16   # parallel feather readers


# ------------------------------------------------------------------
# Step 1: read one feather and return (gene_name, mod_indices, betas)
# ------------------------------------------------------------------
def _load_one_feather(path: Path, cell_ids: list[str], mod_vocab: dict[str, int]):
    tbl      = feather.read_table(path)
    df       = tbl.to_pandas()
    id_col   = "CellID" if "CellID" in df.columns else df.columns[0]
    beta_cols = [c for c in df.columns if c.startswith("beta_")]
    if not beta_cols:
        return None
    df        = df.set_index(id_col).reindex(cell_ids)
    betas     = df[beta_cols].fillna(0).values.astype(np.float32)   # [N, M_g]
    mod_idx   = np.array([mod_vocab[c] for c in beta_cols], dtype=np.int64)
    gene_name = path.stem.replace("_betadata", "")
    return gene_name, mod_idx, betas


# ------------------------------------------------------------------
# Step 2: build vocab (parallel schema reads)
# ------------------------------------------------------------------
def build_vocab_parallel(paths: list[Path], n_workers: int = 16) -> dict[str, int]:
    """Read column names from all feathers in parallel, build mod vocab."""
    all_mods: set[str] = set()

    def _get_cols(p: Path) -> list[str]:
        with pa.memory_map(str(p), "r") as src:
            import pyarrow.ipc as ipc
            schema = ipc.open_file(src).schema
        return [c for c in schema.names if c.startswith("beta_")]

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_get_cols, p): p for p in paths}
        done = 0
        for fut in as_completed(futs):
            all_mods.update(fut.result())
            done += 1
            if done % 200 == 0:
                log.info(f"  Vocab: scanned {done}/{len(paths)} files …")

    return {name: i for i, name in enumerate(sorted(all_mods))}


# ------------------------------------------------------------------
# Step 3: load all feathers in parallel, accumulate X and rec_target
# ------------------------------------------------------------------
def build_beta_matrix_parallel(
    paths: list[Path],
    cell_ids: list[str],
    mod_vocab: dict[str, int],
    n_workers: int = 16,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns
    -------
    X          : [N, n_mods] signed betas summed across genes
    rec_target : [N, n_mods] mean |beta| across genes
    gene_names : list of gene names
    """
    n_cells = len(cell_ids)
    n_mods  = len(mod_vocab)
    X          = np.zeros((n_cells, n_mods), dtype=np.float32)
    rec_acc    = np.zeros((n_cells, n_mods), dtype=np.float32)
    rec_counts = np.zeros(n_mods, dtype=np.float32)
    gene_names: list[str] = []

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_load_one_feather, p, cell_ids, mod_vocab): p for p in paths}
        done = 0
        for fut in as_completed(futs):
            result = fut.result()
            if result is None:
                continue
            gene_name, mod_idx, betas = result
            np.add.at(X,       (slice(None), mod_idx), betas)
            np.add.at(rec_acc, (slice(None), mod_idx), np.abs(betas))
            rec_counts[mod_idx] += 1.0
            gene_names.append(gene_name)
            done += 1
            if done % 200 == 0:
                log.info(f"  Data: loaded {done}/{len(paths)} genes …")

    rec_counts = np.maximum(rec_counts, 1.0)
    rec_target = rec_acc / rec_counts[None, :]
    return X, rec_target, gene_names


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # 1. Load h5ad
    log.info("Loading h5ad …")
    adata          = anndata.read_h5ad(H5AD_PATH)
    cell_ids       = list(adata.obs_names)
    spatial_coords = adata.obsm["spatial"].astype(np.float32)
    cell_type      = adata.obs["cell_type"].values.astype(str)
    umap_coords    = adata.obsm["X_umap"].astype(np.float32)
    log.info(f"  {len(cell_ids)} cells  ·  {len(set(cell_type))} cell types")

    # 2. Build modulator vocabulary
    feather_paths = sorted(Path(FEATHER_DIR).glob("*_betadata.feather"))
    log.info(f"Found {len(feather_paths)} feather files. Building vocabulary …")
    mod_vocab = build_vocab_parallel(feather_paths, n_workers=N_WORKERS)
    n_mods    = len(mod_vocab)
    log.info(f"  {n_mods} unique modulators")

    # 3. Load all betadata → X, rec_target
    log.info("Loading all betadata (parallel) …")
    X, rec_target, gene_names = build_beta_matrix_parallel(
        feather_paths, cell_ids, mod_vocab, n_workers=N_WORKERS
    )
    log.info(f"  X: {X.shape}  non-zero entries: {(X != 0).sum():,}")

    # 4. Build spatial graph
    edge_index, edge_weight = build_spatial_graph(spatial_coords, k=SPATIAL_K)

    # 5. Package dataset
    dataset = FunctionalNicheDataset(
        cell_ids=cell_ids,
        gene_betas=[],
        edge_index=edge_index,
        edge_weight=edge_weight,
        mod_vocab=mod_vocab,
        gene_names=sorted(gene_names),
        rec_target=torch.from_numpy(rec_target).float(),
        _beta_matrix=torch.from_numpy(X).float(),
    )

    # 6. Train
    log.info("Training SimpleNicheModel …")
    z = train_simple(
        dataset=dataset,
        output_dir=OUT_DIR,
        hidden_dim=HIDDEN_DIM,
        mlp_layers=MLP_LAYERS,
        gcn_layers=GCN_LAYERS,
        epochs=EPOCHS,
        lr=LR,
        alpha=0.1,
        beta=0.1,
        device_str="auto",
        log_every=100,
    )

    # 7. Leiden clustering
    log.info("Clustering embeddings …")
    cluster_results = cluster_embeddings(z, resolutions=[RESOLUTION])
    niche_labels    = cluster_results[RESOLUTION].astype(str)
    n_niches        = len(set(niche_labels))
    log.info(f"  {n_niches} niches at resolution {RESOLUTION}")
    log.info("  Niche sizes: " + str(pd.Series(niche_labels).value_counts().sort_index().to_dict()))

    # Save
    pd.DataFrame({
        "CellID":    cell_ids,
        "niche":     niche_labels,
        "cell_type": cell_type,
    }).to_parquet(Path(OUT_DIR) / "niche_labels.parquet", index=False)

    # 8. Plots
    log.info("Generating plots …")
    _plot_spatial(spatial_coords, niche_labels, cell_type, n_niches)
    _plot_umap_side_by_side(umap_coords, niche_labels, cell_type)
    _plot_model_umap(z, niche_labels, cell_type)
    log.info(f"All outputs saved to {OUT_DIR}")


# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------

_PALETTE = (
    ["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
     "#42d4f4","#f032e6","#bfef45","#fabed4","#469990",
     "#dcbeff","#9a6324","#800000","#aaffc3","#808000",
     "#ffd8b1","#000075","#a9a9a9","#ffe119","#ffffff"]
)

def _palette(labels) -> tuple[dict, list]:
    unique = sorted(set(labels), key=lambda x: (int(x) if x.isdigit() else 999, x))
    cmap   = {k: _PALETTE[i % len(_PALETTE)] for i, k in enumerate(unique)}
    colors = [cmap[l] for l in labels]
    return cmap, colors


def _legend_handles(cmap: dict, title: str = "") -> list:
    return [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=v, markersize=7, label=str(k))
        for k, v in cmap.items()
    ]


def _plot_spatial(coords, niche_labels, cell_type, n_niches):
    niche_cmap, niche_colors = _palette(niche_labels)
    ct_cmap,    ct_colors    = _palette(cell_type)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    for ax, colors, cmap, title, subtitle in [
        (axes[0], niche_colors, niche_cmap,
         f"Functional microniches  ({n_niches} niches)",
         "SimpleNicheModel (signed-β MLP + spatial GCN)"),
        (axes[1], ct_colors, ct_cmap,
         "Known cell types  (reference)",
         "from h5ad annotations"),
    ]:
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=5, alpha=0.85, rasterized=True)
        ax.legend(handles=_legend_handles(cmap), bbox_to_anchor=(1.01, 1),
                  loc="upper left", fontsize=8, frameon=True,
                  ncol=max(1, len(cmap) // 15))
        ax.set_title(f"{title}\n{subtitle}", fontsize=12)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal"); ax.invert_yaxis()

    plt.suptitle(
        f"Human Tonsil — Functional Microniches\n"
        f"{len(niche_labels):,} cells · {EPOCHS} epochs · hidden_dim={HIDDEN_DIM}",
        fontsize=13,
    )
    plt.tight_layout()
    out = Path(OUT_DIR) / "spatial_niches.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out}")


def _plot_umap_side_by_side(umap_coords, niche_labels, cell_type):
    """Plot niches and cell types on the original scRNA-seq UMAP."""
    niche_cmap, niche_colors = _palette(niche_labels)
    ct_cmap,    ct_colors    = _palette(cell_type)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, colors, cmap, title in [
        (axes[0], niche_colors, niche_cmap, "Functional niches on original UMAP"),
        (axes[1], ct_colors,    ct_cmap,    "Cell types on original UMAP"),
    ]:
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=colors,
                   s=4, alpha=0.7, rasterized=True)
        ax.legend(handles=_legend_handles(cmap), bbox_to_anchor=(1.01, 1),
                  loc="upper left", fontsize=8, frameon=True,
                  ncol=max(1, len(cmap) // 15))
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    plt.suptitle("Tonsil — functional niches projected on original scRNA-seq UMAP",
                 fontsize=12)
    plt.tight_layout()
    out = Path(OUT_DIR) / "original_umap_niches.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out}")


def _plot_model_umap(z, niche_labels, cell_type):
    """Compute a fresh UMAP from the model embeddings and plot."""
    log.info("  Computing UMAP of model embeddings …")
    import umap as umap_lib
    coords = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(z)

    niche_cmap, niche_colors = _palette(niche_labels)
    ct_cmap,    ct_colors    = _palette(cell_type)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, colors, cmap, title in [
        (axes[0], niche_colors, niche_cmap, "Functional niches"),
        (axes[1], ct_colors,    ct_cmap,    "Cell types"),
    ]:
        ax.scatter(coords[:, 0], coords[:, 1], c=colors,
                   s=4, alpha=0.7, rasterized=True)
        ax.legend(handles=_legend_handles(cmap), bbox_to_anchor=(1.01, 1),
                  loc="upper left", fontsize=8, frameon=True,
                  ncol=max(1, len(cmap) // 15))
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    plt.suptitle("UMAP of SimpleNicheModel embeddings — Tonsil", fontsize=12)
    plt.tight_layout()
    out = Path(OUT_DIR) / "model_umap_niches.png"
    plt.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out}")


if __name__ == "__main__":
    main()
