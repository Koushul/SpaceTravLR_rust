"""
Functional Spatial Niche Embeddings — generalized runner
=========================================================

Point at any AnnData + betadata folder to get per-cell functional
microniche embeddings that reflect both regulatory state and spatial
microenvironment.

Key features
------------
• Automatic marker gene selection: ranks betadata genes by differential
  expression / variance across cell types, selects the top-N that have
  betadata files.
• Automatic spatial-anchor ranking loss: embed cells so that the
  "focus" population is ordered by proximity to an "anchor" cell type
  (e.g. B_germinal_center ordered by distance to T_follicular_helper).
• Fully generalised: no hard-coded cell types, gene names, or paths.

Usage
-----
    PYTHONPATH=scripts python3 scripts/functional_niches/run_niches.py \\
        --h5ad      /path/to/data.h5ad \\
        --feather-dir /path/to/betadata_run/ \\
        --out-dir   /path/to/output/ \\
        [--cell-type-col  cell_type]         # obs column with cell type labels
        [--focus-type     B_germinal_center]  # population to embed functionally
        [--anchor-type    T_follicular_helper]# reference population for proximity
        [--n-genes        30]                 # number of marker genes to use
        [--hidden-dim     64]
        [--epochs         800]
        [--spatial-k      6]
        [--leiden-res     0.2]
        [--n-workers      16]
"""

from __future__ import annotations

import argparse
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
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))
from functional_niches.dataset import build_spatial_graph
from functional_niches.functional_model import (
    SpatialFunctionalModel, TripletSpatialLoss, TfhRankingLoss,
    train_functional,
)
from functional_niches.cluster import cluster_embeddings

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Gene selection
# ──────────────────────────────────────────────────────────────────

def select_marker_genes(
    adata: anndata.AnnData,
    feather_dir: str,
    cell_type_col: str,
    n_genes: int = 30,
    method: str = "wilcoxon",
) -> list[str]:
    """
    Automatically select the top marker genes that:
      1. Are differentially expressed across cell types (scanpy DE)
      2. Have a corresponding *_betadata.feather file

    Returns
    -------
    List of up to n_genes gene names, sorted by DE score.
    """
    import scanpy as sc
    import warnings as _w
    _w.filterwarnings("ignore")

    available = {
        p.stem.replace("_betadata", "")
        for p in Path(feather_dir).glob("*_betadata.feather")
    }
    log.info(f"  {len(available)} genes have betadata files")

    # Run DE (rank_genes_groups) if not already done
    adata_de = adata.copy()
    if adata_de.raw is None:
        # Use log1p-normalised X for DE if not already normalised
        if adata_de.X.max() > 50:    # likely raw counts
            sc.pp.normalize_total(adata_de, target_sum=1e4)
            sc.pp.log1p(adata_de)

    sc.tl.rank_genes_groups(
        adata_de, groupby=cell_type_col, method=method,
        key_added="rank_genes", use_raw=False,
    )
    rg = adata_de.uns["rank_genes"]

    # Collect top markers per group, ranked by score
    scored: dict[str, float] = {}   # gene → max score across groups
    groups = rg["names"].dtype.names
    for grp in groups:
        names  = rg["names"][grp]
        scores = rg["scores"][grp]
        for gene, score in zip(names, scores):
            if gene in available:
                # keep the highest score seen for this gene across groups
                if gene not in scored or score > scored[gene]:
                    scored[gene] = float(score)

    ranked = sorted(scored, key=scored.get, reverse=True)
    selected = ranked[:n_genes]
    log.info(f"  Selected {len(selected)} marker genes: {selected}")
    return selected


# ──────────────────────────────────────────────────────────────────
# Spatial composition features (generalised)
# ──────────────────────────────────────────────────────────────────

def build_spatial_features_general(
    spatial_coords: np.ndarray,
    cell_type:      np.ndarray,
    anchor_type:    str | None,
    ks:             tuple[int, ...] = (10, 30, 60),
    anchor_k:       int = 10,
    rbf_sigma:      float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Build spatial composition features for any dataset.

    Parameters
    ----------
    anchor_type : cell type used as the spatial anchor (e.g. 'T_follicular_helper').
                  If None, no anchor RBF features are added and anchor_dist is None.
    rbf_sigma   : RBF bandwidth for anchor distances.
                  Defaults to the median distance to the nearest anchor cell.

    Returns
    -------
    spat_X      : [N, 3*n_ct + anchor_k]  or  [N, 3*n_ct]  spatial features
    anchor_dist : [N] mean distance to anchor_k nearest anchor cells, or None
    """
    N        = len(spatial_coords)
    ct_types = sorted(set(cell_type))
    n_ct     = len(ct_types)

    feats = []
    for k in ks:
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(spatial_coords)
        _, idxs = nbrs.kneighbors(spatial_coords)
        idxs = idxs[:, 1:]
        comp = np.zeros((N, n_ct), dtype=np.float32)
        for i, ct in enumerate(ct_types):
            flag       = (cell_type == ct).astype(float)
            comp[:, i] = flag[idxs].mean(axis=1)
        feats.append(comp)

    anchor_dist_arr = None
    if anchor_type is not None:
        anchor_mask = cell_type == anchor_type
        n_anchor    = anchor_mask.sum()
        if n_anchor == 0:
            log.warning(f"  anchor_type '{anchor_type}' not found — skipping anchor features")
        else:
            actual_k = min(anchor_k, n_anchor)
            nbrs_a   = NearestNeighbors(n_neighbors=actual_k).fit(
                spatial_coords[anchor_mask]
            )
            d_a, _ = nbrs_a.kneighbors(spatial_coords)
            anchor_dist_arr = d_a.mean(axis=1).astype(np.float32)

            if rbf_sigma is None:
                rbf_sigma = float(np.median(anchor_dist_arr))
            anchor_feat = np.exp(-d_a / (rbf_sigma + 1e-8)).astype(np.float32)
            feats.append(anchor_feat)

    spat_X = np.concatenate(feats, axis=1)
    log.info(f"  Spatial features: {spat_X.shape}  "
             f"({len(ks)}×{n_ct} comp"
             + (f" + {anchor_k} anchor RBF" if anchor_dist_arr is not None else "")
             + ")")
    return spat_X, anchor_dist_arr


# ──────────────────────────────────────────────────────────────────
# Feather loading
# ──────────────────────────────────────────────────────────────────

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
    return path.stem.replace("_betadata", ""), mod_idx, betas


def build_vocab_parallel(paths: list[Path], n_workers: int = 16) -> dict[str, int]:
    all_mods: set[str] = set()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for cols in ex.map(_get_schema, paths):
            all_mods.update(c for c in cols if c.startswith("beta_"))
    return {name: i for i, name in enumerate(sorted(all_mods))}


def load_feathers(
    paths:    list[Path],
    cell_ids: list[str],
    mod_vocab: dict[str, int],
    n_workers: int = 16,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns
    -------
    beta_X       : [N, G×M]  full signed beta matrix (one block per gene)
    gene_activity: [N, G]    mean|β| per gene (used as reconstruction target)
    gene_names   : list[str] sorted gene names
    """
    n_cells = len(cell_ids)
    n_mods  = len(mod_vocab)
    results: list[tuple[str, np.ndarray, np.ndarray]] = []

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_load_one, p, cell_ids, mod_vocab): p for p in paths}
        for fut in as_completed(futs):
            res = fut.result()
            if res is None:
                continue
            results.append(res)

    results.sort(key=lambda x: x[0])
    gene_names = [r[0] for r in results]

    blocks, activity_cols = [], []
    for _, mod_idx, betas in results:
        block = np.zeros((n_cells, n_mods), dtype=np.float32)
        block[:, mod_idx] = betas
        blocks.append(block)
        activity_cols.append(np.abs(betas).mean(axis=1))

    beta_X        = np.concatenate(blocks, axis=1)
    gene_activity = np.stack(activity_cols, axis=1)
    log.info(f"  beta_X [N, G×M]: {beta_X.shape}   gene_activity [N, G]: {gene_activity.shape}")
    return beta_X, gene_activity, gene_names


# ──────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────

def evaluate(
    z:           np.ndarray,
    true_labels: np.ndarray | None,
    resolutions: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30),
) -> dict:
    best = {"ari": -1.0, "nmi": 0.0, "res": None, "n": 0, "labels": None}
    for res in resolutions:
        r    = cluster_embeddings(z, resolutions=[res])
        pred = r[res].astype(str)
        n    = len(set(pred))
        if true_labels is not None:
            ari = adjusted_rand_score(true_labels, pred)
            nmi = normalized_mutual_info_score(true_labels, pred, average_method="arithmetic")
        else:
            ari = float(n)   # no ground truth: pick resolution by cluster count heuristic
            nmi = 0.0
        if ari > best["ari"]:
            best = {"ari": ari, "nmi": nmi, "res": res, "n": n, "labels": pred}
    return best


# ──────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────

_PAL = ["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4",
        "#f032e6","#bfef45","#469990","#dcbeff","#9a6324","#800000",
        "#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9","#ffe119",
        "#4e9ddb","#c0a040","#e6beff","#9A9900","#A77500"]


def _pal(labels):
    unique = sorted(set(labels), key=lambda x: (int(x) if str(x).isdigit() else 999, str(x)))
    cm = {k: _PAL[i % len(_PAL)] for i, k in enumerate(unique)}
    return cm, [cm[l] for l in labels]


def _handles(cm):
    return [plt.Line2D([0],[0], marker="o", color="w",
                        markerfacecolor=v, markersize=7, label=str(k))
            for k, v in cm.items()]


def make_plots(
    spatial_coords: np.ndarray,
    umap_coords:    np.ndarray | None,
    z:              np.ndarray,
    niche_labels:   np.ndarray,
    cell_type:      np.ndarray,
    anchor_dist:    np.ndarray | None,
    anchor_type:    str | None,
    focus_type:     str | None,
    ref_labels:     np.ndarray | None,
    ref_col:        str,
    best_meta:      dict,
    out_dir:        str,
    sample_name:    str,
) -> None:
    out = Path(out_dir)
    ari, nmi = best_meta.get("ari", 0), best_meta.get("nmi", 0)

    niche_cm, niche_col = _pal(niche_labels)
    ct_cm,    ct_col    = _pal(cell_type)

    # ── 1. Spatial ────────────────────────────────────────────────
    n_cols = 2 + (ref_labels is not None) + (anchor_dist is not None)
    fig, axes = plt.subplots(1, n_cols, figsize=(10 * n_cols, 9))
    ax_list = [(axes[0], niche_col, niche_cm,
                f"Functional microniches ({best_meta['n']})\nARI={ari:.3f} NMI={nmi:.3f}"),
               (axes[1], ct_col,    ct_cm,    f"Cell type ({ref_col})")]
    if ref_labels is not None:
        ref_cm, ref_col_c = _pal(ref_labels)
        ax_list.append((axes[2], ref_col_c, ref_cm, ref_col))
    for ax, colors, cm, title in ax_list:
        ax.scatter(spatial_coords[:,0], spatial_coords[:,1],
                   c=colors, s=5, alpha=0.85, rasterized=True)
        ax.legend(handles=_handles(cm), bbox_to_anchor=(1.01,1), loc="upper left",
                  fontsize=7, frameon=True, ncol=max(1, len(cm)//15))
        ax.set_title(title, fontsize=11); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal"); ax.invert_yaxis()

    if anchor_dist is not None and n_cols > (2 + (ref_labels is not None)):
        ax_a = axes[-1]
        dist_norm = (anchor_dist - anchor_dist.min()) / (anchor_dist.ptp() + 1e-6)
        sc = ax_a.scatter(spatial_coords[:,0], spatial_coords[:,1],
                           c=dist_norm, cmap="RdYlBu_r", s=5, alpha=0.9, rasterized=True)
        plt.colorbar(sc, ax=ax_a, label=f"Proximity to {anchor_type}")
        ax_a.set_title(f"{anchor_type} proximity (red=close)", fontsize=11)
        ax_a.set_aspect("equal"); ax_a.invert_yaxis()
        ax_a.set_xlabel("x"); ax_a.set_ylabel("y")

    plt.suptitle(f"{sample_name} — Functional Microniches", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(out / "spatial.png"), dpi=180, bbox_inches="tight")
    plt.close()
    log.info("  Saved spatial.png")

    # ── 2. UMAP of model embeddings ────────────────────────────────
    log.info("  Computing model UMAP …")
    import umap as umap_lib
    model_umap = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(z)
    np.save(str(out / "umap_coords.npy"), model_umap)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, colors, cm, title in [
        (axes[0], niche_col, niche_cm, f"Functional niches (ARI={ari:.3f})"),
        (axes[1], ct_col,    ct_cm,    "Cell type"),
    ]:
        ax.scatter(model_umap[:,0], model_umap[:,1], c=colors, s=4, alpha=0.7, rasterized=True)
        ax.legend(handles=_handles(cm), bbox_to_anchor=(1.01,1), loc="upper left",
                  fontsize=7, ncol=max(1, len(cm)//15))
        ax.set_title(title, fontsize=11); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    plt.suptitle(f"{sample_name} — Model UMAP", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out / "umap_model.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # ── 3. Original UMAP overlay (if available) ────────────────────
    if umap_coords is not None:
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for ax, colors, cm, title in [
            (axes[0], niche_col, niche_cm, f"Functional niches on original UMAP"),
            (axes[1], ct_col,    ct_cm,    "Cell type on original UMAP"),
        ]:
            ax.scatter(umap_coords[:,0], umap_coords[:,1],
                       c=colors, s=4, alpha=0.7, rasterized=True)
            ax.legend(handles=_handles(cm), bbox_to_anchor=(1.01,1), loc="upper left",
                      fontsize=7, ncol=max(1, len(cm)//15))
            ax.set_title(title, fontsize=11); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        plt.suptitle(f"{sample_name} — Original UMAP", fontsize=12)
        plt.tight_layout()
        plt.savefig(str(out / "umap_original.png"), dpi=180, bbox_inches="tight")
        plt.close()
        log.info("  Saved umap_original.png")
    log.info("  Saved umap_model.png")

    # ── 4. Composition heatmap ──────────────────────────────────────
    comp_col = ref_labels if ref_labels is not None else cell_type
    comp_name = ref_col
    comp = pd.crosstab(
        pd.Series(niche_labels, name="niche"),
        pd.Series(comp_col,     name=comp_name),
        normalize="index",
    )
    try:
        comp = comp.reindex(sorted(comp.index, key=int))
    except (ValueError, TypeError):
        pass

    fig, ax = plt.subplots(figsize=(max(12, len(comp.columns)*0.65),
                                     max(5,  len(comp)*0.45)))
    sns.heatmap(comp, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
                linewidths=0.4, cbar_kws={"label": "fraction"})
    ax.set_title(f"Niche composition by {comp_name}  (ARI={ari:.4f})", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(out / "composition.png"), dpi=180, bbox_inches="tight")
    plt.close()
    log.info("  Saved composition.png")

    # ── 5. Anchor proximity per niche ──────────────────────────────
    if anchor_dist is not None and anchor_type is not None:
        focus_mask = (cell_type == focus_type) if focus_type else np.ones(len(cell_type), bool)
        niche_order = sorted(set(niche_labels), key=lambda x: int(x) if str(x).isdigit() else 999)
        fig, ax = plt.subplots(figsize=(max(8, len(niche_order)*0.8), 5))
        data = [anchor_dist[(niche_labels == n) & focus_mask] for n in niche_order]
        data = [d for d in data if len(d) > 0]
        labels_bp = [n for n, d in zip(niche_order, [anchor_dist[(niche_labels==n)&focus_mask]
                                                       for n in niche_order]) if len(d) > 0]
        ax.boxplot(data, labels=labels_bp)
        ax.set_xlabel("Functional niche")
        ax.set_ylabel(f"Distance to nearest {anchor_type}")
        ax.set_title(f"{focus_type or 'all'} cells: {anchor_type} proximity per niche", fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(str(out / "anchor_proximity_per_niche.png"), dpi=150, bbox_inches="tight")
        plt.close()
        log.info("  Saved anchor_proximity_per_niche.png")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Functional Spatial Niche Embeddings — generalised runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    parser.add_argument("--h5ad",        required=True, help="AnnData .h5ad file")
    parser.add_argument("--feather-dir", required=True, help="Directory of *_betadata.feather files")
    parser.add_argument("--out-dir",     required=True, help="Output directory")

    # Cell type / label config
    parser.add_argument("--cell-type-col",  default="cell_type",
                        help="obs column with coarse cell type labels")
    parser.add_argument("--ref-label-col",  default=None,
                        help="obs column with finer reference labels for ARI evaluation "
                             "(e.g. 'cell_type_2'); if omitted, uses cell-type-col")
    parser.add_argument("--focus-type",     default=None,
                        help="Cell type to apply the anchor-proximity ranking loss on "
                             "(e.g. 'B_germinal_center'). If omitted, ranking loss is skipped.")
    parser.add_argument("--anchor-type",    default=None,
                        help="Reference cell type for proximity signal "
                             "(e.g. 'T_follicular_helper'). If omitted, proximity loss is skipped.")

    # Gene selection
    parser.add_argument("--genes",     nargs="+", default=None,
                        help="Explicit gene list. If omitted, genes are chosen automatically.")
    parser.add_argument("--n-genes",   type=int, default=30,
                        help="Number of marker genes to auto-select")
    parser.add_argument("--de-method", default="wilcoxon",
                        choices=["wilcoxon", "t-test", "logreg"],
                        help="Differential expression method for gene selection")

    # Model hyperparameters
    parser.add_argument("--hidden-dim",  type=int,   default=64)
    parser.add_argument("--mlp-layers",  type=int,   default=2)
    parser.add_argument("--gcn-layers",  type=int,   default=2)
    parser.add_argument("--epochs",      type=int,   default=800)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--spatial-k",   type=int,   default=6,
                        help="kNN for the spatial graph fed to the GCN")
    parser.add_argument("--leiden-res",  type=float, default=None,
                        help="Fixed Leiden resolution. If omitted, swept over 0.15–0.30.")

    # Loss weights
    parser.add_argument("--w-triplet",  type=float, default=1.0)
    parser.add_argument("--w-rec",      type=float, default=0.05)
    parser.add_argument("--w-smooth",   type=float, default=0.3)
    parser.add_argument("--w-anchor",   type=float, default=2.0,
                        help="Weight for the anchor-proximity ranking loss")
    parser.add_argument("--w-nbr",      type=float, default=0.5,
                        help="Weight for the neighbourhood composition regression loss")

    # Runtime
    parser.add_argument("--n-workers",  type=int, default=16)
    parser.add_argument("--device",     default="auto")
    parser.add_argument("--name",       default=None,
                        help="Sample name for plot titles (defaults to h5ad filename stem)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    warnings.filterwarnings("ignore")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sample_name = args.name or Path(args.h5ad).stem

    # ── 1. Load h5ad ─────────────────────────────────────────────
    log.info(f"Loading {args.h5ad} …")
    adata          = anndata.read_h5ad(args.h5ad)
    cell_ids       = list(adata.obs_names)
    cell_type      = adata.obs[args.cell_type_col].values.astype(str)

    ref_col_name   = args.ref_label_col or args.cell_type_col
    ref_labels     = adata.obs[ref_col_name].values.astype(str) \
                     if ref_col_name in adata.obs.columns else None

    # Spatial coordinates
    spatial_key = next((k for k in ("spatial", "spatial_unscaled", "X_umap")
                        if k in adata.obsm), None)
    if spatial_key is None:
        raise ValueError("No spatial coordinates found in adata.obsm. "
                         "Expected 'spatial', 'spatial_unscaled', or 'X_umap'.")
    spatial_coords = adata.obsm[spatial_key].astype(np.float32)
    if spatial_coords.shape[1] > 2:
        spatial_coords = spatial_coords[:, :2]

    umap_coords = adata.obsm.get("X_umap")
    if umap_coords is not None:
        umap_coords = umap_coords.astype(np.float32)

    focus_mask = (cell_type == args.focus_type) if args.focus_type else np.ones(len(cell_type), bool)
    log.info(f"  {len(cell_ids)} cells  ·  {len(set(cell_type))} cell types")
    if args.focus_type:
        log.info(f"  Focus: {focus_mask.sum()} '{args.focus_type}' cells")
    if args.anchor_type:
        log.info(f"  Anchor: {(cell_type==args.anchor_type).sum()} '{args.anchor_type}' cells")

    # ── 2. Gene selection ─────────────────────────────────────────
    feather_dir = args.feather_dir
    if args.genes:
        gene_set = set(args.genes)
        all_feathers = sorted(Path(feather_dir).glob("*_betadata.feather"))
        feather_paths = [p for p in all_feathers
                         if p.stem.replace("_betadata","") in gene_set]
        found = sorted({p.stem.replace("_betadata","") for p in feather_paths})
        missing = gene_set - set(found)
        if missing:
            log.warning(f"  Not found in betadata: {sorted(missing)}")
        log.info(f"  Using {len(feather_paths)} specified genes: {found}")
    else:
        log.info(f"  Auto-selecting top-{args.n_genes} marker genes …")
        selected = select_marker_genes(
            adata, feather_dir, args.cell_type_col,
            n_genes=args.n_genes, method=args.de_method,
        )
        all_feathers  = sorted(Path(feather_dir).glob("*_betadata.feather"))
        feather_paths = [p for p in all_feathers
                         if p.stem.replace("_betadata","") in set(selected)]

    if not feather_paths:
        raise FileNotFoundError(
            "No betadata feathers matched the selected genes. "
            "Check --feather-dir and gene names."
        )

    # ── 3. Load feathers ──────────────────────────────────────────
    log.info("Loading betadata …")
    mod_vocab = build_vocab_parallel(feather_paths, n_workers=args.n_workers)
    beta_X, gene_activity, gene_names = load_feathers(
        feather_paths, cell_ids, mod_vocab, n_workers=args.n_workers,
    )
    log.info(f"  Genes used: {sorted(gene_names)}")

    # Save gene list
    with open(out / "selected_genes.json", "w") as f:
        json.dump({"genes": sorted(gene_names)}, f, indent=2)

    # ── 4. Spatial features ───────────────────────────────────────
    log.info("Building spatial features …")
    spat_X, anchor_dist = build_spatial_features_general(
        spatial_coords, cell_type,
        anchor_type=args.anchor_type,
        ks=(10, 30, 60),
        anchor_k=10,
        rbf_sigma=None,   # auto-compute from data
    )

    # ── 5. Spatial graph ──────────────────────────────────────────
    edge_index, edge_weight = build_spatial_graph(spatial_coords, k=args.spatial_k)

    # ── 6. Train ──────────────────────────────────────────────────
    log.info("Training SpatialFunctionalModel …")
    z = train_functional(
        beta_X      = beta_X,
        spat_X      = spat_X,
        rec_target  = gene_activity,
        tfh_dist    = anchor_dist if anchor_dist is not None else np.zeros(len(cell_ids), dtype=np.float32),
        gc_mask     = focus_mask,
        edge_index  = edge_index,
        edge_weight = edge_weight,
        cell_ids    = cell_ids,
        output_dir  = str(out),
        hidden_dim  = args.hidden_dim,
        mlp_layers  = args.mlp_layers,
        gcn_layers  = args.gcn_layers,
        epochs      = args.epochs,
        lr          = args.lr,
        w_triplet   = args.w_triplet,
        w_rec       = args.w_rec,
        w_smooth    = args.w_smooth,
        w_tfh_rank  = args.w_anchor if (anchor_dist is not None and focus_mask.any()) else 0.0,
        w_nbr_comp  = args.w_nbr   if (anchor_dist is not None) else 0.0,
        device_str  = args.device,
        log_every   = max(1, args.epochs // 8),
    )

    # ── 7. Cluster ────────────────────────────────────────────────
    log.info("Clustering …")
    resolutions = (args.leiden_res,) if args.leiden_res else (0.15, 0.20, 0.25, 0.30)
    best = evaluate(z, ref_labels, resolutions=resolutions)
    log.info(f"  Best: ARI={best['ari']:.4f}  NMI={best['nmi']:.4f}  "
             f"n={best['n']}  res={best['res']}")

    niche_labels = best["labels"]

    # Sort niches by anchor distance if available (closest anchor = niche 0)
    if anchor_dist is not None:
        means = {n: anchor_dist[niche_labels == n].mean() for n in set(niche_labels)}
        order = sorted(means, key=means.get)
    else:
        # Sort by size (largest first)
        counts = pd.Series(niche_labels).value_counts()
        order  = list(counts.index)
    remap = {old: str(i) for i, old in enumerate(order)}
    niche_labels = np.array([remap[n] for n in niche_labels])

    # ── 8. Save ───────────────────────────────────────────────────
    label_df = pd.DataFrame({
        "CellID":          cell_ids,
        "niche":           niche_labels,
        args.cell_type_col: cell_type,
    })
    if ref_labels is not None and ref_col_name != args.cell_type_col:
        label_df[ref_col_name] = ref_labels
    if anchor_dist is not None:
        label_df["anchor_dist"] = anchor_dist
    label_df.to_parquet(out / "niche_labels.parquet", index=False)
    np.save(str(out / "embeddings.npy"), z)

    # ── 9. Report ─────────────────────────────────────────────────
    if focus_mask.any() and ref_labels is not None:
        comp = pd.crosstab(
            pd.Series(niche_labels[focus_mask], name="niche"),
            pd.Series(ref_labels[focus_mask],   name=ref_col_name),
        )
        log.info(f"\n{args.focus_type or 'All'} cells: niche × {ref_col_name}\n{comp.to_string()}")

    # ── 10. Plots ─────────────────────────────────────────────────
    log.info("Generating plots …")
    make_plots(
        spatial_coords = spatial_coords,
        umap_coords    = umap_coords,
        z              = z,
        niche_labels   = niche_labels,
        cell_type      = cell_type,
        anchor_dist    = anchor_dist,
        anchor_type    = args.anchor_type,
        focus_type     = args.focus_type,
        ref_labels     = ref_labels,
        ref_col        = ref_col_name,
        best_meta      = best,
        out_dir        = str(out),
        sample_name    = sample_name,
    )

    log.info(f"\nAll outputs saved to {out}")
    log.info(f"  niche_labels.parquet  embeddings.npy  selected_genes.json")
    log.info(f"  spatial.png  umap_model.png  composition.png  anchor_proximity_per_niche.png")


if __name__ == "__main__":
    main()
