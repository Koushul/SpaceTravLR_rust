"""
Apply SimpleNicheModel to tonsil betadata and iterate toward maximizing
ARI/NMI against cell_type_2 labels.

Three input representations are evaluated:
  A. gene-activity   [N, G]       — mean|β| per gene per cell
  B. mod-sum-signed  [N, M]       — signed β summed across genes per modulator
  C. mod-abs         [N, M]       — mean|β| per modulator across genes
  D. joint           [N, G+M]     — concat A + C

Each is passed through BetaMLP → SpatialGCN → Leiden.
We grid-search epochs / spatial_k / loss weights and report ARI/NMI at every step.

Usage:
    cd SpaceTravLR_rust
    PYTHONPATH=scripts python3 scripts/functional_niches/run_tonsil.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
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
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from functional_niches.dataset import build_spatial_graph, FunctionalNicheDataset
from functional_niches.simple_model import train_simple, SimpleNicheModel, TripletSpatialLoss
from functional_niches.losses import build_adj_mask, spatial_smoothness_loss
from functional_niches.cluster import cluster_embeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
FEATHER_DIR = "/ix1/ylee/kor11/djishnu_kor11/tonsil_ablation/runs/tf_lr_tfl__full_2"
H5AD_PATH   = "/ix1/ylee/kor11/djishnu_kor11/tonsil_ablation/snrna_human_tonsil.h5ad"
OUT_DIR     = "/tmp/tonsil_niches_v2"
N_WORKERS   = 16


# ------------------------------------------------------------------
# Parallel feather loading
# ------------------------------------------------------------------
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


def load_all_feathers(paths, cell_ids, mod_vocab, n_workers=16):
    """
    Returns four representations accumulated across all genes:
      gene_activity [N, G]  — mean|β| per (cell, gene)
      X_signed      [N, M]  — sum of signed β per (cell, modulator)
      X_abs         [N, M]  — mean|β| per (cell, modulator)
      gene_names    list[str]
    """
    n_cells = len(cell_ids)
    n_mods  = len(mod_vocab)

    gene_activity_list: list[np.ndarray] = []    # one [N] per gene
    gene_name_list:     list[str]        = []
    X_signed  = np.zeros((n_cells, n_mods), dtype=np.float32)
    X_abs_acc = np.zeros((n_cells, n_mods), dtype=np.float32)
    X_abs_cnt = np.zeros(n_mods, dtype=np.float32)

    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_load_one, p, cell_ids, mod_vocab): p for p in paths}
        for fut in as_completed(futs):
            result = fut.result()
            if result is None:
                continue
            gene_name, mod_idx, betas = result
            np.add.at(X_signed,  (slice(None), mod_idx), betas)
            np.add.at(X_abs_acc, (slice(None), mod_idx), np.abs(betas))
            X_abs_cnt[mod_idx] += 1.0
            gene_activity_list.append(np.abs(betas).mean(axis=1))  # [N]
            gene_name_list.append(gene_name)
            done += 1
            if done % 300 == 0:
                log.info(f"  Loaded {done}/{len(paths)} genes …")

    X_abs_cnt = np.maximum(X_abs_cnt, 1.0)
    X_abs   = X_abs_acc / X_abs_cnt[None, :]

    # Stack gene activity into [N, G]
    gene_activity = np.stack(gene_activity_list, axis=1)  # [N, n_genes_loaded]
    return gene_activity, X_signed, X_abs, gene_name_list


# ------------------------------------------------------------------
# Evaluate ARI / NMI at multiple resolutions
# ------------------------------------------------------------------
def evaluate(z: np.ndarray, true_labels: np.ndarray,
             resolutions=(0.15, 0.20, 0.25, 0.30)) -> dict:
    best = {"ari": -1.0, "nmi": 0.0, "res": None, "n": 0}
    rows = []
    for res in resolutions:
        cluster_res = cluster_embeddings(z, resolutions=[res])
        pred = cluster_res[res].astype(str)
        ari  = adjusted_rand_score(true_labels, pred)
        nmi  = normalized_mutual_info_score(true_labels, pred, average_method="arithmetic")
        n    = len(set(pred))
        rows.append({"res": res, "n_clusters": n, "ari": ari, "nmi": nmi})
        if ari > best["ari"]:
            best = {"ari": ari, "nmi": nmi, "res": res, "n": n}
    return best, rows


# ------------------------------------------------------------------
# Train one config, return ARI/NMI
# ------------------------------------------------------------------
def run_one(
    X_in: np.ndarray,
    rec_target: np.ndarray,
    cell_ids: list[str],
    mod_vocab: dict[str, int],
    gene_names: list[str],
    edge_index,
    edge_weight,
    true_labels: np.ndarray,
    tag: str,
    hidden_dim: int = 64,
    mlp_layers: int = 2,
    gcn_layers: int = 2,
    epochs: int = 700,
    lr: float = 1e-3,
    alpha: float = 0.05,
    beta_loss: float = 0.3,
    spatial_k: int = 6,
    resolutions: tuple = (0.15, 0.20, 0.25, 0.30),
    out_dir: str = OUT_DIR,
) -> tuple[dict, np.ndarray]:
    log.info(f"\n{'='*60}")
    log.info(f"  [{tag}]  X={X_in.shape}  hidden={hidden_dim}  epochs={epochs}  "
             f"k={spatial_k}  alpha={alpha}  beta={beta_loss}")
    log.info(f"{'='*60}")

    dataset = FunctionalNicheDataset(
        cell_ids=cell_ids,
        gene_betas=[],
        edge_index=edge_index,
        edge_weight=edge_weight,
        mod_vocab=mod_vocab,
        gene_names=gene_names,
        rec_target=torch.from_numpy(rec_target).float(),
        _beta_matrix=torch.from_numpy(X_in).float(),
    )

    out = Path(out_dir) / tag
    z = train_simple(
        dataset=dataset,
        output_dir=str(out),
        hidden_dim=hidden_dim,
        mlp_layers=mlp_layers,
        gcn_layers=gcn_layers,
        epochs=epochs,
        lr=lr,
        alpha=alpha,
        beta=beta_loss,
        device_str="auto",
        log_every=200,
    )

    best, rows = evaluate(z, true_labels, resolutions=resolutions)
    log.info(f"  Best → ARI={best['ari']:.4f}  NMI={best['nmi']:.4f}  "
             f"n={best['n']}  res={best['res']}")
    return best, z, rows


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
    umap_coords    = adata.obsm["X_umap"].astype(np.float32)
    cell_type      = adata.obs["cell_type"].values.astype(str)
    true_labels    = adata.obs["cell_type_2"].values.astype(str)
    n_true         = len(set(true_labels))
    log.info(f"  {len(cell_ids)} cells  ·  {n_true} cell_type_2 labels")

    # 2. Vocab + data load
    # Only load the GC-relevant target genes requested by the user.
    # IL21R and MIF are absent from this run; IL7R (the receptor) is included.
    TARGET_GENES = {"BCL6", "IL21", "IL7", "IL7R", "PAX5", "AICDA", "CXCR4"}
    all_feathers  = sorted(Path(FEATHER_DIR).glob("*_betadata.feather"))
    feather_paths = [
        p for p in all_feathers
        if p.stem.replace("_betadata", "") in TARGET_GENES
    ]
    found = {p.stem.replace("_betadata", "") for p in feather_paths}
    missing = TARGET_GENES - found
    if missing:
        log.warning(f"  Genes not found in run directory: {missing}")
    log.info(f"Using {len(feather_paths)} target feathers: {sorted(found)}")
    mod_vocab = build_vocab_parallel(feather_paths, n_workers=N_WORKERS)
    n_mods    = len(mod_vocab)
    log.info(f"  {n_mods} unique modulators. Loading data …")

    gene_activity, X_signed, X_abs, gene_names = load_all_feathers(
        feather_paths, cell_ids, mod_vocab, n_workers=N_WORKERS
    )
    n_genes = len(gene_names)
    log.info(f"  gene_activity {gene_activity.shape}  X_signed {X_signed.shape}  X_abs {X_abs.shape}")

    # 3. Spatial graph — tested at k=6 and k=15
    edges_k6,  weights_k6  = build_spatial_graph(spatial_coords, k=6)
    edges_k15, weights_k15 = build_spatial_graph(spatial_coords, k=15)

    # ------------------------------------------------------------------
    # Systematic iteration
    # Track best config across all runs.
    # ------------------------------------------------------------------
    # rec_target for each representation: same shape as X so decoder always matches
    # We use mean|β| per modulator for mod-based inputs, and gene_activity itself
    # for the gene-based input (the model reconstructs its own input as a regulariser)
    rec_mods_np = X_abs.astype(np.float32)              # [N, 777]
    rec_gene_np = gene_activity.astype(np.float32)      # [N, 7]
    rec_joint_np = np.concatenate([rec_gene_np, rec_mods_np], axis=1)  # [N, 784]

    results: list[dict] = []
    best_overall = {"ari": -1.0}
    best_z = None

    def _run(tag, X_np, rec_np, ei, ew, **kw):
        nonlocal best_overall, best_z
        b, z, rows = run_one(
            X_in=X_np, rec_target=rec_np,
            cell_ids=cell_ids, mod_vocab=mod_vocab, gene_names=gene_names,
            edge_index=ei, edge_weight=ew,
            true_labels=true_labels, tag=tag, **kw,
        )
        entry = {"tag": tag, **b, **kw}
        results.append(entry)
        log.info(f"  [RESULT] {tag}: ARI={b['ari']:.4f}  NMI={b['nmi']:.4f}")
        if b["ari"] > best_overall["ari"]:
            best_overall = {"tag": tag, **b}
            best_z = z
        with open(Path(OUT_DIR) / "iteration_log.json", "w") as f:
            json.dump(results, f, indent=2)
        return b, z

    # ------------------------------------------------------------------
    # Round 1: representations at baseline config
    # ------------------------------------------------------------------
    log.info("\n\n--- ROUND 1: Input representation comparison ---")
    base = dict(epochs=700, lr=1e-3, alpha=0.05, beta_loss=0.3,
                hidden_dim=64, spatial_k=6, resolutions=(0.15, 0.20, 0.25, 0.30))

    X_joint = np.concatenate([gene_activity.astype(np.float32),
                               X_abs.astype(np.float32)], axis=1)

    # A: gene activity [N, G] — which GC-program genes are highly regulated per cell
    _run("A_gene_activity",
         gene_activity.astype(np.float32), rec_gene_np,
         edges_k6, weights_k6, **base)

    # B: signed-beta per modulator [N, M] — direction of regulation per TF/LR
    _run("B_mod_signed",
         X_signed.astype(np.float32), rec_mods_np,
         edges_k6, weights_k6, **base)

    # C: abs-beta per modulator [N, M] — magnitude of regulation per TF/LR
    _run("C_mod_abs",
         X_abs.astype(np.float32), rec_mods_np,
         edges_k6, weights_k6, **base)

    # D: gene activity + mod_abs concatenated [N, G+M]
    _run("D_gene_plus_abs",
         X_joint, rec_joint_np, edges_k6, weights_k6, **base)

    log.info("\nRound 1 summary:")
    for r in results:
        log.info(f"  {r['tag']:<25} ARI={r['ari']:.4f}  NMI={r['nmi']:.4f}")

    # ------------------------------------------------------------------
    # Round 2: best representation, sweep spatial k and loss weights
    # ------------------------------------------------------------------
    best_r1 = max(results, key=lambda x: x["ari"])
    log.info(f"\n--- ROUND 2: Best repr = {best_r1['tag']}, sweep k and loss ---")

    # Pick best input from round 1
    repr_map = {
        "A_gene_activity": (gene_activity.astype(np.float32), rec_gene_np),
        "B_mod_signed":    (X_signed.astype(np.float32), rec_mods_np),
        "C_mod_abs":       (X_abs.astype(np.float32), rec_mods_np),
        "D_gene_plus_abs": (X_joint, rec_joint_np),
    }
    best_X, best_rec = repr_map[best_r1["tag"]]

    configs_r2 = [
        dict(spatial_k=6,  alpha=0.05, beta_loss=0.3),
        dict(spatial_k=6,  alpha=0.05, beta_loss=0.5),
        dict(spatial_k=6,  alpha=0.1,  beta_loss=0.1),
        dict(spatial_k=15, alpha=0.05, beta_loss=0.3),
        dict(spatial_k=15, alpha=0.05, beta_loss=0.5),
        dict(spatial_k=15, alpha=0.1,  beta_loss=0.1),
    ]
    for cfg in configs_r2:
        k = cfg.pop("spatial_k")
        ei, ew = (edges_k15, weights_k15) if k == 15 else (edges_k6, weights_k6)
        tag = f"R2_{best_r1['tag']}_k{k}_a{cfg['alpha']}_b{cfg['beta_loss']}"
        _run(tag, best_X, best_rec, ei, ew,
             epochs=700, lr=1e-3, hidden_dim=64,
             resolutions=(0.15, 0.20, 0.25, 0.30), spatial_k=k, **cfg)

    # ------------------------------------------------------------------
    # Round 3: best config so far, increase capacity + epochs
    # ------------------------------------------------------------------
    best_r2 = max(results, key=lambda x: x["ari"])
    log.info(f"\n--- ROUND 3: Scale up best config = {best_r2['tag']} ---")
    best_X2, best_rec2 = repr_map.get(
        best_r1["tag"], (best_X, best_rec)
    )
    k3 = best_r2.get("spatial_k", 6)
    ei3, ew3 = (edges_k15, weights_k15) if k3 == 15 else (edges_k6, weights_k6)
    a3 = best_r2.get("alpha", 0.05)
    b3 = best_r2.get("beta_loss", 0.3)

    for hd, ml, ep in [(128, 3, 1000), (64, 2, 1500)]:
        tag = f"R3_hd{hd}_ml{ml}_ep{ep}"
        _run(tag, best_X2, best_rec2, ei3, ew3,
             epochs=ep, lr=1e-3, hidden_dim=hd, mlp_layers=ml, gcn_layers=2,
             alpha=a3, beta_loss=b3, spatial_k=k3,
             resolutions=(0.15, 0.20, 0.25, 0.30))

    # ------------------------------------------------------------------
    # Final summary and plots
    # ------------------------------------------------------------------
    log.info("\n\n" + "="*70)
    log.info("FULL ITERATION SUMMARY")
    log.info("="*70)
    log.info(f"{'Tag':<40} {'ARI':>8} {'NMI':>8} {'Clusters':>9} {'Res':>6}")
    log.info("-"*70)
    for r in sorted(results, key=lambda x: -x["ari"]):
        log.info(f"{r['tag']:<40} {r['ari']:>8.4f} {r['nmi']:>8.4f} {r['n']:>9} {r['res']:>6}")
    log.info("="*70)

    best_final = max(results, key=lambda x: x["ari"])
    log.info(f"\nBest: {best_final['tag']}  ARI={best_final['ari']:.4f}  NMI={best_final['nmi']:.4f}")

    # Recluster best_z at the best resolution
    if best_z is not None:
        best_res = best_final["res"]
        cr  = cluster_embeddings(best_z, resolutions=[best_res])
        niche_labels = cr[best_res].astype(str)

        # Sort niches by size
        size_order = (pd.Series(niche_labels).value_counts()
                      .sort_values(ascending=False).index.tolist())
        remap = {o: str(i) for i, o in enumerate(size_order)}
        niche_labels = np.array([remap[n] for n in niche_labels])

        label_df = pd.DataFrame({
            "CellID": cell_ids, "niche": niche_labels,
            "cell_type": cell_type, "cell_type_2": true_labels,
        })
        label_df.to_parquet(Path(OUT_DIR) / "best_niche_labels.parquet", index=False)
        np.save(str(Path(OUT_DIR) / "best_embeddings.npy"), best_z)

        _plot_all(spatial_coords, umap_coords, best_z, niche_labels, cell_type, true_labels,
                  best_final)

    # Save iteration log
    with open(Path(OUT_DIR) / "iteration_log.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nAll results saved to {OUT_DIR}")


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

_PAL = (["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4",
          "#f032e6","#bfef45","#469990","#dcbeff","#9a6324","#800000",
          "#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9","#ffe119",
          "#4e9ddb","#c0a040"])


def _pal(labels):
    unique = sorted(set(labels), key=lambda x: (int(x) if x.isdigit() else 999, x))
    cm = {k: _PAL[i % len(_PAL)] for i, k in enumerate(unique)}
    return cm, [cm[l] for l in labels]


def _handles(cm):
    return [plt.Line2D([0],[0],marker="o",color="w",
                        markerfacecolor=v,markersize=7,label=str(k))
            for k,v in cm.items()]


def _plot_all(spatial_coords, umap_coords, z, niche_labels, cell_type, cell_type_2, best):
    n_niches = len(set(niche_labels))
    ari = best["ari"]; nmi = best["nmi"]
    tag = best["tag"]

    niche_cm, niche_col = _pal(niche_labels)
    ct_cm,    ct_col    = _pal(cell_type)
    ct2_cm,   ct2_col   = _pal(cell_type_2)

    # ── Spatial: niches vs cell_type_2 ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(30, 9))
    for ax, colors, cm, title in [
        (axes[0], niche_col, niche_cm,
         f"Functional microniches ({n_niches})\nARI={ari:.3f}  NMI={nmi:.3f}"),
        (axes[1], ct2_col,  ct2_cm,   "cell_type_2 (reference)"),
        (axes[2], ct_col,   ct_cm,    "cell_type (coarse)"),
    ]:
        ax.scatter(spatial_coords[:,0], spatial_coords[:,1],
                   c=colors, s=5, alpha=0.85, rasterized=True)
        ax.legend(handles=_handles(cm), bbox_to_anchor=(1.01,1), loc="upper left",
                  fontsize=7, frameon=True, ncol=max(1, len(cm)//15))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal"); ax.invert_yaxis()

    plt.suptitle(f"Tonsil functional microniches — {tag}\n"
                 f"ARI vs cell_type_2 = {ari:.4f}  NMI = {nmi:.4f}", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(Path(OUT_DIR) / "spatial_best.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # ── UMAP (model embeddings): niches vs cell_type_2 ────────────
    log.info("Computing UMAP of model embeddings …")
    import umap as umap_lib
    coords = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(z)

    fig, axes = plt.subplots(1, 3, figsize=(28, 8))
    for ax, colors, cm, title in [
        (axes[0], niche_col, niche_cm, "Functional niches"),
        (axes[1], ct2_col,   ct2_cm,   "cell_type_2"),
        (axes[2], ct_col,    ct_cm,    "cell_type"),
    ]:
        ax.scatter(coords[:,0], coords[:,1], c=colors, s=4, alpha=0.7, rasterized=True)
        ax.legend(handles=_handles(cm), bbox_to_anchor=(1.01,1), loc="upper left",
                  fontsize=7, frameon=True, ncol=max(1, len(cm)//15))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    plt.suptitle(f"UMAP of SimpleNicheModel embeddings — {tag}", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(Path(OUT_DIR) / "umap_best.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # ── UMAP coloured on original scRNA UMAP ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, colors, cm, title in [
        (axes[0], niche_col, niche_cm, f"Functional niches on scRNA UMAP\n(ARI={ari:.3f}, NMI={nmi:.3f})"),
        (axes[1], ct2_col,   ct2_cm,   "cell_type_2 on scRNA UMAP"),
    ]:
        ax.scatter(umap_coords[:,0], umap_coords[:,1],
                   c=colors, s=4, alpha=0.7, rasterized=True)
        ax.legend(handles=_handles(cm), bbox_to_anchor=(1.01,1), loc="upper left",
                  fontsize=7, frameon=True, ncol=max(1, len(cm)//15))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(str(Path(OUT_DIR) / "scrna_umap_best.png"), dpi=180, bbox_inches="tight")
    plt.close()

    # ── Composition heatmap ────────────────────────────────────────
    comp = pd.crosstab(
        pd.Series(niche_labels, name="niche"),
        pd.Series(cell_type_2,  name="cell_type_2"),
        normalize="index",
    )
    comp = comp.reindex(sorted(comp.index, key=int))

    fig, ax = plt.subplots(figsize=(max(12, len(comp.columns)*0.7),
                                     max(6,  len(comp)*0.5)))
    sns.heatmap(comp, ax=ax, cmap="YlOrRd", annot=True, fmt=".2f",
                linewidths=0.4, cbar_kws={"label": "fraction of niche"})
    ax.set_title(f"Niche composition  (ARI={ari:.4f}  NMI={nmi:.4f})", fontsize=12)
    ax.set_xlabel("cell_type_2"); ax.set_ylabel("Functional niche")
    plt.tight_layout()
    plt.savefig(str(Path(OUT_DIR) / "composition_heatmap.png"), dpi=180, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved composition_heatmap.png")

    # ── Iteration progress plot ────────────────────────────────────
    try:
        with open(Path(OUT_DIR) / "iteration_log.json") as f:
            all_results = json.load(f)
        tags = [r["tag"] for r in all_results]
        aris = [r["ari"] for r in all_results]
        nmis = [r["nmi"] for r in all_results]

        fig, ax = plt.subplots(figsize=(max(10, len(tags)*0.6), 5))
        x = np.arange(len(tags))
        ax.bar(x, aris, label="ARI", alpha=0.8, color="#4363d8")
        ax.bar(x, nmis, label="NMI", alpha=0.5, color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.axhline(max(aris), color="#4363d8", ls="--", lw=0.8, alpha=0.6)
        ax.legend(); ax.set_title("ARI / NMI across all iterations")
        ax.set_ylabel("Score")
        plt.tight_layout()
        plt.savefig(str(Path(OUT_DIR) / "iteration_progress.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
