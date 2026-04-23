"""
Benchmark SimpleNicheModel vs PCA baselines on synthetic data with known niches.

Metrics
-------
- ARI  (Adjusted Rand Index)    — clustering agreement, chance-adjusted
- NMI  (Normalized Mutual Info) — information-theoretic agreement

Baselines
---------
1. PCA on flat signed-beta matrix  [N, G × M]   — retains sign info
2. PCA + 2-hop spatial smoothing   — approximates the GCN step
3. PCA on mean|β| rec_target       — loses sign info; confounded by sign-coded niches

SimpleNicheModel:
  precompute [N, G × M]  →  MLP  →  2-layer GCN  →  z ∈ ℝ^D
  (all three objectives: triplet spatial contrastive, MSE reconstruction, smoothness)

Usage
-----
    python -m functional_niches.benchmark
    python -m functional_niches.benchmark --multi --output-dir /tmp/bench
    python -m functional_niches.benchmark --n-cells 2000 --epochs 500
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .synth import make_synthetic_dataset
from .simple_model import train_simple
from .cluster import cluster_embeddings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_betas(gene_betas, n_cells: int, n_mods_total: int) -> np.ndarray:
    """[N, G × n_mods_total] signed beta matrix (numpy, for sklearn PCA)."""
    parts = []
    for gb in gene_betas:
        mat = np.zeros((n_cells, n_mods_total), dtype=np.float32)
        mat[:, gb.mod_indices[0].numpy()] = gb.beta_values.numpy()
        parts.append(mat)
    return np.concatenate(parts, axis=1)


def _spatial_smooth(z: np.ndarray, edge_index, n_hops: int = 2) -> np.ndarray:
    """Row-normalised mean-neighbour smoothing (n_hops iterations)."""
    import scipy.sparse as sp
    n = z.shape[0]
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    adj = sp.csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n))
    deg_inv = np.where(adj.sum(1).A1 > 0, 1.0 / adj.sum(1).A1, 0.0)
    D_inv = sp.diags(deg_inv)
    A = D_inv @ adj
    out = z.copy()
    for _ in range(n_hops):
        out = A @ out
    return out


def best_ari_nmi(
    z: np.ndarray,
    true_labels: np.ndarray,
    resolutions: list[float] = [0.3, 0.5, 1.0, 1.5],
) -> dict:
    cluster_results = cluster_embeddings(z, resolutions=resolutions)
    best = {"ari": -1.0, "nmi": 0.0, "resolution": None, "n_clusters": 0}
    per_res = []
    for r, labels in cluster_results.items():
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels, average_method="arithmetic")
        n_cl = len(set(labels))
        per_res.append({"resolution": r, "ari": ari, "nmi": nmi, "n_clusters": n_cl})
        if ari > best["ari"]:
            best = {"ari": ari, "nmi": nmi, "resolution": r, "n_clusters": n_cl}
    return {"best": best, "per_resolution": per_res}


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(
    n_cells: int = 1000,
    n_genes: int = 8,
    n_niches: int = 5,
    n_mods_shared: int = 200,
    n_mods_gene: int = 20,
    n_active_mods: int = 20,
    beta_signal: float = 1.5,
    beta_noise: float = 0.1,
    sparsity: float = 0.65,
    cell_noise_scale: float = 0.0,
    gene_specific_programs: bool = False,
    hidden_dim: int = 64,
    mlp_layers: int = 2,
    gcn_layers: int = 2,
    epochs: int = 500,
    lr: float = 1e-3,
    alpha: float = 0.1,
    beta_loss: float = 0.1,
    resolutions: list[float] = [0.3, 0.5, 1.0, 1.5],
    output_dir: str = "benchmark_output",
    seed: int = 42,
) -> dict:
    """Run a single benchmark scenario. Returns results dict."""

    log.info(
        f"Generating: {n_cells} cells, {n_genes} genes, {n_niches} niches, "
        f"cell_noise={cell_noise_scale}, gene_specific={gene_specific_programs}"
    )
    synth = make_synthetic_dataset(
        n_cells=n_cells,
        n_genes=n_genes,
        n_mods_shared=n_mods_shared,
        n_mods_gene=n_mods_gene,
        n_niches=n_niches,
        n_active_mods=n_active_mods,
        beta_signal=beta_signal,
        beta_noise=beta_noise,
        sparsity=sparsity,
        cell_noise_scale=cell_noise_scale,
        seed=seed,
        gene_specific_programs=gene_specific_programs,
    )
    true_labels = synth.true_labels
    n_mods_total = len(synth.mod_vocab)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    log.info("  PCA on signed betas …")
    t0 = time.time()
    flat = _flat_betas(synth.dataset.gene_betas, n_cells, n_mods_total)
    z_pca = PCA(n_components=hidden_dim, random_state=seed).fit_transform(flat)
    t_pca = time.time() - t0
    pca_metrics = best_ari_nmi(z_pca, true_labels, resolutions)
    log.info(f"    ARI={pca_metrics['best']['ari']:.4f}  NMI={pca_metrics['best']['nmi']:.4f}  ({t_pca:.1f}s)")
    np.save(str(out / "pca_embeddings.npy"), z_pca)

    log.info("  PCA + spatial smooth …")
    z_smooth = _spatial_smooth(z_pca, synth.dataset.edge_index, n_hops=gcn_layers)
    smooth_metrics = best_ari_nmi(z_smooth, true_labels, resolutions)
    log.info(f"    ARI={smooth_metrics['best']['ari']:.4f}  NMI={smooth_metrics['best']['nmi']:.4f}")
    np.save(str(out / "pca_smooth_embeddings.npy"), z_smooth)

    log.info("  PCA on mean|β| (rec_target) …")
    z_rec = PCA(n_components=hidden_dim, random_state=seed).fit_transform(
        synth.dataset.rec_target.numpy()
    )
    rec_metrics = best_ari_nmi(z_rec, true_labels, resolutions)
    log.info(f"    ARI={rec_metrics['best']['ari']:.4f}  NMI={rec_metrics['best']['nmi']:.4f}")
    np.save(str(out / "rec_pca_embeddings.npy"), z_rec)

    # ------------------------------------------------------------------
    # SimpleNicheModel
    # ------------------------------------------------------------------
    log.info("  Training SimpleNicheModel …")
    t0 = time.time()
    z_model = train_simple(
        dataset=synth.dataset,
        output_dir=str(out / "model"),
        hidden_dim=hidden_dim,
        mlp_layers=mlp_layers,
        gcn_layers=gcn_layers,
        epochs=epochs,
        lr=lr,
        alpha=alpha,
        beta=beta_loss,
        device_str="auto",
        log_every=epochs // 5,
    )
    t_model = time.time() - t0
    model_metrics = best_ari_nmi(z_model, true_labels, resolutions)
    log.info(
        f"    ARI={model_metrics['best']['ari']:.4f}  NMI={model_metrics['best']['nmi']:.4f}  ({t_model:.1f}s)"
    )
    np.save(str(out / "model_embeddings.npy"), z_model)
    np.save(str(out / "true_labels.npy"), true_labels)

    results = {
        "params": {
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_niches": n_niches,
            "n_mods_total": n_mods_total,
            "n_active_mods": n_active_mods,
            "beta_signal": beta_signal,
            "beta_noise": beta_noise,
            "sparsity": sparsity,
            "cell_noise_scale": cell_noise_scale,
            "gene_specific_programs": gene_specific_programs,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "seed": seed,
        },
        "pca":         {**pca_metrics,   "time_s": t_pca},
        "pca_smooth":  {**smooth_metrics, "time_s": t_pca},
        "pca_rec":     {**rec_metrics,    "time_s": t_pca},
        "model":       {**model_metrics,  "time_s": t_model},
    }

    with open(out / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    _print_table(results)
    return results


def _print_table(results: dict) -> None:
    p = results["params"]
    log.info("\n" + "=" * 75)
    log.info(
        f"  {p['n_cells']} cells · {p['n_genes']} genes · {p['n_niches']} niches · "
        f"{p['n_mods_total']} mods"
    )
    log.info(
        f"  signal={p['beta_signal']}  noise={p['beta_noise']}  "
        f"cell_noise={p['cell_noise_scale']}  sparsity={p['sparsity']}  "
        f"gene_specific={p['gene_specific_programs']}"
    )
    log.info("-" * 75)
    log.info(f"{'Method':<30} {'ARI':>8} {'NMI':>8} {'Clusters':>10} {'Time(s)':>9}")
    log.info("-" * 75)
    rows = [
        ("PCA (signed betas)",       "pca"),
        ("PCA + spatial smooth",     "pca_smooth"),
        ("PCA (mean|β| only)",       "pca_rec"),
        ("SimpleNicheModel",         "model"),
    ]
    for label, key in rows:
        if key not in results:
            continue
        b = results[key]["best"]
        t = results[key].get("time_s", 0)
        log.info(f"{label:<30} {b['ari']:>8.4f} {b['nmi']:>8.4f} {b['n_clusters']:>10} {t:>9.1f}")
    log.info("=" * 75)


# ---------------------------------------------------------------------------
# Multi-scenario benchmark
# ---------------------------------------------------------------------------

def run_multi_benchmark(
    output_dir: str = "benchmark_output",
    epochs: int = 500,
    hidden_dim: int = 64,
    seed: int = 42,
) -> list[dict]:
    """
    Three scenarios that progressively stress the model:

    A  Sign-coded niches, no cell noise — proves model captures sign patterns
       (PCA-on-|β| fails; model and PCA-on-signed succeed)
    B  Sign-coded + moderate cell noise — tests spatial GNN denoising
    C  Gene-specific programs + noise — tests cross-gene integration
    """
    scenarios = [
        {
            "name": "A: sign-coded niches\n(low noise)",
            "n_cells": 1000, "n_genes": 8, "n_niches": 5,
            "n_mods_shared": 200, "n_mods_gene": 20, "n_active_mods": 20,
            "beta_signal": 1.5, "beta_noise": 0.1, "sparsity": 0.65,
            "cell_noise_scale": 0.0, "gene_specific_programs": False,
        },
        {
            "name": "B: sign-coded + cell noise\n(spatial GNN helps)",
            "n_cells": 1000, "n_genes": 8, "n_niches": 5,
            "n_mods_shared": 200, "n_mods_gene": 20, "n_active_mods": 20,
            "beta_signal": 1.5, "beta_noise": 0.1, "sparsity": 0.65,
            "cell_noise_scale": 1.0, "gene_specific_programs": False,
        },
        {
            "name": "C: gene-specific + noise\n(cross-gene aggregation)",
            "n_cells": 1000, "n_genes": 8, "n_niches": 5,
            "n_mods_shared": 200, "n_mods_gene": 20, "n_active_mods": 20,
            "beta_signal": 1.5, "beta_noise": 0.1, "sparsity": 0.65,
            "cell_noise_scale": 1.0, "gene_specific_programs": True,
        },
    ]

    all_results = []
    for sc in scenarios:
        name_safe = sc["name"].split("\n")[0].replace(" ", "_").replace(":", "").replace("/", "")
        sc_out = str(Path(output_dir) / name_safe)
        kw = {k: v for k, v in sc.items() if k != "name"}
        res = run_benchmark(
            **kw,
            hidden_dim=hidden_dim,
            epochs=epochs,
            output_dir=sc_out,
            seed=seed,
        )
        res["scenario_name"] = sc["name"]
        all_results.append(res)

    _save_comparison_chart(all_results, output_dir)

    with open(Path(output_dir) / "multi_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def _save_comparison_chart(all_results: list[dict], output_dir: str) -> None:
    names   = [r["scenario_name"] for r in all_results]
    methods = [
        ("PCA (signed β)",   "pca",        "#4C72B0"),
        ("PCA + smooth",     "pca_smooth",  "#55A868"),
        ("PCA (mean|β|)",    "pca_rec",     "#8172B2"),
        ("SimpleNicheModel", "model",       "#DD8452"),
    ]

    x = np.arange(len(names))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ["ARI", "NMI"]):
        for i, (label, key, color) in enumerate(methods):
            vals = [r[key]["best"]["ari" if metric == "ARI" else "nmi"]
                    for r in all_results]
            bars = ax.bar(x + (i - 1.5) * width, vals, width,
                          label=label, color=color, alpha=0.87)
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.3f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("\\n", "\n") for n in names], fontsize=9)
        ax.set_ylim(0, 1.10)
        ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric}: SimpleNicheModel vs PCA baselines", fontsize=12)
        ax.legend(fontsize=9)

    plt.suptitle(
        "Functional Microniche Embeddings — SimpleNicheModel vs PCA\n"
        f"({all_results[0]['params']['n_cells']} cells · "
        f"{all_results[0]['params']['n_genes']} genes · "
        f"{all_results[0]['params']['n_niches']} niches · "
        f"{all_results[0]['params']['epochs']} epochs)",
        fontsize=11,
    )
    plt.tight_layout()
    out = Path(output_dir) / "benchmark_comparison.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    log.info(f"Saved comparison chart → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Benchmark SimpleNicheModel vs PCA")
    parser.add_argument("--multi", action="store_true",
                        help="Run three difficulty scenarios and produce a comparison chart")
    parser.add_argument("--n-cells",       type=int,   default=1000)
    parser.add_argument("--n-genes",       type=int,   default=8)
    parser.add_argument("--n-niches",      type=int,   default=5)
    parser.add_argument("--n-mods-shared", type=int,   default=200)
    parser.add_argument("--n-mods-gene",   type=int,   default=20)
    parser.add_argument("--n-active-mods", type=int,   default=20)
    parser.add_argument("--beta-signal",   type=float, default=1.5)
    parser.add_argument("--beta-noise",    type=float, default=0.1)
    parser.add_argument("--sparsity",      type=float, default=0.65)
    parser.add_argument("--cell-noise",    type=float, default=0.0,
                        help="Additional per-cell iid noise (tests spatial GNN advantage)")
    parser.add_argument("--gene-specific", action="store_true",
                        help="Use gene-specific niche programs")
    parser.add_argument("--hidden-dim",    type=int,   default=64)
    parser.add_argument("--mlp-layers",    type=int,   default=2)
    parser.add_argument("--gcn-layers",    type=int,   default=2)
    parser.add_argument("--epochs",        type=int,   default=500)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--alpha",         type=float, default=0.1)
    parser.add_argument("--beta-loss",     type=float, default=0.1)
    parser.add_argument("--output-dir",    default="benchmark_output")
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    if args.multi:
        run_multi_benchmark(
            output_dir=args.output_dir,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
        )
    else:
        run_benchmark(
            n_cells=args.n_cells,
            n_genes=args.n_genes,
            n_niches=args.n_niches,
            n_mods_shared=args.n_mods_shared,
            n_mods_gene=args.n_mods_gene,
            n_active_mods=args.n_active_mods,
            beta_signal=args.beta_signal,
            beta_noise=args.beta_noise,
            sparsity=args.sparsity,
            cell_noise_scale=args.cell_noise,
            gene_specific_programs=args.gene_specific,
            hidden_dim=args.hidden_dim,
            mlp_layers=args.mlp_layers,
            gcn_layers=args.gcn_layers,
            epochs=args.epochs,
            lr=args.lr,
            alpha=args.alpha,
            beta_loss=args.beta_loss,
            output_dir=args.output_dir,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
