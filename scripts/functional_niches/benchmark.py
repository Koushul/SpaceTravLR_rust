"""
Benchmark FunctionalNicheModel vs PCA on synthetic data with known niches.

Metrics
-------
- ARI  (Adjusted Rand Index)    — clustering agreement, chance-adjusted
- NMI  (Normalized Mutual Info) — information-theoretic agreement

Both metrics are in [0, 1] (or [-1, 1] for ARI).  Higher is better.

PCA baseline
------------
We flatten the per-gene beta matrices into a single [N, n_genes * n_mods_total]
matrix, apply PCA to reduce to the same dimensionality as the model embedding,
then run the same Leiden clustering at the same resolution.  This is a strong
baseline because the raw beta signal already contains the niche information —
the model has to learn to extract it more efficiently.

Usage
-----
    python -m functional_niches.benchmark
    python -m functional_niches.benchmark --n-cells 3000 --epochs 200 --output-dir /tmp/bench
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .synth import make_synthetic_dataset
from .train import train
from .cluster import cluster_embeddings

log = logging.getLogger(__name__)


def flatten_betas(gene_betas: list, n_cells: int, n_mods_total: int) -> np.ndarray:
    """
    Flatten all gene beta matrices into a single [N, G * n_mods_total] matrix.
    Retains SIGN information — appropriate PCA baseline.
    """
    parts = []
    for gb in gene_betas:
        mat = np.zeros((n_cells, n_mods_total), dtype=np.float32)
        mod_idx = gb.mod_indices[0].numpy()   # [M_g]
        mat[:, mod_idx] = gb.beta_values.numpy()
        parts.append(mat)
    return np.concatenate(parts, axis=1)


def rec_target_pca(dataset, n_components: int) -> np.ndarray:
    """
    PCA on the model's own reconstruction target (mean |beta|).
    This is a 'same-input' baseline: uses only what the model reconstructs.
    Note: loses sign information, so sign-discriminative niches will confuse it.
    """
    from sklearn.decomposition import PCA
    rec = dataset.rec_target.numpy()
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(rec)


def spatial_smooth_pca(
    z_pca: np.ndarray,
    edge_index: "torch.LongTensor",  # noqa: F821
    n_hops: int = 2,
) -> np.ndarray:
    """
    Apply n_hops of mean-neighbor smoothing on the PCA embedding.
    Approximates what the spatial GNN does to the model embedding.
    """
    import scipy.sparse as sp
    import torch
    n = z_pca.shape[0]
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    data = np.ones(len(src), dtype=np.float32)
    adj = sp.csr_matrix((data, (src, dst)), shape=(n, n))
    # row-normalise
    deg = np.array(adj.sum(axis=1)).ravel()
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    D_inv = sp.diags(deg_inv)
    adj_norm = D_inv @ adj

    z = z_pca.copy()
    for _ in range(n_hops):
        z = adj_norm @ z
    return z


def best_ari_nmi(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    resolutions: list[float] = [0.3, 0.5, 1.0, 1.5],
) -> dict:
    """
    Run Leiden at multiple resolutions; return the best ARI, corresponding NMI,
    and the resolution that achieved it.
    """
    results = cluster_embeddings(embeddings, resolutions=resolutions)
    best = {"ari": -1.0, "nmi": 0.0, "resolution": None, "n_clusters": 0}
    per_res = []

    for r, labels in results.items():
        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels, average_method="arithmetic")
        n_clusters = len(set(labels))
        per_res.append({"resolution": r, "ari": ari, "nmi": nmi, "n_clusters": n_clusters})
        if ari > best["ari"]:
            best = {"ari": ari, "nmi": nmi, "resolution": r, "n_clusters": n_clusters}

    return {"best": best, "per_resolution": per_res}


def run_benchmark(
    n_cells: int = 600,
    n_genes: int = 8,
    n_niches: int = 5,
    n_mods_shared: int = 300,
    n_mods_gene: int = 20,
    n_active_mods: int = 20,
    beta_signal: float = 1.0,
    beta_noise: float = 0.3,
    sparsity: float = 0.75,
    cell_noise_scale: float = 0.0,
    hidden_dim: int = 32,
    epochs: int = 150,
    lr: float = 1e-3,
    alpha: float = 0.1,
    beta_loss: float = 0.1,
    output_dir: str = "benchmark_output",
    resolutions: list[float] = [0.3, 0.5, 1.0, 1.5],
    seed: int = 42,
    gene_specific_programs: bool = False,
) -> dict:
    """
    End-to-end benchmark.

    Returns
    -------
    results dict with keys 'model', 'pca', 'params'
    """
    log.info(
        f"Generating synthetic dataset: {n_cells} cells, {n_genes} genes, "
        f"{n_niches} niches"
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

    # ---------------------------------------------------------------
    # 1) PCA baselines
    # ---------------------------------------------------------------
    log.info("Running PCA baseline ...")
    t0 = time.time()
    flat = flatten_betas(synth.dataset.gene_betas, n_cells, n_mods_total)
    pca = PCA(n_components=hidden_dim, random_state=seed)
    z_pca = pca.fit_transform(flat)
    t_pca = time.time() - t0
    log.info(f"  PCA done in {t_pca:.1f}s")

    pca_metrics = best_ari_nmi(z_pca, true_labels, resolutions)
    log.info(
        f"  PCA best: ARI={pca_metrics['best']['ari']:.4f}  "
        f"NMI={pca_metrics['best']['nmi']:.4f}  "
        f"@ r={pca_metrics['best']['resolution']}  "
        f"({pca_metrics['best']['n_clusters']} clusters)"
    )
    np.save(str(out / "pca_embeddings.npy"), z_pca)

    # PCA + spatial smoothing (same n_hops as GNN layers)
    log.info("Running PCA + spatial smoothing baseline ...")
    z_pca_smooth = spatial_smooth_pca(z_pca, synth.dataset.edge_index, n_hops=2)
    pca_smooth_metrics = best_ari_nmi(z_pca_smooth, true_labels, resolutions)
    log.info(
        f"  PCA+smooth best: ARI={pca_smooth_metrics['best']['ari']:.4f}  "
        f"NMI={pca_smooth_metrics['best']['nmi']:.4f}  "
        f"@ r={pca_smooth_metrics['best']['resolution']}  "
        f"({pca_smooth_metrics['best']['n_clusters']} clusters)"
    )
    np.save(str(out / "pca_smooth_embeddings.npy"), z_pca_smooth)

    # PCA on rec_target (same input as decoder, loses sign information)
    log.info("Running PCA on rec_target (same-input baseline) ...")
    z_rec_pca = rec_target_pca(synth.dataset, n_components=hidden_dim)
    rec_pca_metrics = best_ari_nmi(z_rec_pca, true_labels, resolutions)
    log.info(
        f"  RecPCA best: ARI={rec_pca_metrics['best']['ari']:.4f}  "
        f"NMI={rec_pca_metrics['best']['nmi']:.4f}  "
        f"@ r={rec_pca_metrics['best']['resolution']}"
    )
    np.save(str(out / "rec_pca_embeddings.npy"), z_rec_pca)

    # ---------------------------------------------------------------
    # 2) FunctionalNicheModel
    # ---------------------------------------------------------------
    log.info("Training FunctionalNicheModel ...")
    t0 = time.time()
    z_model = train(
        dataset=synth.dataset,
        output_dir=str(out / "model"),
        hidden_dim=hidden_dim,
        embed_dim=min(32, hidden_dim // 2),
        gene_embed_dim=16,
        n_heads=4,
        gnn_layers=2,
        gnn_heads=4,
        dropout=0.1,
        epochs=epochs,
        lr=lr,
        alpha=alpha,
        beta=beta_loss,
        device_str="auto",
        log_every=50,
    )
    t_model = time.time() - t0
    log.info(f"  Model trained in {t_model:.1f}s")

    model_metrics = best_ari_nmi(z_model, true_labels, resolutions)
    log.info(
        f"  Model best: ARI={model_metrics['best']['ari']:.4f}  "
        f"NMI={model_metrics['best']['nmi']:.4f}  "
        f"@ r={model_metrics['best']['resolution']}  "
        f"({model_metrics['best']['n_clusters']} clusters)"
    )

    np.save(str(out / "model_embeddings.npy"), z_model)
    np.save(str(out / "true_labels.npy"), true_labels)

    # ---------------------------------------------------------------
    # 3) Summary
    # ---------------------------------------------------------------
    results = {
        "params": {
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_niches": n_niches,
            "n_mods_shared": n_mods_shared,
            "n_mods_gene": n_mods_gene,
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
        "pca": {**pca_metrics, "time_s": t_pca},
        "pca_smooth": {**pca_smooth_metrics, "time_s": t_pca},
        "pca_rec_target": {**rec_pca_metrics, "time_s": t_pca},
        "model": {**model_metrics, "time_s": t_model},
    }

    with open(out / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    _print_table(results)
    return results


def _print_table(results: dict) -> None:
    p = results["params"]
    log.info("\n" + "=" * 75)
    log.info(f"  Benchmark: {p['n_cells']} cells  {p['n_genes']} genes  "
             f"{p['n_niches']} niches  {p['n_mods_total']} mods total")
    log.info(f"  Beta signal={p['beta_signal']}  noise={p['beta_noise']}  "
             f"cell_noise={p.get('cell_noise_scale', 0.0)}  sparsity={p['sparsity']}  "
             f"gene_specific={p.get('gene_specific_programs', False)}")
    log.info("-" * 75)
    log.info(f"{'Method':<30} {'ARI':>8} {'NMI':>8} {'Clusters':>10} {'Res':>6}")
    log.info("-" * 75)
    for label, key in [
        ("PCA on raw betas", "pca"),
        ("PCA + spatial smooth", "pca_smooth"),
        ("PCA on rec_target (|beta|)", "pca_rec_target"),
        ("FuncNiche Model", "model"),
    ]:
        if key not in results:
            continue
        b = results[key]["best"]
        log.info(
            f"{label:<30} {b['ari']:>8.4f} {b['nmi']:>8.4f} "
            f"{b['n_clusters']:>10} {b['resolution']:>6}"
        )
    log.info("=" * 75)


def run_multi_benchmark(
    output_dir: str = "benchmark_output",
    epochs: int = 300,
    hidden_dim: int = 64,
    seed: int = 42,
) -> list[dict]:
    """
    Run three difficulty levels and produce a comparative table + bar chart.

    Returns list of result dicts (one per scenario).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = [
        {
            # Easy: strong signal, no cell noise. Both PCA variants and model should succeed.
            # PCA-on-rec_target fails because niches differ only in sign pattern.
            "name": "Scenario A\n(sign-coded niches)",
            "n_cells": 200, "n_genes": 5, "n_niches": 3,
            "n_mods_shared": 60, "n_mods_gene": 8, "n_active_mods": 12,
            "beta_signal": 1.5, "beta_noise": 0.1, "sparsity": 0.65,
            "cell_noise_scale": 0.0, "gene_specific_programs": False,
        },
        {
            # Medium: sign-coded niches + moderate cell noise.
            # PCA on raw betas degrades; spatial GNN in model helps.
            "name": "Scenario B\n(sign-coded + cell noise)",
            "n_cells": 200, "n_genes": 5, "n_niches": 3,
            "n_mods_shared": 60, "n_mods_gene": 8, "n_active_mods": 12,
            "beta_signal": 1.5, "beta_noise": 0.1, "sparsity": 0.65,
            "cell_noise_scale": 0.8, "gene_specific_programs": False,
        },
        {
            # Hard: gene-specific programs, high noise.
            # Requires cross-gene aggregation to find niche identity.
            "name": "Scenario C\n(gene-specific + high noise)",
            "n_cells": 200, "n_genes": 5, "n_niches": 3,
            "n_mods_shared": 60, "n_mods_gene": 8, "n_active_mods": 12,
            "beta_signal": 1.5, "beta_noise": 0.1, "sparsity": 0.65,
            "cell_noise_scale": 0.8, "gene_specific_programs": True,
        },
    ]

    all_results = []
    for sc in scenarios:
        sc_name_safe = sc["name"].replace("\n", "_").replace(" ", "").replace("(", "").replace(")", "")
        sc_out = f"{output_dir}/{sc_name_safe}"
        res = run_benchmark(
            n_cells=sc["n_cells"],
            n_genes=sc["n_genes"],
            n_niches=sc["n_niches"],
            n_mods_shared=sc["n_mods_shared"],
            n_mods_gene=sc["n_mods_gene"],
            n_active_mods=sc["n_active_mods"],
            beta_signal=sc["beta_signal"],
            beta_noise=sc["beta_noise"],
            sparsity=sc["sparsity"],
            cell_noise_scale=sc.get("cell_noise_scale", 0.0),
            gene_specific_programs=sc.get("gene_specific_programs", False),
            hidden_dim=hidden_dim,
            epochs=epochs,
            output_dir=sc_out,
            seed=seed,
        )
        res["scenario_name"] = sc["name"]
        all_results.append(res)

    # --- combined bar chart ---
    out = Path(output_dir)
    names = [r["scenario_name"] for r in all_results]
    pca_ari = [r["pca"]["best"]["ari"] for r in all_results]
    pca_smooth_ari = [r.get("pca_smooth", r["pca"])["best"]["ari"] for r in all_results]
    model_ari = [r["model"]["best"]["ari"] for r in all_results]
    pca_nmi = [r["pca"]["best"]["nmi"] for r in all_results]
    pca_smooth_nmi = [r.get("pca_smooth", r["pca"])["best"]["nmi"] for r in all_results]
    model_nmi = [r["model"]["best"]["nmi"] for r in all_results]

    x = np.arange(len(names))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, v1, v2, v3, ylabel in [
        (axes[0], "ARI", pca_ari, pca_smooth_ari, model_ari, "Adjusted Rand Index"),
        (axes[1], "NMI", pca_nmi, pca_smooth_nmi, model_nmi, "Normalized Mutual Info"),
    ]:
        bars1 = ax.bar(x - width, v1, width, label="PCA", color="#4C72B0", alpha=0.85)
        bars2 = ax.bar(x, v2, width, label="PCA + spatial smooth", color="#55A868", alpha=0.85)
        bars3 = ax.bar(x + width, v3, width, label="FuncNiche Model", color="#DD8452", alpha=0.85)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{metric}: Three Methods", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("\\n", "\n") for n in names], fontsize=9)
        ax.set_ylim(0, 1.10)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=9)
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                ax.annotate(f"{bar.get_height():.3f}",
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=7)

    plt.suptitle(
        "Functional Microniche Embeddings — ARI & NMI vs PCA\n"
        f"({scenarios[0]['n_cells']} cells, {scenarios[0]['n_genes']} genes, "
        f"{scenarios[0]['n_niches']} niches, {epochs} epochs)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(str(out / "benchmark_comparison.png"), dpi=150)
    plt.close()
    log.info(f"Saved comparison chart to {out / 'benchmark_comparison.png'}")

    with open(out / "multi_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Benchmark functional niches vs PCA")
    parser.add_argument("--n-cells", type=int, default=600)
    parser.add_argument("--n-genes", type=int, default=8)
    parser.add_argument("--n-niches", type=int, default=5)
    parser.add_argument("--n-mods-shared", type=int, default=300)
    parser.add_argument("--n-mods-gene", type=int, default=20)
    parser.add_argument("--n-active-mods", type=int, default=20)
    parser.add_argument("--beta-signal", type=float, default=1.0)
    parser.add_argument("--beta-noise", type=float, default=0.3)
    parser.add_argument("--sparsity", type=float, default=0.75)
    parser.add_argument("--cell-noise-scale", type=float, default=0.0,
                        help="Additional per-cell iid noise (tests spatial GNN advantage)")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta-loss", type=float, default=0.5)
    parser.add_argument("--output-dir", default="benchmark_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi", action="store_true",
                        help="Run three difficulty scenarios and produce a comparison chart")
    args = parser.parse_args()

    if getattr(args, "multi", False):
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
            cell_noise_scale=args.cell_noise_scale,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            alpha=args.alpha,
            beta_loss=args.beta_loss,
            output_dir=args.output_dir,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
