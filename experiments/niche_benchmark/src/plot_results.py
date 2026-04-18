"""Render side-by-side spatial maps and metric bar plots."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from _common import FIG_DIR, LABELS_DIR, RESULTS_DIR


def _categorical_palette(n: int) -> list:
    if n <= 20:
        cmap = matplotlib.colormaps["tab20"]
        return [cmap(i % 20) for i in range(n)]
    cmap = matplotlib.colormaps["nipy_spectral"]
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def _scatter(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray, title: str) -> None:
    cats = sorted(np.unique(labels).tolist(), key=lambda x: str(x))
    colors = _categorical_palette(len(cats))
    color_map = {c: colors[i] for i, c in enumerate(cats)}
    rgb = np.array([color_map[v] for v in labels])
    ax.scatter(coords[:, 0], coords[:, 1], c=rgb, s=2.0, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{title}  (k={len(cats)})", fontsize=11)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(RESULTS_DIR / "tonsil_prepared.h5ad")
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    gt = adata.obs["microniche_gt"].astype(str).to_numpy()

    methods = []
    for name in ["banksy", "nichecompass"]:
        csv = LABELS_DIR / f"{name}.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv).set_index("cell_id")
        df = df.reindex(adata.obs_names)
        labels = df["label"].astype(str).fillna("NA").to_numpy()
        methods.append((name, labels))

    n_panels = 1 + len(methods)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    _scatter(axes[0], coords, gt, "Ground truth (cell_type_2)")
    for ax, (name, labels) in zip(axes[1:], methods):
        _scatter(ax, coords, labels, f"{name}")
    fig.suptitle("Functional microniches: SlideTags human tonsil", fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "spatial_microniches.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    metrics_csv = RESULTS_DIR / "metrics.csv"
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        plot_metrics = ["ari", "nmi", "ami", "v_measure", "homogeneity", "completeness", "spatial_purity_k10"]
        df_plot = df[df["method"] != "ground_truth"][["method"] + plot_metrics].set_index("method")
        ax = df_plot.plot(kind="bar", figsize=(10, 5), width=0.85, colormap="tab10")
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.set_title("Microniche identification: BANKSY vs NicheCompass\n(SlideTags human tonsil, ground truth = cell_type_2)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0), frameon=False)
        plt.xticks(rotation=0)
        plt.tight_layout()
        out2 = FIG_DIR / "metrics_bars.png"
        plt.savefig(out2, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"wrote {out2}")


if __name__ == "__main__":
    main()
