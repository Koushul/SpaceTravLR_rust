"""UMAP and spatial visualisations for functional microniche embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import umap


_PALETTE = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())


def _label_to_color(labels: np.ndarray) -> tuple[list, dict]:
    unique = sorted(set(labels))
    color_map = {lbl: _PALETTE[i % len(_PALETTE)] for i, lbl in enumerate(unique)}
    colors = [color_map[l] for l in labels]
    return colors, color_map


def plot_umap(
    z: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "UMAP — Functional Niches",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """
    Compute UMAP embedding and save coloured scatter plot.

    Returns
    -------
    umap_coords : [N, 2]
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    coords = reducer.fit_transform(z)

    colors, color_map = _label_to_color(labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=5, alpha=0.7, rasterized=True)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=6, label=lbl)
        for lbl, c in sorted(color_map.items())
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=7, ncol=max(1, len(color_map) // 20))
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return coords


def plot_spatial(
    spatial_coords: np.ndarray,
    labels: np.ndarray,
    output_path: str,
    title: str = "Spatial — Functional Niches",
    spot_size: float = 20.0,
) -> None:
    """Save a spatial scatter plot coloured by niche labels."""
    colors, color_map = _label_to_color(labels)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        spatial_coords[:, 0], spatial_coords[:, 1],
        c=colors, s=spot_size, alpha=0.8, rasterized=True,
    )
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markersize=6, label=lbl)
        for lbl, c in sorted(color_map.items())
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=7, ncol=max(1, len(color_map) // 20))
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_all_plots(
    z: np.ndarray,
    label_df: pd.DataFrame,
    spatial_coords: Optional[np.ndarray],
    output_dir: str,
    resolution: float = 0.5,
) -> None:
    """Save UMAP and (optionally) spatial plots for one resolution."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    col = f"niche_{resolution}"
    if col not in label_df.columns:
        col = label_df.columns[1]

    labels = label_df[col].values.astype(str)
    plot_umap(z, labels, str(out / "umap.png"), title=f"UMAP — Niches r={resolution}")

    if spatial_coords is not None:
        plot_spatial(
            spatial_coords, labels,
            str(out / "spatial.png"),
            title=f"Spatial — Niches r={resolution}",
        )
