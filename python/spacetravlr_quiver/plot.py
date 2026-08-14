"""Matplotlib quiver plotting for Rust transition grids."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_transition_panel(
    ax,
    umap: np.ndarray,
    cell_types: Sequence[str],
    palette: Mapping[str, Any],
    grid: Mapping[str, Any],
    title: str,
    *,
    scatter_size: float = 8,
    quiver_width: float = 0.0025,
) -> None:
    plot_df = pd.DataFrame(
        {"x": umap[:, 0], "y": umap[:, 1], "cell_type": list(cell_types)}
    )
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="cell_type",
        s=scatter_size,
        ax=ax,
        alpha=0.75,
        edgecolor="none",
        palette=palette,
        legend=False,
    )
    gx = np.asarray(grid["grid_points_x"])
    gy = np.asarray(grid["grid_points_y"])
    u = np.asarray(grid["u"])
    v = np.asarray(grid["v"])
    mag = np.hypot(u, v)
    keep = mag > 0
    ax.quiver(
        gx[keep],
        gy[keep],
        u[keep],
        v[keep],
        angles="xy",
        scale_units="xy",
        scale=1,
        headwidth=3,
        headlength=3,
        headaxislength=3,
        width=quiver_width,
        alpha=0.9,
        color="black",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_aspect("equal", adjustable="datalim")


def plot_quiver_side_by_side(
    umap: np.ndarray,
    cell_types: Sequence[str],
    panels: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    out_path: str | None = None,
    figsize_per: tuple[float, float] = (5.0, 4.5),
    dpi: int = 200,
    suptitle: str | None = None,
) -> plt.Figure:
    """Draw KO|OE (or any list) side-by-side panels sharing the same UMAP."""
    n = len(panels)
    fig, axes = plt.subplots(
        1, n, figsize=(figsize_per[0] * n, figsize_per[1]), dpi=dpi, squeeze=False
    )
    cts = sorted(set(map(str, cell_types)))
    cmap = plt.get_cmap("tab20")
    palette = {ct: cmap(i % 20) for i, ct in enumerate(cts)}
    for ax, (title, grid) in zip(axes[0], panels):
        plot_transition_panel(ax, umap, cell_types, palette, grid, title)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette[ct],
            markersize=6,
            label=ct,
        )
        for ct in cts
    ]
    fig.legend(handles=handles, loc="center right", fontsize=7, frameon=False, title="cell_type")
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.86, 0.95 if suptitle else 0.98])
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
    return fig
