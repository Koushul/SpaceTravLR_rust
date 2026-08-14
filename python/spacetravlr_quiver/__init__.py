"""SpaceTravLR UMAP transition / quiver fields (Rust backend via PyO3)."""

from __future__ import annotations

from spacetravlr_quiver._lib import (  # type: ignore[attr-defined]
    __version__,
    col_delta_cor_partial_py as col_delta_cor_partial,
    col_delta_cor_py as col_delta_cor,
    compute_transition_grid,
    pearson_velocity_vs_expr_delta,
    round_delta_py as round_delta,
    umap_grid_axes_py as umap_grid_axes,
    umap_knn,
)

from .plot import plot_quiver_side_by_side, plot_transition_panel
from .sweep import run_il21_parameter_sweep
from .umap_embed import ensure_umap_embedding

__all__ = [
    "__version__",
    "col_delta_cor",
    "col_delta_cor_partial",
    "compute_transition_grid",
    "pearson_velocity_vs_expr_delta",
    "round_delta",
    "umap_grid_axes",
    "umap_knn",
    "plot_quiver_side_by_side",
    "plot_transition_panel",
    "ensure_umap_embedding",
    "run_il21_parameter_sweep",
]
