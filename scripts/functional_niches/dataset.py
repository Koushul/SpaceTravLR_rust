"""Data loading, vocabulary building, and spatial graph construction."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pyarrow.feather as feather
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass
class GeneBetadata:
    gene_name: str
    gene_index: int
    mod_indices: torch.LongTensor    # [N, M_g]
    beta_values: torch.FloatTensor   # [N, M_g]
    n_mods: int                      # M_g (no padding within a gene)


@dataclass
class FunctionalNicheDataset:
    cell_ids: list[str]
    gene_betas: list[GeneBetadata]
    edge_index: torch.LongTensor     # [2, E]
    edge_weight: torch.FloatTensor   # [E]
    mod_vocab: dict[str, int]        # modulator name → index
    gene_names: list[str]
    # precomputed reconstruction target: [N, n_mods_total]
    rec_target: torch.FloatTensor
    # precomputed flat signed-beta input: [N, G * n_mods_total]
    # built lazily via .get_beta_matrix(); None until first call
    _beta_matrix: "Optional[torch.FloatTensor]" = None

    def get_beta_matrix(self, concat_genes: bool = True) -> torch.FloatTensor:
        """Return (and cache) the flat signed-beta cell-feature matrix."""
        if self._beta_matrix is None:
            n_cells = len(self.cell_ids)
            n_mods = len(self.mod_vocab)
            self._beta_matrix = make_beta_matrix(
                self.gene_betas, n_cells, n_mods, concat_genes=concat_genes
            )
        return self._beta_matrix


def build_modulator_vocab(feather_dir: str) -> dict[str, int]:
    all_mods: set[str] = set()
    for path in sorted(Path(feather_dir).glob("*_betadata.feather")):
        cols = feather.read_table(path).column_names
        all_mods.update(c for c in cols if c.startswith("beta_"))
    return {name: i for i, name in enumerate(sorted(all_mods))}


def build_spatial_graph(
    spatial_coords: np.ndarray,
    k: int = 6,
    sigma: Optional[float] = None,
) -> tuple[torch.LongTensor, torch.FloatTensor]:
    """
    Build a kNN spatial graph.

    Parameters
    ----------
    spatial_coords : [N, 2] array of (x, y) coordinates
    k : number of neighbors
    sigma : RBF bandwidth; defaults to median pairwise distance

    Returns
    -------
    edge_index : [2, E]
    edge_weight : [E]
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(spatial_coords)
    distances, indices = nbrs.kneighbors(spatial_coords)
    # drop self-loop (index 0)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    if sigma is None:
        sigma = float(np.median(distances))

    N = spatial_coords.shape[0]
    src = np.repeat(np.arange(N), k)
    dst = indices.ravel()
    dists = distances.ravel()

    weights = np.exp(-(dists ** 2) / (sigma ** 2 + 1e-8))

    edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
    edge_weight = torch.from_numpy(weights).float()
    return edge_index, edge_weight


def _load_gene_betadata(
    path: Path,
    cell_order: list[str],
    mod_vocab: dict[str, int],
    gene_index: int,
) -> GeneBetadata:
    tbl = feather.read_table(path)
    df = tbl.to_pandas()

    # align cells
    id_col = "CellID" if "CellID" in df.columns else df.columns[0]
    df = df.set_index(id_col).reindex(cell_order)

    beta_cols = [c for c in df.columns if c.startswith("beta_")]
    beta_vals = df[beta_cols].values.astype(np.float32)  # [N, M_g]
    mod_idx = np.array([mod_vocab[c] for c in beta_cols], dtype=np.int64)  # [M_g]
    # broadcast mod indices across all cells
    mod_idx_mat = np.broadcast_to(mod_idx[None, :], (len(cell_order), len(mod_idx))).copy()

    return GeneBetadata(
        gene_name=path.stem.replace("_betadata", ""),
        gene_index=gene_index,
        mod_indices=torch.from_numpy(mod_idx_mat).long(),
        beta_values=torch.from_numpy(beta_vals).float(),
        n_mods=len(beta_cols),
    )


def _build_rec_target(
    gene_betas: list[GeneBetadata],
    n_cells: int,
    n_mods_total: int,
) -> torch.FloatTensor:
    """mean |beta| across genes, for each modulator in the global vocab."""
    accumulator = torch.zeros(n_cells, n_mods_total)
    counts = torch.zeros(n_mods_total)

    for gb in gene_betas:
        mod_idx = gb.mod_indices[0]  # [M_g] — same for all cells
        abs_betas = gb.beta_values.abs()  # [N, M_g]
        accumulator.index_add_(1, mod_idx, abs_betas)
        counts.index_add_(0, mod_idx, torch.ones(gb.n_mods))

    counts = counts.clamp(min=1.0)
    return accumulator / counts.unsqueeze(0)


def make_beta_matrix(
    gene_betas: list[GeneBetadata],
    n_cells: int,
    n_mods_total: int,
    concat_genes: bool = True,
) -> torch.FloatTensor:
    """
    Flatten all gene beta matrices into a single dense cell-feature matrix.

    Parameters
    ----------
    gene_betas : list of G GeneBetadata objects
    n_cells : N
    n_mods_total : size of the global modulator vocabulary
    concat_genes : if True return [N, G × n_mods_total] (one block per gene);
                   if False return [N, n_mods_total] (signed betas summed across genes)

    Returns
    -------
    [N, G × n_mods_total] or [N, n_mods_total] float tensor
    """
    if concat_genes:
        parts = []
        for gb in gene_betas:
            mat = torch.zeros(n_cells, n_mods_total)
            mat.scatter_(1, gb.mod_indices, gb.beta_values)
            parts.append(mat)
        return torch.cat(parts, dim=1)   # [N, G * n_mods_total]
    else:
        mat = torch.zeros(n_cells, n_mods_total)
        for gb in gene_betas:
            mat.scatter_add_(1, gb.mod_indices, gb.beta_values)
        return mat   # [N, n_mods_total]


def load_dataset(
    feather_dir: str,
    spatial_coords: np.ndarray,
    cell_ids: list[str],
    k: int = 6,
    sigma: Optional[float] = None,
    mod_vocab: Optional[dict[str, int]] = None,
) -> FunctionalNicheDataset:
    """
    Load all *_betadata.feather files and construct the full dataset.

    Parameters
    ----------
    feather_dir : directory containing *_betadata.feather files
    spatial_coords : [N, 2] spatial coordinates aligned to cell_ids
    cell_ids : list of N cell IDs (defines row order)
    k : spatial kNN
    sigma : RBF sigma for edge weights
    mod_vocab : pre-built vocab; built from scratch if None
    """
    feather_paths = sorted(Path(feather_dir).glob("*_betadata.feather"))
    if not feather_paths:
        raise FileNotFoundError(f"No *_betadata.feather files in {feather_dir}")

    if mod_vocab is None:
        mod_vocab = build_modulator_vocab(feather_dir)

    gene_betas = [
        _load_gene_betadata(p, cell_ids, mod_vocab, i)
        for i, p in enumerate(feather_paths)
    ]
    gene_names = [gb.gene_name for gb in gene_betas]

    edge_index, edge_weight = build_spatial_graph(spatial_coords, k=k, sigma=sigma)
    rec_target = _build_rec_target(gene_betas, len(cell_ids), len(mod_vocab))

    return FunctionalNicheDataset(
        cell_ids=cell_ids,
        gene_betas=gene_betas,
        edge_index=edge_index,
        edge_weight=edge_weight,
        mod_vocab=mod_vocab,
        gene_names=gene_names,
        rec_target=rec_target,
    )
