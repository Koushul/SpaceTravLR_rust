"""
Synthetic data generator for benchmarking functional microniche embeddings.

Design
------
We simulate a tissue section laid out on a 2D grid.  Cells are assigned to
K spatially contiguous niches (blobs) with KNOWN ground-truth labels.  Each
niche has a characteristic regulatory program: a sparse set of 'active'
modulators with niche-specific mean beta values.  Noise and Lasso-style
sparsity are added to make the problem realistic.

The synthetic dataset is returned as a `FunctionalNicheDataset` together
with the ground-truth niche labels so downstream code can compute ARI / NMI.

Key parameters
--------------
n_cells         : total number of simulated cells  (e.g. 2000)
n_genes         : number of target genes           (e.g. 10)
n_mods_shared   : modulators shared across all genes (e.g. 200)
n_mods_gene     : extra gene-specific modulators    (e.g. 20)
n_niches        : number of ground-truth niches     (e.g. 5)
n_active_mods   : active modulators per niche       (e.g. 30)
beta_signal     : mean absolute beta for active mods (e.g. 1.5)
beta_noise      : std of background noise           (e.g. 0.1)
sparsity        : fraction of mods forced to zero   (e.g. 0.7)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .dataset import FunctionalNicheDataset, GeneBetadata, build_spatial_graph


@dataclass
class SyntheticDataset:
    dataset: FunctionalNicheDataset
    true_labels: np.ndarray       # [N] int — ground-truth niche IDs
    spatial_coords: np.ndarray    # [N, 2]
    mod_vocab: dict[str, int]


def _make_spatial_grid(n_cells: int, seed: int = 42) -> np.ndarray:
    """Arrange cells on an approximately square grid with small jitter."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n_cells)))
    xs, ys = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1)[:n_cells].astype(np.float32)
    coords += rng.normal(0, 0.1, size=coords.shape).astype(np.float32)
    return coords


def _assign_spatial_niches(
    coords: np.ndarray,
    n_niches: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Assign each cell to the nearest of K randomly placed niche centres.
    This produces spatially contiguous blobs (Voronoi-like regions).
    """
    rng = np.random.default_rng(seed)
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    centres = rng.uniform(
        [x_min, y_min], [x_max, y_max], size=(n_niches, 2)
    ).astype(np.float32)

    dists = np.linalg.norm(coords[:, None, :] - centres[None, :, :], axis=-1)
    return dists.argmin(axis=1).astype(int)


def _make_niche_programs(
    n_niches: int,
    n_mods_total: int,
    n_active_per_niche: int,
    beta_signal: float,
    seed: int = 1,
    gene_specific: bool = False,
    n_genes: int = 1,
) -> np.ndarray:
    """
    Build niche regulatory programs.

    Parameters
    ----------
    gene_specific : if True, return [n_niches, n_genes, n_mods_total] — each
        gene has its own per-niche program (harder for PCA, easier for model).
        if False, return [n_niches, n_mods_total] (shared across genes).
    """
    rng = np.random.default_rng(seed)

    if gene_specific:
        programs = np.zeros((n_niches, n_genes, n_mods_total), dtype=np.float32)
        for k in range(n_niches):
            for g in range(n_genes):
                # Each (niche, gene) pair activates a distinct random subset
                active = rng.choice(n_mods_total, size=n_active_per_niche, replace=False)
                signs = rng.choice([-1, 1], size=len(active))
                programs[k, g, active] = signs * beta_signal
    else:
        programs = np.zeros((n_niches, n_mods_total), dtype=np.float32)
        for k in range(n_niches):
            block_start = (k * n_active_per_niche) % n_mods_total
            active = np.arange(block_start, block_start + n_active_per_niche) % n_mods_total
            signs = rng.choice([-1, 1], size=len(active))
            programs[k, active] = signs * beta_signal

    return programs


def make_synthetic_dataset(
    n_cells: int = 2000,
    n_genes: int = 10,
    n_mods_shared: int = 200,
    n_mods_gene: int = 20,
    n_niches: int = 5,
    n_active_mods: int = 30,
    beta_signal: float = 1.5,
    beta_noise: float = 0.15,
    sparsity: float = 0.70,
    spatial_k: int = 6,
    seed: int = 42,
    gene_specific_programs: bool = False,
    cell_noise_scale: float = 0.0,
) -> SyntheticDataset:
    """
    Generate a synthetic FunctionalNicheDataset with known ground-truth labels.

    Each niche has a distinct sparse regulatory signature.  A fraction
    `sparsity` of all beta values is forced to zero (mimicking Lasso output).

    Parameters
    ----------
    n_cells             : number of simulated cells
    n_genes             : number of target genes
    n_mods_shared       : modulators shared across all genes
    n_mods_gene         : extra gene-specific modulators (varies per gene)
    n_niches            : number of ground-truth functional niches
    n_active_mods       : modulators actively up/down-regulated per niche
    beta_signal         : mean |beta| for active regulators in each niche
    beta_noise          : std of Gaussian noise added to all betas
    sparsity            : fraction of betas zeroed (Lasso sparsity)
    spatial_k           : kNN for the spatial graph
    seed                : random seed
    gene_specific_programs : if True, each gene has its own niche program
    cell_noise_scale    : additional per-cell iid noise on top of beta_noise
                          (simulates measurement noise; high values make
                          individual cells ambiguous, requiring spatial GNN)

    Returns
    -------
    SyntheticDataset with .dataset, .true_labels, .spatial_coords
    """
    rng = np.random.default_rng(seed)

    # --- spatial layout ---
    coords = _make_spatial_grid(n_cells, seed=seed)
    true_labels = _assign_spatial_niches(coords, n_niches, seed=seed)

    # --- modulator vocabulary ---
    shared_mods = [f"beta_shared_{i:04d}" for i in range(n_mods_shared)]
    # gene-specific mods vary in count (mimicking the real 971–1037 range)
    gene_specific_mods_per_gene = [
        [f"beta_gene{g}_{i:03d}" for i in range(n_mods_gene + rng.integers(-5, 6))]
        for g in range(n_genes)
    ]
    all_mods = sorted(set(shared_mods + [
        m for gm in gene_specific_mods_per_gene for m in gm
    ]))
    mod_vocab = {m: i for i, m in enumerate(all_mods)}
    n_mods_total = len(mod_vocab)

    # --- niche regulatory programs on the global mod space ---
    programs = _make_niche_programs(
        n_niches, n_mods_total, n_active_mods, beta_signal, seed=seed + 1,
        gene_specific=gene_specific_programs, n_genes=n_genes,
    )

    cell_ids = [f"cell_{i:05d}" for i in range(n_cells)]

    gene_betas: list[GeneBetadata] = []
    for g in range(n_genes):
        gene_mods = shared_mods + gene_specific_mods_per_gene[g]
        gene_mod_idx = np.array([mod_vocab[m] for m in gene_mods], dtype=np.int64)
        M_g = len(gene_mod_idx)

        # mean beta from niche program, subsetted to this gene's mods
        if gene_specific_programs:
            # programs shape: [n_niches, n_genes, n_mods_total]
            niche_means = programs[true_labels, g][:, gene_mod_idx]  # [N, M_g]
        else:
            # programs shape: [n_niches, n_mods_total]
            niche_means = programs[true_labels][:, gene_mod_idx]      # [N, M_g]

        # add Gaussian noise
        betas = niche_means + rng.normal(0, beta_noise, size=(n_cells, M_g)).astype(np.float32)

        # optional additional per-cell noise (makes individual cells ambiguous)
        if cell_noise_scale > 0:
            betas += rng.normal(0, cell_noise_scale, size=(n_cells, M_g)).astype(np.float32)

        # apply Lasso-style sparsity: zero background entries only
        is_background = niche_means == 0.0
        zero_mask = rng.random(size=(n_cells, M_g)) < sparsity
        betas = np.where(is_background & zero_mask, 0.0, betas).astype(np.float32)

        mod_indices_mat = np.broadcast_to(
            gene_mod_idx[None, :], (n_cells, M_g)
        ).copy().astype(np.int64)

        gene_betas.append(GeneBetadata(
            gene_name=f"gene_{g:03d}",
            gene_index=g,
            mod_indices=torch.from_numpy(mod_indices_mat).long(),
            beta_values=torch.from_numpy(betas).float(),
            n_mods=M_g,
        ))

    edge_index, edge_weight = build_spatial_graph(coords, k=spatial_k)

    # reconstruction target: mean |beta| across genes per modulator
    from .dataset import _build_rec_target
    rec_target = _build_rec_target(gene_betas, n_cells, n_mods_total)

    dataset = FunctionalNicheDataset(
        cell_ids=cell_ids,
        gene_betas=gene_betas,
        edge_index=edge_index,
        edge_weight=edge_weight,
        mod_vocab=mod_vocab,
        gene_names=[f"gene_{g:03d}" for g in range(n_genes)],
        rec_target=rec_target,
    )

    return SyntheticDataset(
        dataset=dataset,
        true_labels=true_labels,
        spatial_coords=coords,
        mod_vocab=mod_vocab,
    )
