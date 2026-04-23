"""Training losses: spatial InfoNCE contrastive, reconstruction, spatial smoothness."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DGILoss(nn.Module):
    """
    Spatial InfoNCE contrastive loss (replaces standard DGI).

    For each cell i, treat its spatial neighbours as POSITIVE samples and all
    other cells as NEGATIVES.  Uses temperature-scaled cosine similarity.

    This directly trains the GNN to produce embeddings where spatial neighbours
    are more similar than distant cells — the core property needed for niche
    discovery.

    L = -mean_i( mean_{j in N(i)} log softmax(sim(z_i, z_j) / tau)[j] )
    """

    def __init__(self, hidden_dim: int, tau: float = 0.5):
        super().__init__()
        self.tau = tau
        # projection head (MLP) to decouple contrastive space from embedding space
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        z_raw_pos: Tensor,
        z_raw_neg: Tensor,
        z_gnn: Tensor,
        adj_mask: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        z_raw_pos : [N, D] pre-GNN encoder outputs (real)
        z_raw_neg : [N, D] pre-GNN encoder outputs (shuffled, unused but kept for API)
        z_gnn     : [N, D] GNN-smoothed embeddings — THESE are used for contrast
        adj_mask  : [N, N] normalised adjacency with 1s for neighbours

        Returns
        -------
        scalar InfoNCE loss
        """
        N = z_gnn.size(0)

        # Project embeddings
        h = F.normalize(self.proj(z_gnn), dim=-1)  # [N, D]

        # Pairwise similarity matrix
        sim = torch.mm(h, h.T) / self.tau            # [N, N]

        # Mask out self-loops from positives
        eye = torch.eye(N, device=z_gnn.device).bool()
        sim = sim.masked_fill(eye, float("-inf"))

        # adj_mask[i, j] = 1 if j is a neighbour of i
        # For numerical stability, use log_softmax + gather pattern
        log_prob = F.log_softmax(sim, dim=-1)        # [N, N]

        # Replace -inf (masked diagonal) with 0 BEFORE masking with pos_mask
        # to avoid 0 * -inf = nan
        log_prob = log_prob.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # Mean log-prob over positive (neighbour) pairs
        pos_mask = (adj_mask > 0).float()
        n_pos = pos_mask.sum(dim=-1).clamp(min=1.0)
        loss = -(pos_mask * log_prob).sum(dim=-1) / n_pos
        return loss.mean()


def build_adj_mask(edge_index: Tensor, n_nodes: int) -> Tensor:
    """
    Build a dense row-normalised adjacency matrix from edge_index.
    For large graphs (>10k nodes) this is memory-intensive; caller should
    decide whether to use sparse ops instead.
    """
    src, dst = edge_index
    adj = torch.zeros(n_nodes, n_nodes, device=edge_index.device)
    adj[src, dst] = 1.0
    row_sum = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
    return adj / row_sum


def spatial_smoothness_loss(z: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
    """
    Penalise large differences between spatially adjacent cell embeddings.

    L = mean over edges of (weight_e * ||z_src - z_dst||^2)
    """
    src, dst = edge_index
    diff = z[src] - z[dst]
    return (edge_weight * (diff ** 2).sum(dim=-1)).mean()


def total_loss(
    z_raw_pos: Tensor,
    z_raw_neg: Tensor,
    z_gnn: Tensor,
    x_rec: Tensor,
    x_target: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor,
    dgi_loss_fn: DGILoss,
    adj_mask: Tensor,
    alpha: float = 0.1,
    beta: float = 0.1,
) -> tuple[Tensor, dict[str, float]]:
    """
    Compute the combined training loss.

    Parameters
    ----------
    z_raw_pos : pre-GNN encoder output for real betas
    z_raw_neg : pre-GNN encoder output for shuffled betas
    z_gnn     : GNN-smoothed output (used as global context in DGI)
    x_rec     : reconstructed modulator summary from decoder
    x_target  : target modulator summary
    edge_index, edge_weight : spatial graph
    dgi_loss_fn : DGILoss module
    adj_mask  : normalised adjacency (passed to spatial_smoothness)
    alpha     : reconstruction loss weight
    beta      : spatial smoothness weight

    Returns
    -------
    loss : scalar Tensor
    components : dict with 'dgi', 'rec', 'spatial' values
    """
    l_dgi = dgi_loss_fn(z_raw_pos, z_raw_neg, z_gnn, adj_mask)
    l_rec = F.mse_loss(x_rec, x_target)
    l_spatial = spatial_smoothness_loss(z_gnn, edge_index, edge_weight)

    loss = l_dgi + alpha * l_rec + beta * l_spatial
    return loss, {
        "dgi": l_dgi.item(),
        "rec": l_rec.item(),
        "spatial": l_spatial.item(),
    }
