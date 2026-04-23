"""Training losses: spatial InfoNCE contrastive, reconstruction, spatial smoothness."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DGILoss(nn.Module):
    """
    Spatial contrastive loss: neighbor vs. non-neighbor triplets.

    For each cell i:
      - Positive j: a random spatial neighbour (from the kNN graph)
      - Negative k: a random non-neighbour cell

    Loss: softplus(d(z_i, z_j) - d(z_i, z_k) + margin)

    where d = negative cosine similarity.

    This is O(N) per epoch (one triplet per cell), making it practical on CPU.
    The projection head decouples the contrastive space from the embedding space.
    """

    def __init__(self, hidden_dim: int, tau: float = 0.5, margin: float = 0.5):
        super().__init__()
        self.margin = margin
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
        z_raw_neg : [N, D] pre-GNN encoder outputs (shuffled, unused)
        z_gnn     : [N, D] GNN-smoothed embeddings
        adj_mask  : [N, N] normalised adjacency (entries > 0 for neighbours)

        Returns
        -------
        scalar triplet loss
        """
        N = z_gnn.size(0)
        h = F.normalize(self.proj(z_gnn), dim=-1)   # [N, D]

        # For each cell, sample one positive neighbour
        # and one negative non-neighbour
        is_neighbor = (adj_mask > 0)                 # [N, N] bool

        # Random positive: pick one neighbour per anchor
        # Mask non-neighbours with -inf before sampling
        pos_logits = torch.where(is_neighbor, torch.zeros(N, N, device=h.device),
                                 torch.full((N, N), float("-inf"), device=h.device))
        pos_idx = pos_logits.softmax(dim=-1).multinomial(1).squeeze(1)   # [N]

        # Random negative: pick one non-neighbour per anchor
        neg_logits = torch.where(~is_neighbor,
                                 torch.zeros(N, N, device=h.device),
                                 torch.full((N, N), float("-inf"), device=h.device))
        # Mask diagonal too
        eye = torch.eye(N, device=h.device, dtype=torch.bool)
        neg_logits = neg_logits.masked_fill(eye, float("-inf"))
        neg_idx = neg_logits.softmax(dim=-1).multinomial(1).squeeze(1)   # [N]

        h_a = h                          # [N, D] anchors
        h_p = h[pos_idx]                 # [N, D] positives
        h_n = h[neg_idx]                 # [N, D] negatives

        # Cosine distance: 1 - cosine_sim (smaller = more similar)
        d_pos = 1.0 - (h_a * h_p).sum(dim=-1)   # [N]
        d_neg = 1.0 - (h_a * h_n).sum(dim=-1)   # [N]

        # Triplet margin loss: penalise when pos is further than neg - margin
        return F.softplus(d_pos - d_neg + self.margin).mean()


def build_adj_mask(edge_index: Tensor, n_nodes: int) -> Tensor:
    """
    Build a dense binary adjacency matrix from edge_index.
    Entry [i, j] = 1 if j is a spatial neighbour of i.
    """
    src, dst = edge_index
    adj = torch.zeros(n_nodes, n_nodes, device=edge_index.device)
    adj[src, dst] = 1.0
    return adj


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
