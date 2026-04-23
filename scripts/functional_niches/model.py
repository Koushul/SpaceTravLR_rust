"""Model components: ModulatorEncoder, CellEncoder, SpatialGNN, FunctionalNicheModel."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import GATv2Conv

from .dataset import GeneBetadata


class ModulatorEncoder(nn.Module):
    """
    Encode a variable-length set of (modulator_id, beta_value) pairs into a
    fixed-size vector via attention-weighted pooling.

    Shared weights across all genes — the modulator embedding gives each
    regulator a unique learnable identity.
    """

    def __init__(self, n_modulators: int, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.mod_embedding = nn.Embedding(n_modulators, embed_dim)
        self.value_proj = nn.Linear(1, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_query = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, mod_indices: Tensor, beta_values: Tensor) -> Tensor:
        """
        Parameters
        ----------
        mod_indices : [B, M] int64
        beta_values : [B, M] float32

        Returns
        -------
        [B, hidden_dim]
        """
        id_emb = self.mod_embedding(mod_indices)              # [B, M, embed_dim]
        val_emb = self.value_proj(beta_values.unsqueeze(-1))  # [B, M, embed_dim]
        x = self.mlp(torch.cat([id_emb, val_emb], dim=-1))   # [B, M, hidden_dim]

        scores = (x * self.attn_query).sum(dim=-1)            # [B, M]
        weights = scores.softmax(dim=-1).unsqueeze(-1)        # [B, M, 1]
        return (weights * x).sum(dim=1)                       # [B, hidden_dim]


class CellEncoder(nn.Module):
    """
    Aggregate G per-gene modulator summaries into a single cell embedding
    using self-attention over gene tokens.
    """

    def __init__(
        self,
        n_modulators: int,
        n_genes: int,
        embed_dim: int = 32,
        mod_hidden: int = 64,
        gene_embed_dim: int = 16,
        cell_dim: int = 64,
        n_heads: int = 4,
    ):
        super().__init__()
        self.mod_encoder = ModulatorEncoder(n_modulators, embed_dim, mod_hidden)
        self.gene_embedding = nn.Embedding(n_genes, gene_embed_dim)
        token_dim = mod_hidden + gene_embed_dim
        # token_dim must be divisible by n_heads
        assert token_dim % n_heads == 0, (
            f"token_dim={token_dim} must be divisible by n_heads={n_heads}"
        )
        self.self_attn = nn.MultiheadAttention(token_dim, n_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(token_dim)
        self.out_proj = nn.Linear(token_dim, cell_dim)

    def forward(self, gene_betas: list[GeneBetadata], device: torch.device) -> Tensor:
        """
        Parameters
        ----------
        gene_betas : list of G GeneBetadata objects

        Returns
        -------
        [N, cell_dim]
        """
        tokens = []
        for gb in gene_betas:
            mi = gb.mod_indices.to(device)
            bv = gb.beta_values.to(device)
            h_g = self.mod_encoder(mi, bv)           # [N, mod_hidden]
            g_idx = torch.full((h_g.size(0),), gb.gene_index, dtype=torch.long, device=device)
            g_emb = self.gene_embedding(g_idx)        # [N, gene_embed_dim]
            tokens.append(torch.cat([h_g, g_emb], dim=-1))  # [N, token_dim]

        tokens = torch.stack(tokens, dim=1)                   # [N, G, token_dim]
        attended, _ = self.self_attn(tokens, tokens, tokens)  # [N, G, token_dim]
        attended = self.layer_norm(attended + tokens)         # residual
        pooled = attended.mean(dim=1)                         # [N, token_dim]
        return self.out_proj(pooled)                          # [N, cell_dim]


class SpatialGNN(nn.Module):
    """Two-layer GATv2 graph neural network for spatial context integration."""

    def __init__(
        self,
        in_dim: int = 64,
        hidden_dim: int = 64,
        n_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            GATv2Conv(
                in_dim if i == 0 else hidden_dim,
                hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                edge_dim=1,
            )
            for i in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index, edge_attr=edge_weight.unsqueeze(-1))
            x = norm(x + x_new)
            x = F.gelu(x)
            x = self.dropout(x)
        return x


class FunctionalNicheModel(nn.Module):
    """Full model: CellEncoder → SpatialGNN → decoder."""

    def __init__(
        self,
        n_modulators: int,
        n_genes: int,
        n_mods_total: int,
        embed_dim: int = 32,
        mod_hidden: int = 64,
        gene_embed_dim: int = 16,
        cell_dim: int = 64,
        n_heads: int = 4,
        gnn_layers: int = 2,
        gnn_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cell_encoder = CellEncoder(
            n_modulators=n_modulators,
            n_genes=n_genes,
            embed_dim=embed_dim,
            mod_hidden=mod_hidden,
            gene_embed_dim=gene_embed_dim,
            cell_dim=cell_dim,
            n_heads=n_heads,
        )
        self.spatial_gnn = SpatialGNN(
            in_dim=cell_dim,
            hidden_dim=cell_dim,
            n_layers=gnn_layers,
            heads=gnn_heads,
            dropout=dropout,
        )
        self.decoder = nn.Linear(cell_dim, n_mods_total)

    def forward(
        self,
        gene_betas: list[GeneBetadata],
        edge_index: Tensor,
        edge_weight: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        z_raw : [N, cell_dim] — pre-GNN cell encoder output
        z     : [N, cell_dim] — GNN-smoothed embeddings
        x_rec : [N, n_mods_total] — reconstructed modulator summary
        """
        z_raw = self.cell_encoder(gene_betas, device)          # [N, cell_dim]
        z = self.spatial_gnn(z_raw, edge_index, edge_weight)   # [N, cell_dim]
        x_rec = self.decoder(z)                                 # [N, n_mods_total]
        return z_raw, z, x_rec

    @torch.no_grad()
    def encode(
        self,
        gene_betas: list[GeneBetadata],
        edge_index: Tensor,
        edge_weight: Tensor,
        device: torch.device,
    ) -> Tensor:
        self.eval()
        _, z, _ = self.forward(gene_betas, edge_index, edge_weight, device)
        return z
