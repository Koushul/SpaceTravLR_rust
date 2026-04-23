"""
Simple, fast functional microniche model.

Design
------
Precompute a flat signed-beta matrix  X ∈ ℝ^{N × (G·M)}  ONCE before training
(one block per gene, signed beta values).  Then train:

    z_raw = MLP(X)                    # cell encoder:   [N, D]
    z     = GCN(z_raw, spatial_graph) # spatial context: [N, D]

Training objectives (same as the full model, but on a static input matrix):
  L_triplet : spatial neighbour vs. non-neighbour contrastive
  L_rec     : reconstruct mean|β| summary
  L_smooth  : spatial smoothness on z

Because X is precomputed, every epoch is a single BLAS call (X → MLP) plus one
GCN pass — no Python loops, no per-token embedding lookups.

Typical speed on CPU (2000 cells, 10 genes, 1000 mods):
  Old model: ~8 s/epoch   (per-token embedding + MLP per gene)
  This model: ~0.05 s/epoch  (one matmul per epoch)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from torch_geometric.nn import GCNConv

from .dataset import FunctionalNicheDataset, make_beta_matrix
from .losses import build_adj_mask, spatial_smoothness_loss

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BetaMLP(nn.Module):
    """
    MLP cell encoder operating on the precomputed flat beta matrix.

    Input:  [N, in_dim]  (in_dim = G × n_mods_total for concat, or n_mods_total for sum)
    Output: [N, hidden_dim]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SpatialGCN(nn.Module):
    """Two-layer GCN for spatial context integration."""

    def __init__(self, hidden_dim: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x + F.gelu(conv(x, edge_index, edge_weight=edge_weight)))
            x = self.drop(x)
        return x


class SimpleNicheModel(nn.Module):
    """
    Simple two-stage functional microniche encoder.

    Stage 1 — BetaMLP: maps the flat signed-beta vector to a cell embedding.
    Stage 2 — SpatialGCN: integrates spatial neighbourhood context.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        mlp_layers: int = 2,
        gcn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = BetaMLP(in_dim, hidden_dim, mlp_layers, dropout)
        self.gnn = SpatialGCN(hidden_dim, gcn_layers, dropout)
        self.decoder = nn.Linear(hidden_dim, in_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        z_raw : [N, D] MLP output before GNN
        z     : [N, D] GNN-smoothed embedding
        x_rec : [N, in_dim] reconstruction of input
        """
        z_raw = self.encoder(x)
        z = self.gnn(z_raw, edge_index, edge_weight)
        x_rec = self.decoder(z)
        return z_raw, z, x_rec

    @torch.no_grad()
    def embed(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> Tensor:
        self.eval()
        _, z, _ = self.forward(x, edge_index, edge_weight)
        return z


# ---------------------------------------------------------------------------
# Contrastive loss (triplet, O(N) per epoch)
# ---------------------------------------------------------------------------

class TripletSpatialLoss(nn.Module):
    """
    Spatial triplet loss with a lightweight projection head.

    For each cell i, sample:
      positive j ∈ spatial neighbours
      negative k ∉ spatial neighbours

    L = mean softplus( d(h_i, h_j) - d(h_i, h_k) + margin )
    where d(u,v) = 1 - cos_sim(u, v)  ∈ [0, 2].
    """

    def __init__(self, hidden_dim: int, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, z: Tensor, adj: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z   : [N, D] embeddings
        adj : [N, N] binary adjacency (1 = spatial neighbour)
        """
        N = z.size(0)
        h = F.normalize(self.proj(z), dim=-1)   # [N, D]
        is_nbr = adj.bool()                      # [N, N]

        # Sample positive (neighbour) for each anchor
        pos_w = torch.where(is_nbr, torch.ones(N, N, device=h.device),
                            torch.full((N, N), -1e9, device=h.device))
        pos_idx = pos_w.softmax(dim=-1).multinomial(1).squeeze(1)  # [N]

        # Sample negative (non-neighbour, not self)
        eye = torch.eye(N, dtype=torch.bool, device=h.device)
        neg_w = torch.where(~is_nbr & ~eye, torch.ones(N, N, device=h.device),
                            torch.full((N, N), -1e9, device=h.device))
        neg_idx = neg_w.softmax(dim=-1).multinomial(1).squeeze(1)  # [N]

        d_pos = 1.0 - (h * h[pos_idx]).sum(dim=-1)   # [N]
        d_neg = 1.0 - (h * h[neg_idx]).sum(dim=-1)   # [N]
        return F.softplus(d_pos - d_neg + self.margin).mean()


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_simple(
    dataset: FunctionalNicheDataset,
    output_dir: str,
    hidden_dim: int = 64,
    mlp_layers: int = 2,
    gcn_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    alpha: float = 0.1,    # reconstruction weight
    beta: float = 0.1,     # spatial smoothness weight
    concat_genes: bool = True,
    device_str: str = "auto",
    log_every: int = 50,
) -> np.ndarray:
    """
    Train the SimpleNicheModel on the precomputed beta matrix.

    Parameters
    ----------
    dataset     : FunctionalNicheDataset
    output_dir  : where to save model.pt, cell_embeddings.parquet, history
    hidden_dim  : embedding dimension
    mlp_layers  : depth of the MLP encoder
    gcn_layers  : depth of the spatial GCN
    dropout     : dropout rate
    epochs      : training epochs
    lr          : Adam learning rate
    weight_decay: L2 regularisation
    alpha       : reconstruction loss coefficient
    beta        : spatial smoothness loss coefficient
    concat_genes: if True, concatenate per-gene beta blocks [N, G*M];
                  if False, sum signed betas across genes [N, M]
    device_str  : 'auto', 'cpu', or 'cuda'

    Returns
    -------
    z : [N, hidden_dim] numpy embedding matrix
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if device_str == "auto" else torch.device(device_str)
    log.info(f"Training SimpleNicheModel on {device}")

    n_cells = len(dataset.cell_ids)
    n_mods = len(dataset.mod_vocab)
    n_genes = len(dataset.gene_names)

    # Use precomputed matrix if available (streaming load path), otherwise build it
    if dataset._beta_matrix is not None:
        log.info("Using precomputed beta matrix from dataset …")
        X = dataset._beta_matrix.to(device)
    else:
        log.info("Precomputing beta matrix …")
        X = make_beta_matrix(
            dataset.gene_betas, n_cells, n_mods, concat_genes=concat_genes
        ).to(device)
    in_dim = X.size(1)
    log.info(f"  Input shape: {X.shape}  ({in_dim} features per cell)")

    # L2-normalise each cell's input vector so the model learns regulatory
    # direction and pattern, not absolute regulatory strength.
    # Cells with high global beta magnitude vs low are otherwise hard to
    # separate because the first Linear layer saturates on the magnitude axis.
    X = F.normalize(X, dim=1)

    edge_index = dataset.edge_index.to(device)
    edge_weight = dataset.edge_weight.to(device)
    rec_target = dataset.rec_target.to(device)      # [N, rec_dim]
    rec_dim = rec_target.size(1)

    model = SimpleNicheModel(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        mlp_layers=mlp_layers,
        gcn_layers=gcn_layers,
        dropout=dropout,
    ).to(device)
    # Decoder outputs rec_dim so it always matches rec_target exactly
    model.decoder = nn.Linear(hidden_dim, rec_dim).to(device)

    contrastive = TripletSpatialLoss(hidden_dim).to(device)
    optimizer = optim.Adam(
        list(model.parameters()) + list(contrastive.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

    adj = build_adj_mask(edge_index, n_cells)   # [N, N] binary

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        contrastive.train()
        optimizer.zero_grad()

        z_raw, z, x_rec = model(X, edge_index, edge_weight)

        l_triplet = contrastive(z, adj)
        l_rec = F.mse_loss(x_rec, rec_target)
        l_smooth = spatial_smoothness_loss(z, edge_index, edge_weight)

        loss = l_triplet + alpha * l_rec + beta * l_smooth
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - t0
            log.info(
                f"Epoch {epoch:4d}/{epochs} | loss={loss.item():.4f} "
                f"triplet={l_triplet.item():.4f} rec={l_rec.item():.4f} "
                f"smooth={l_smooth.item():.4f} | {elapsed:.1f}s"
            )
            history.append({
                "epoch": epoch,
                "loss": loss.item(),
                "triplet": l_triplet.item(),
                "rec": l_rec.item(),
                "smooth": l_smooth.item(),
            })

    with open(out / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(
        {
            "model_state": model.state_dict(),
            "contrastive_state": contrastive.state_dict(),
            "config": {
                "in_dim": in_dim,
                "hidden_dim": hidden_dim,
                "mlp_layers": mlp_layers,
                "gcn_layers": gcn_layers,
                "dropout": dropout,
                "concat_genes": concat_genes,
                "n_mods": n_mods,
                "n_genes": n_genes,
            },
        },
        out / "model.pt",
    )

    model.eval()
    with torch.no_grad():
        _, z_final, _ = model(X, edge_index, edge_weight)
    z_np = z_final.cpu().numpy()

    emb_df = pd.DataFrame(
        z_np, columns=[f"z_{i}" for i in range(z_np.shape[1])]
    )
    emb_df.insert(0, "CellID", dataset.cell_ids)
    emb_df.to_parquet(out / "cell_embeddings.parquet", index=False)
    log.info(f"Saved embeddings → {out / 'cell_embeddings.parquet'}")

    return z_np
