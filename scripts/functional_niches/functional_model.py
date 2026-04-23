"""
SpatialFunctionalNicheModel
===========================

Goal: embed each cell so that cells close to Tfh (GC Light Zone) are
separated from cells far from Tfh (GC Dark Zone) despite both being
B_germinal_center — i.e. capture *functional* microenvironment, not just
cell identity.

Architecture
------------

    beta_X  [N, G]  ──►  BetaMLP(G→D)  ──┐
                                           ├──► cat [N, 2D]
    spat_X  [N, S]  ──►  SpatMLP(S→D)  ──┘
                             │
                          Linear(2D→D)  → z_raw  [N, D]
                             │
                          SpatialGCN(k-hop)  → z  [N, D]

Training objectives
-------------------

1. L_triplet  — spatial triplet: spatially adjacent cells should be more
                similar in embedding than distant cells (same as before)

2. L_rec      — reconstruct mean|β| per gene (gene-activity regression,
                ensures beta signal is preserved in the embedding)

3. L_smooth   — spatial smoothness on z (explicit spatial coherence)

4. L_tfh_rank — **new** ranking loss on Tfh proximity:
                For each GC B cell, pairs it with a GC B cell that is
                *closer* to Tfh (positive) and one that is *farther* (negative).
                The embedding should order cells by their Tfh proximity.

                L = mean softplus( cos_sim(z_far, z_anchor) -
                                   cos_sim(z_near, z_anchor) + margin )

                This is the signal that directly drives DZ/LZ/IZ separation.

5. L_nbr_comp — **new** neighbourhood composition regression:
                Predict the local fraction of Tfh among k=30 spatial neighbours
                from z. A simple linear head. Forces the embedding to encode
                how many Tfh are nearby — the core functional variable.

                L = MSE(linear(z), tfh_fraction_per_cell)

The final embedding z comes from the GCN.  Leiden on z recovers niches.
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

from .losses import build_adj_mask, spatial_smoothness_loss

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Model components
# ──────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int = 2, dropout: float = 0.1):
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


class _GCN(nn.Module):
    def __init__(self, hidden_dim: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x + F.gelu(conv(x, edge_index, edge_weight=edge_weight)))
            x = self.drop(x)
        return x


class SpatialFunctionalModel(nn.Module):
    """
    Dual-stream encoder:
      Stream 1: regulatory (beta gene-activity) → BetaMLP
      Stream 2: spatial composition features     → SpatMLP
    Fused → linear → SpatialGCN → z

    Auxiliary heads (used only during training):
      rec_head      : reconstruct rec_target (gene activity regulariser)
      nbr_comp_head : predict local Tfh fraction (functional auxiliary)
    """

    def __init__(
        self,
        beta_dim:    int,
        spat_dim:    int,
        hidden_dim:  int = 64,
        mlp_layers:  int = 2,
        gcn_layers:  int = 2,
        dropout:     float = 0.1,
        rec_dim:     int = 1,
    ):
        super().__init__()
        self.beta_enc = _MLP(beta_dim, hidden_dim, mlp_layers, dropout)
        self.spat_enc = _MLP(spat_dim, hidden_dim, mlp_layers, dropout)
        self.fusion   = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gnn      = _GCN(hidden_dim, gcn_layers, dropout)
        self.rec_head      = nn.Linear(hidden_dim, rec_dim)
        self.nbr_comp_head = nn.Linear(hidden_dim, 1)   # predict local Tfh fraction

    def forward(
        self,
        beta_x: Tensor,
        spat_x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns
        -------
        z_raw : [N, D]  pre-GNN fused embedding
        z     : [N, D]  GNN-smoothed embedding
        x_rec : [N, rec_dim]  reconstruction
        """
        h_beta = self.beta_enc(beta_x)          # [N, D]
        h_spat = self.spat_enc(spat_x)          # [N, D]
        z_raw  = self.fusion(torch.cat([h_beta, h_spat], dim=-1))  # [N, D]
        z      = self.gnn(z_raw, edge_index, edge_weight)
        x_rec  = self.rec_head(z)
        return z_raw, z, x_rec


# ──────────────────────────────────────────────────────────────────
# Contrastive losses
# ──────────────────────────────────────────────────────────────────

class TripletSpatialLoss(nn.Module):
    """Standard spatial triplet: neighbours closer than non-neighbours."""

    def __init__(self, hidden_dim: int, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, z: Tensor, adj: Tensor) -> Tensor:
        N = z.size(0)
        h = F.normalize(self.proj(z), dim=-1)
        is_nbr = adj.bool()
        eye    = torch.eye(N, dtype=torch.bool, device=z.device)

        pos_w  = torch.where(is_nbr,
                              torch.ones(N, N, device=z.device),
                              torch.full((N, N), -1e9, device=z.device))
        pos_idx = pos_w.softmax(dim=-1).multinomial(1).squeeze(1)

        neg_w  = torch.where(~is_nbr & ~eye,
                              torch.ones(N, N, device=z.device),
                              torch.full((N, N), -1e9, device=z.device))
        neg_idx = neg_w.softmax(dim=-1).multinomial(1).squeeze(1)

        d_pos = 1.0 - (h * h[pos_idx]).sum(dim=-1)
        d_neg = 1.0 - (h * h[neg_idx]).sum(dim=-1)
        return F.softplus(d_pos - d_neg + self.margin).mean()


class TfhRankingLoss(nn.Module):
    """
    Ranking loss: cells closer to Tfh should have embeddings that are
    more similar to other 'near-Tfh' cells than to 'far-Tfh' cells.

    Implementation: for each GC B cell anchor, sample:
      positive = a GC B cell that is strictly closer to Tfh
      negative = a GC B cell that is strictly farther from Tfh
    Then apply a triplet margin loss.

    Only applied to GC B cells (gc_mask), since that is the population
    that moves through DZ/LZ/IZ based on Tfh proximity.
    """

    def __init__(self, hidden_dim: int, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        z:           Tensor,     # [N, D] all cells
        tfh_dist:    Tensor,     # [N] distance to nearest Tfh
        gc_mask:     Tensor,     # [N] bool — GC B cells
    ) -> Tensor:
        gc_idx   = gc_mask.nonzero(as_tuple=True)[0]   # [n_gc]
        n_gc     = gc_idx.size(0)
        if n_gc < 4:
            return z.new_zeros(())

        h       = F.normalize(self.proj(z), dim=-1)
        h_gc    = h[gc_idx]                             # [n_gc, D]
        d_gc    = tfh_dist[gc_idx]                      # [n_gc]

        # Pairwise: d_gc_i < d_gc_j  → i is 'nearer to Tfh' (positive for j)
        # For each anchor, sample one nearer (positive) and one farther (negative)
        perm    = torch.randperm(n_gc, device=z.device)
        d_perm  = d_gc[perm]
        h_perm  = h_gc[perm]

        anchor_d = d_gc
        other_d  = d_perm
        other_h  = h_perm

        # Where other_d < anchor_d: other cell is *closer* to Tfh — use as positive
        # Where other_d > anchor_d: other cell is *farther* — use as negative
        # For cells where the random pair has no clear ordering, skip with zero loss
        near_mask = (other_d < anchor_d - 10.0)   # at least 10 units closer
        far_mask  = (other_d > anchor_d + 10.0)

        if near_mask.sum() < 2:
            return z.new_zeros(())

        # Use the near/far pairs as (anchor, pos=near, neg=far)
        # But near_mask and far_mask may not coincide: use near_mask for pairs
        # Pair each anchor that has a near partner with a far partner (cyclic)
        far_idx  = far_mask.nonzero(as_tuple=True)[0]
        near_idx = near_mask.nonzero(as_tuple=True)[0]
        if far_idx.size(0) == 0 or near_idx.size(0) == 0:
            return z.new_zeros(())

        # Anchors = cells with a 'near' partner; positives = those near cells;
        # negatives = random far cells
        anchors  = h_gc[near_idx]
        positives= other_h[near_idx]
        neg_sel  = far_idx[torch.randint(far_idx.size(0), (near_idx.size(0),),
                                          device=z.device)]
        negatives= h_gc[neg_sel]

        d_pos = 1.0 - (anchors * positives).sum(dim=-1)
        d_neg = 1.0 - (anchors * negatives).sum(dim=-1)
        return F.softplus(d_pos - d_neg + self.margin).mean()


# ──────────────────────────────────────────────────────────────────
# Spatial feature builder
# ──────────────────────────────────────────────────────────────────

def build_spatial_features(
    spatial_coords: np.ndarray,
    cell_type:      np.ndarray,
    ks:             tuple[int, ...] = (10, 30, 60),
    tfh_k:          int = 10,
    rbf_sigma:      float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build spatial composition features:
      - local cell-type fraction at k=10,30,60 spatial neighbours   [N, 3×n_ct]
      - RBF-encoded distance to each of the k_tfh nearest Tfh cells [N, tfh_k]

    Returns
    -------
    spat_X   : [N, 3*n_ct + tfh_k]  spatial feature matrix
    tfh_dist : [N]  mean distance to tfh_k nearest Tfh cells (for ranking loss)
    """
    from sklearn.neighbors import NearestNeighbors

    N        = len(spatial_coords)
    ct_types = sorted(set(cell_type))
    n_ct     = len(ct_types)

    feats = []
    for k in ks:
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(spatial_coords)
        _, idxs = nbrs.kneighbors(spatial_coords)
        idxs = idxs[:, 1:]
        comp = np.zeros((N, n_ct), dtype=np.float32)
        for i, ct in enumerate(ct_types):
            flag       = (cell_type == ct).astype(float)
            comp[:, i] = flag[idxs].mean(axis=1)
        feats.append(comp)

    # Tfh proximity
    tfh_mask = cell_type == "T_follicular_helper"
    if tfh_mask.sum() == 0:
        tfh_feat = np.zeros((N, tfh_k), dtype=np.float32)
        tfh_dist_arr = np.zeros(N, dtype=np.float32)
    else:
        nbrs_tfh = NearestNeighbors(n_neighbors=min(tfh_k, tfh_mask.sum())).fit(
            spatial_coords[tfh_mask]
        )
        d_tfh, _ = nbrs_tfh.kneighbors(spatial_coords)
        tfh_feat     = np.exp(-d_tfh / rbf_sigma).astype(np.float32)
        tfh_dist_arr = d_tfh.mean(axis=1).astype(np.float32)

    spat_X = np.concatenate(feats + [tfh_feat], axis=1)
    log.info(f"  Spatial features: {spat_X.shape}  "
             f"(3×{n_ct} comp + {tfh_k} Tfh RBF)")
    return spat_X, tfh_dist_arr


# ──────────────────────────────────────────────────────────────────
# Training function
# ──────────────────────────────────────────────────────────────────

def train_functional(
    beta_X:      np.ndarray,    # [N, G]  gene-activity (or other beta repr)
    spat_X:      np.ndarray,    # [N, S]  spatial composition features
    rec_target:  np.ndarray,    # [N, G]  reconstruction target (same as beta_X)
    tfh_dist:    np.ndarray,    # [N]     mean Tfh distance per cell
    gc_mask:     np.ndarray,    # [N]     bool — GC B cells
    edge_index:  "torch.LongTensor",
    edge_weight: "torch.FloatTensor",
    cell_ids:    list[str],
    output_dir:  str,
    hidden_dim:  int   = 64,
    mlp_layers:  int   = 2,
    gcn_layers:  int   = 2,
    dropout:     float = 0.1,
    epochs:      int   = 800,
    lr:          float = 1e-3,
    # loss weights
    w_triplet:   float = 1.0,
    w_rec:       float = 0.05,
    w_smooth:    float = 0.3,
    w_tfh_rank:  float = 2.0,   # strong signal for DZ/LZ separation
    w_nbr_comp:  float = 0.5,
    device_str:  str   = "auto",
    log_every:   int   = 100,
) -> np.ndarray:
    """
    Train SpatialFunctionalModel and return per-cell embeddings [N, D].

    Key training signals:
      - L_triplet  : neighbours similar in embedding space
      - L_tfh_rank : GC B cells ordered by Tfh proximity in embedding
      - L_nbr_comp : embedding predicts local Tfh fraction (soft supervision)
      - L_rec      : regulariser, reconstruct gene-activity from z
      - L_smooth   : spatial coherence
    """
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if device_str == "auto" else torch.device(device_str))
    log.info(f"Training SpatialFunctionalModel on {device}")

    N       = len(cell_ids)
    beta_t  = torch.from_numpy(F.normalize(torch.from_numpy(beta_X).float(), dim=1).numpy()).to(device)
    spat_t  = torch.from_numpy(spat_X).float().to(device)
    # Normalise spatial features per-column
    spat_mu = spat_t.mean(0, keepdim=True)
    spat_sd = spat_t.std(0, keepdim=True).clamp(min=1e-6)
    spat_t  = (spat_t - spat_mu) / spat_sd

    rec_t   = torch.from_numpy(rec_target).float().to(device)
    tfh_t   = torch.from_numpy(tfh_dist).float().to(device)
    gc_t    = torch.from_numpy(gc_mask).bool().to(device)

    # Local Tfh fraction: fraction of 30-nearest neighbours that are Tfh
    # Extract from spat_X — Tfh is one of the 13 cell types at k=30 (block 1)
    # Just use RBF-encoded tfh proximity as soft supervision target
    tfh_frac = (spat_t[:, -10:].mean(dim=1, keepdim=True))  # mean of RBF Tfh features

    ei = edge_index.to(device)
    ew = edge_weight.to(device)

    model = SpatialFunctionalModel(
        beta_dim=beta_X.shape[1],
        spat_dim=spat_X.shape[1],
        hidden_dim=hidden_dim,
        mlp_layers=mlp_layers,
        gcn_layers=gcn_layers,
        dropout=dropout,
        rec_dim=rec_target.shape[1],
    ).to(device)

    trip_loss = TripletSpatialLoss(hidden_dim).to(device)
    tfh_loss  = TfhRankingLoss(hidden_dim).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(trip_loss.parameters()) + list(tfh_loss.parameters()),
        lr=lr, weight_decay=1e-5,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 20)

    adj = build_adj_mask(ei, N)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train(); trip_loss.train(); tfh_loss.train()
        optimizer.zero_grad()

        z_raw, z, x_rec = model(beta_t, spat_t, ei, ew)

        l_trip   = trip_loss(z, adj)
        l_rec    = F.mse_loss(x_rec, rec_t)
        l_smooth = spatial_smoothness_loss(z, ei, ew)
        l_tfh    = tfh_loss(z, tfh_t, gc_t)
        # Predict local Tfh fraction from embedding (soft supervision)
        l_nbr    = F.mse_loss(model.nbr_comp_head(z), tfh_frac)

        loss = (w_triplet * l_trip  + w_rec     * l_rec   +
                w_smooth  * l_smooth + w_tfh_rank * l_tfh  +
                w_nbr_comp * l_nbr)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % log_every == 0 or epoch == 1:
            log.info(
                f"Epoch {epoch:4d}/{epochs} | "
                f"trip={l_trip.item():.3f}  tfh={l_tfh.item():.3f}  "
                f"nbr={l_nbr.item():.3f}  rec={l_rec.item():.3f}  "
                f"smooth={l_smooth.item():.3f} | "
                f"{time.time()-t0:.1f}s"
            )
            history.append({
                "epoch": epoch, "loss": loss.item(),
                "triplet": l_trip.item(), "tfh_rank": l_tfh.item(),
                "nbr_comp": l_nbr.item(), "rec": l_rec.item(),
                "smooth": l_smooth.item(),
            })

    with open(out / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save({
        "model_state":    model.state_dict(),
        "trip_state":     trip_loss.state_dict(),
        "tfh_state":      tfh_loss.state_dict(),
        "spat_mu":        spat_mu.cpu(),
        "spat_sd":        spat_sd.cpu(),
    }, out / "model.pt")

    model.eval()
    with torch.no_grad():
        _, z_final, _ = model(beta_t, spat_t, ei, ew)
    z_np = z_final.cpu().numpy()

    emb_df = pd.DataFrame(z_np, columns=[f"z_{i}" for i in range(z_np.shape[1])])
    emb_df.insert(0, "CellID", cell_ids)
    emb_df.to_parquet(out / "cell_embeddings.parquet", index=False)
    log.info(f"Saved embeddings → {out / 'cell_embeddings.parquet'}")
    return z_np
