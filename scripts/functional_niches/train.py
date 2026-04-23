"""Training loop and CLI for functional microniche embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

from .dataset import FunctionalNicheDataset, load_dataset
from .losses import DGILoss, build_adj_mask, total_loss
from .model import FunctionalNicheModel

log = logging.getLogger(__name__)


def train(
    dataset: FunctionalNicheDataset,
    output_dir: str,
    hidden_dim: int = 64,
    embed_dim: int = 32,
    gene_embed_dim: int = 16,
    n_heads: int = 4,
    gnn_layers: int = 2,
    gnn_heads: int = 4,
    dropout: float = 0.1,
    epochs: int = 600,
    lr: float = 1e-3,
    alpha: float = 0.1,
    beta: float = 0.1,
    device_str: str = "auto",
    log_every: int = 50,
) -> np.ndarray:
    """
    Full-batch training on the spatial graph.

    Returns
    -------
    z : [N, hidden_dim] numpy array of cell embeddings
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    log.info(f"Training on {device}")

    n_mods = len(dataset.mod_vocab)
    n_genes = len(dataset.gene_names)
    n_cells = len(dataset.cell_ids)

    model = FunctionalNicheModel(
        n_modulators=n_mods,
        n_genes=n_genes,
        n_mods_total=n_mods,
        embed_dim=embed_dim,
        mod_hidden=hidden_dim,
        gene_embed_dim=gene_embed_dim,
        cell_dim=hidden_dim,
        n_heads=n_heads,
        gnn_layers=gnn_layers,
        gnn_heads=gnn_heads,
        dropout=dropout,
    ).to(device)

    dgi_loss_fn = DGILoss(hidden_dim).to(device)
    optimizer = optim.Adam(
        list(model.parameters()) + list(dgi_loss_fn.parameters()),
        lr=lr,
    )

    edge_index = dataset.edge_index.to(device)
    edge_weight = dataset.edge_weight.to(device)
    rec_target = dataset.rec_target.to(device)

    # Binary adjacency for InfoNCE (1 if neighbour, 0 otherwise)
    adj_mask = build_adj_mask(edge_index, n_cells).to(device)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        dgi_loss_fn.train()
        optimizer.zero_grad()

        z_raw_pos, z_gnn, x_rec = model(dataset.gene_betas, edge_index, edge_weight, device)

        perm = torch.randperm(n_cells, device=device)
        gene_betas_shuffled = _shuffle_gene_betas(dataset.gene_betas, perm, device)
        z_raw_neg, _, _ = model(gene_betas_shuffled, edge_index, edge_weight, device)

        loss, components = total_loss(
            z_raw_pos=z_raw_pos,
            z_raw_neg=z_raw_neg,
            z_gnn=z_gnn,
            x_rec=x_rec,
            x_target=rec_target,
            edge_index=edge_index,
            edge_weight=edge_weight,
            dgi_loss_fn=dgi_loss_fn,
            adj_mask=adj_mask,
            alpha=alpha,
            beta=beta,
        )

        loss.backward()
        optimizer.step()

        if epoch % log_every == 0 or epoch == 1:
            elapsed = time.time() - t0
            log.info(
                f"Epoch {epoch:4d}/{epochs} | loss={loss.item():.4f} "
                f"dgi={components['dgi']:.4f} rec={components['rec']:.4f} "
                f"spatial={components['spatial']:.4f} | {elapsed:.1f}s"
            )
            history.append({"epoch": epoch, "loss": loss.item(), **components})

    with open(output_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    torch.save(
        {
            "model_state": model.state_dict(),
            "dgi_state": dgi_loss_fn.state_dict(),
            "config": {
                "n_mods": n_mods,
                "n_genes": n_genes,
                "n_mods_total": n_mods,
                "hidden_dim": hidden_dim,
                "embed_dim": embed_dim,
                "gene_embed_dim": gene_embed_dim,
                "cell_dim": hidden_dim,
                "n_heads": n_heads,
                "gnn_layers": gnn_layers,
                "gnn_heads": gnn_heads,
                "dropout": dropout,
            },
        },
        output_path / "model.pt",
    )

    model.eval()
    with torch.no_grad():
        _, z_final, _ = model(dataset.gene_betas, edge_index, edge_weight, device)
    z_np = z_final.cpu().numpy()

    import pandas as pd
    emb_df = pd.DataFrame(
        z_np,
        columns=[f"z_{i}" for i in range(z_np.shape[1])],
    )
    emb_df.insert(0, "CellID", dataset.cell_ids)
    emb_df.to_parquet(output_path / "cell_embeddings.parquet", index=False)

    log.info(f"Saved embeddings to {output_path / 'cell_embeddings.parquet'}")
    return z_np


def _shuffle_gene_betas(gene_betas, perm, device):
    """Return a list of GeneBetadata with rows shuffled according to perm."""
    from .dataset import GeneBetadata
    cpu_perm = perm.cpu()
    shuffled = []
    for gb in gene_betas:
        shuffled.append(GeneBetadata(
            gene_name=gb.gene_name,
            gene_index=gb.gene_index,
            mod_indices=gb.mod_indices[cpu_perm].to(device),
            beta_values=gb.beta_values[cpu_perm].to(device),
            n_mods=gb.n_mods,
        ))
    return shuffled


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train Functional Microniche Embeddings")
    parser.add_argument("--feather-dir", required=True)
    parser.add_argument("--h5ad", required=True, help="AnnData h5ad with obsm['spatial']")
    parser.add_argument("--spatial-k", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta-loss", type=float, default=0.5)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import anndata
    adata = anndata.read_h5ad(args.h5ad)
    spatial_coords = adata.obsm["spatial"].astype(np.float32)
    cell_ids = list(adata.obs_names)

    dataset = load_dataset(
        feather_dir=args.feather_dir,
        spatial_coords=spatial_coords,
        cell_ids=cell_ids,
        k=args.spatial_k,
    )

    train(
        dataset=dataset,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        lr=args.lr,
        alpha=args.alpha,
        beta=args.beta_loss,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
