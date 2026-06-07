"""Pretrain shared TissueVisionEncoder on train-half spatial context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import SpatialCache, build_spatial_cache, cluster_maps_for_cells, write_json
from models import PretrainHead, TissueVisionEncoder


def pretrain(
    cache: SpatialCache,
    out_dir: Path,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 4e-4,
    n_pretrain_genes: int = 64,
    variant: str = "base",
    device: str = "cpu",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    maps = cluster_maps_for_cells(cache)
    sf = cache.spatial_features
    expr = cache.expr_log1p

    rng = np.random.default_rng(0)
    gene_idx = rng.choice(expr.shape[1], size=min(n_pretrain_genes, expr.shape[1]), replace=False)
    y = expr[:, gene_idx]

    x_maps = torch.from_numpy(maps)
    x_sf = torch.from_numpy(sf)
    y_t = torch.from_numpy(y)

    ds = TensorDataset(x_maps, x_sf, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=len(ds) > batch_size)

    encoder = TissueVisionEncoder(
        n_clusters=cache.num_clusters,
        in_channels=1,
        variant=variant,  # type: ignore[arg-type]
    ).to(dev)
    head = PretrainHead(encoder.feature_dim, len(gene_idx)).to(dev)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-5,
    )
    loss_fn = nn.MSELoss()

    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        encoder.train()
        head.train()
        total = 0.0
        n = 0
        for bm, bsf, by in dl:
            bm = bm.to(dev)
            bsf = bsf.to(dev)
            by = by.to(dev)
            opt.zero_grad()
            feat = encoder(bm, bsf)
            pred = head(feat)
            loss = loss_fn(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 3.0)
            opt.step()
            total += float(loss.item()) * bm.size(0)
            n += bm.size(0)
        history.append({"epoch": epoch, "mse": total / max(n, 1)})

    ckpt = out_dir / f"tissue_encoder_{variant}.pt"
    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "variant": variant,
            "n_clusters": cache.num_clusters,
            "gene_idx": gene_idx.tolist(),
            "history": history,
        },
        ckpt,
    )
    meta = {
        "epochs": epochs,
        "n_pretrain_genes": int(len(gene_idx)),
        "final_mse": history[-1]["mse"] if history else None,
        "variant": variant,
        "checkpoint": str(ckpt),
    }
    write_json(out_dir / f"pretrain_{variant}_meta.json", meta)
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", type=Path, default=Path("data/tonsil_train.h5ad"))
    p.add_argument("--cache", type=Path, default=Path("outputs/train_cache.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/pretrain"))
    p.add_argument("--spatial-dim", type=int, default=16)
    p.add_argument("--radius", type=float, default=300.0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--variant", choices=["base", "deep"], default="base")
    p.add_argument("--force-genes", default="", help="Comma-separated genes to keep through HVG")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    force = [g.strip() for g in args.force_genes.split(",") if g.strip()]
    if args.cache.exists() and not force:
        cache = SpatialCache.load(args.cache)
    else:
        if args.cache.exists():
            args.cache.unlink()
        cache = build_spatial_cache(
            args.h5ad,
            spatial_dim=args.spatial_dim,
            radius=args.radius,
            force_genes=force or None,
        )
        cache.save(args.cache)

    meta = pretrain(
        cache,
        args.out_dir,
        epochs=args.epochs,
        variant=args.variant,
        device=args.device,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
