"""Fine-tune per-gene MLP heads on frozen tissue CNN; export betadata feathers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from betadata_io import write_betadata_feather
from data_utils import SpatialCache, build_spatial_cache, cluster_maps_for_cells, gene_index, write_json
from group_lasso import fit_gene_lasso_anchors, r2_score
from models import GeneHeadMLP, TissueVisionEncoder, apply_output_activation, linear_readout_y


def load_encoder(ckpt_path: Path, device: torch.device) -> TissueVisionEncoder:
    ckpt = torch.load(ckpt_path, map_location=device)
    variant = ckpt.get("variant", "base")
    n_clusters = int(ckpt["n_clusters"])
    encoder = TissueVisionEncoder(n_clusters=n_clusters, variant=variant)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    return encoder


@torch.no_grad()
def encode_all(
    encoder: TissueVisionEncoder,
    cache: SpatialCache,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    maps = cluster_maps_for_cells(cache)
    sf = cache.spatial_features
    feats: list[np.ndarray] = []
    encoder.eval()
    for start in range(0, maps.shape[0], batch_size):
        end = min(start + batch_size, maps.shape[0])
        bm = torch.from_numpy(maps[start:end]).to(device)
        bsf = torch.from_numpy(sf[start:end]).to(device)
        f = encoder(bm, bsf).cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def finetune_gene(
    gene: str,
    cache: SpatialCache,
    encoder: TissueVisionEncoder,
    features: np.ndarray,
    out_dir: Path,
    device: torch.device,
    epochs: int = 30,
    lr: float = 4e-4,
    batch_size: int = 128,
    l1_reg: float = 1e-4,
    score_threshold: float = 0.1,
    align_weight: float = 0.05,
) -> dict:
    gi = gene_index(cache, gene)
    y = cache.expr_log1p[:, gi].astype(np.float64)
    x_mod = cache.modulators.astype(np.float64)
    clusters = cache.clusters
    n_mod = x_mod.shape[1]

    lasso_fits = fit_gene_lasso_anchors(
        y, x_mod, clusters, cache.num_clusters, l1_reg=l1_reg, score_threshold=score_threshold
    )

    mod_names = [f"beta_{lab}" for lab in cache.cluster_labels]
    beta_columns = ["beta0"] + mod_names
    n_betas = len(beta_columns)

    # Lasso anchor vector per cell (cluster-specific)
    anchor0 = np.zeros(len(y), dtype=np.float32)
    anchor_rest = np.zeros((len(y), n_mod), dtype=np.float32)
    for c, fit in lasso_fits.items():
        mask = clusters == c
        anchor0[mask] = fit.intercept
        anchor_rest[mask] = fit.coef

    anchors = np.concatenate([anchor0[:, None], anchor_rest], axis=1).astype(np.float32)
    y_lasso = (anchor0 + (x_mod * anchor_rest).sum(axis=1)).astype(np.float32)

    head = GeneHeadMLP(encoder.feature_dim, n_betas).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-5)
    mse = nn.MSELoss()

    feat_t = torch.from_numpy(features.astype(np.float32))
    x_mod_t = torch.from_numpy(cache.modulators.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))
    y_lasso_t = torch.from_numpy(y_lasso)
    anchors_t = torch.from_numpy(anchors)

    ds = TensorDataset(feat_t, x_mod_t, y_t, y_lasso_t, anchors_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=len(ds) > batch_size)

    for _epoch in range(epochs):
        head.train()
        for bf, bx, by, byl, ba in dl:
            bf, bx, by, byl, ba = [t.to(device) for t in (bf, bx, by, byl, ba)]
            opt.zero_grad()
            raw = head(bf)
            betas = apply_output_activation(raw, "sigmoidx2") * ba
            pred = linear_readout_y(betas, bx)
            loss = mse(pred, by) + align_weight * mse(pred, byl)
            loss.backward()
            opt.step()

    head.eval()
    with torch.no_grad():
        raw = head(feat_t.to(device))
        betas = apply_output_activation(raw, "sigmoidx2") * anchors_t.to(device)
        pred = linear_readout_y(betas, x_mod_t.to(device)).cpu().numpy()

    r2_cnn = r2_score(y, pred)
    r2_lasso_all = r2_score(y, y_lasso)

    per_cluster: dict[str, float] = {}
    for c, fit in lasso_fits.items():
        mask = clusters == c
        per_cluster[cache.cluster_labels[c]] = r2_score(y[mask], pred[mask])

    betadata = betas.cpu().numpy().astype(np.float64)
    feather_path = out_dir / "betadata" / f"{gene}_betadata.feather"
    write_betadata_feather(feather_path, cache.obs_names, beta_columns, betadata)

    head_path = out_dir / "gene_heads" / f"{gene}_head.pt"
    head_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"gene": gene, "state_dict": head.state_dict(), "beta_columns": beta_columns}, head_path)

    return {
        "gene": gene,
        "r2_cnn": r2_cnn,
        "r2_lasso": r2_lasso_all,
        "n_lasso_clusters": len(lasso_fits),
        "per_cluster_r2": per_cluster,
        "betadata": str(feather_path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", type=Path, default=Path("data/tonsil_finetune.h5ad"))
    p.add_argument("--cache", type=Path, default=Path("outputs/finetune_cache.npz"))
    p.add_argument("--encoder", type=Path, default=Path("outputs/pretrain/tissue_encoder_base.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/finetune"))
    p.add_argument("--genes", default="AICDA,CD74,MS4A1,CD3D,MKI67")
    p.add_argument("--spatial-dim", type=int, default=16)
    p.add_argument("--radius", type=float, default=300.0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--force-genes", default="")
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

    dev = torch.device(args.device)
    encoder = load_encoder(args.encoder, dev)
    features = encode_all(encoder, cache, dev)

    genes = [g.strip() for g in args.genes.split(",") if g.strip()]
    results = []
    for gene in genes:
        if gene not in cache.gene_names:
            print(f"skip missing gene {gene}")
            continue
        res = finetune_gene(gene, cache, encoder, features, args.out_dir, dev, epochs=args.epochs)
        results.append(res)
        print(f"{gene}: CNN R²={res['r2_cnn']:.4f}  Lasso R²={res['r2_lasso']:.4f}")

    write_json(args.out_dir / "gene_performance.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
