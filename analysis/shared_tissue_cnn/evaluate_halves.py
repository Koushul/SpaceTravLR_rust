"""Compare gene MLP performance on train vs finetune tissue halves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data_utils import SpatialCache, gene_index
from finetune_genes import encode_all, load_encoder
from group_lasso import fit_gene_lasso_anchors, r2_score
from models import GeneHeadMLP, apply_output_activation, linear_readout_y


def eval_gene_on_cache(
    gene: str,
    head_path: Path,
    encoder_ckpt: Path,
    cache: SpatialCache,
    device: torch.device,
) -> float:
    ckpt = torch.load(head_path, map_location=device)
    beta_columns = ckpt["beta_columns"]
    n_betas = len(beta_columns)
    n_mod = n_betas - 1
    encoder = load_encoder(encoder_ckpt, device)
    head = GeneHeadMLP(encoder.feature_dim, n_betas).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    gi = gene_index(cache, gene)
    y = cache.expr_log1p[:, gi].astype(np.float64)
    x_mod = cache.modulators.astype(np.float64)
    clusters = cache.clusters

    lasso_fits = fit_gene_lasso_anchors(y, x_mod, clusters, cache.num_clusters)
    anchor0 = np.zeros(len(y), dtype=np.float32)
    anchor_rest = np.zeros((len(y), n_mod), dtype=np.float32)
    for c, fit in lasso_fits.items():
        mask = clusters == c
        anchor0[mask] = fit.intercept
        anchor_rest[mask] = fit.coef

    anchors = np.concatenate([anchor0[:, None], anchor_rest], axis=1).astype(np.float32)
    features = encode_all(encoder, cache, device)

    with torch.no_grad():
        raw = head(torch.from_numpy(features).to(device))
        betas = apply_output_activation(raw, "sigmoidx2") * torch.from_numpy(anchors).to(device)
        pred = linear_readout_y(
            betas,
            torch.from_numpy(cache.modulators.astype(np.float32)).to(device),
        ).cpu().numpy()
    return r2_score(y, pred)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", type=Path, default=Path("outputs/train_cache.npz"))
    p.add_argument("--finetune-cache", type=Path, default=Path("outputs/finetune_cache.npz"))
    p.add_argument("--encoder", type=Path, default=Path("outputs/pretrain/tissue_encoder_deep.pt"))
    p.add_argument("--heads-dir", type=Path, default=Path("outputs/finetune/gene_heads"))
    p.add_argument("--out", type=Path, default=Path("outputs/half_comparison.json"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    dev = torch.device(args.device)
    train = SpatialCache.load(args.train_cache)
    fin = SpatialCache.load(args.finetune_cache)

    rows = []
    for head_path in sorted(args.heads_dir.glob("*_head.pt")):
        gene = head_path.stem.replace("_head", "")
        if gene not in train.gene_names or gene not in fin.gene_names:
            continue
        rows.append(
            {
                "gene": gene,
                "r2_train_half": eval_gene_on_cache(gene, head_path, args.encoder, train, dev),
                "r2_finetune_half": eval_gene_on_cache(gene, head_path, args.encoder, fin, dev),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
