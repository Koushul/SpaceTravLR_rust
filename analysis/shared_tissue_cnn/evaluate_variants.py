"""Score CNN variants by linear-probe transfer on finetune half."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from data_utils import SpatialCache, gene_index, write_json
from finetune_genes import encode_all, load_encoder


def probe_transfer(
    encoder_ckpt: Path,
    train_cache: SpatialCache,
    finetune_cache: SpatialCache,
    probe_genes: list[str],
    device: str = "cpu",
) -> dict:
    import torch

    dev = torch.device(device)
    enc = load_encoder(encoder_ckpt, dev)
    train_feat = encode_all(enc, train_cache, dev)
    fin_feat = encode_all(enc, finetune_cache, dev)
    scores: dict[str, float] = {}
    for gene in probe_genes:
        if gene not in train_cache.gene_names or gene not in finetune_cache.gene_names:
            continue
        gi_tr = gene_index(train_cache, gene)
        gi_fi = gene_index(finetune_cache, gene)
        y_tr = train_cache.expr_log1p[:, gi_tr]
        y_fi = finetune_cache.expr_log1p[:, gi_fi]
        ridge = Ridge(alpha=1.0)
        ridge.fit(train_feat, y_tr)
        pred = ridge.predict(fin_feat)
        ss_res = float(np.sum((y_fi - pred) ** 2))
        ss_tot = float(np.sum((y_fi - y_fi.mean()) ** 2))
        scores[gene] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mean_r2 = float(np.mean(list(scores.values()))) if scores else float("-inf")
    return {"per_gene_r2": scores, "mean_transfer_r2": mean_r2}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", type=Path, default=Path("outputs/train_cache.npz"))
    p.add_argument("--finetune-cache", type=Path, default=Path("outputs/finetune_cache.npz"))
    p.add_argument("--pretrain-dir", type=Path, default=Path("outputs/pretrain"))
    p.add_argument("--variants", default="base,deep,wide")
    p.add_argument("--probe-genes", default="AICDA,CD74,CD3D,MS4A6A,LYZ")
    p.add_argument("--out", type=Path, default=Path("outputs/variant_transfer.json"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    train = SpatialCache.load(args.train_cache)
    fin = SpatialCache.load(args.finetune_cache)
    genes = [g.strip() for g in args.probe_genes.split(",") if g.strip()]

    results: dict[str, dict] = {}
    for variant in args.variants.split(","):
        variant = variant.strip()
        ckpt = args.pretrain_dir / f"tissue_encoder_{variant}.pt"
        if not ckpt.exists():
            continue
        results[variant] = probe_transfer(ckpt, train, fin, genes, device=args.device)

    best = max(results, key=lambda v: results[v]["mean_transfer_r2"]) if results else "base"
    payload = {"variants": results, "best_by_transfer": best}
    write_json(args.out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
