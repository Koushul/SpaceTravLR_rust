"""Split tonsil AnnData 50/50 for CNN pretrain vs per-gene fine-tune."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np


def split_h5ad(
    h5ad_path: Path,
    out_dir: Path,
    seed: int = 42,
    train_frac: float = 0.5,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(h5ad_path)
    rng = np.random.default_rng(seed)
    n = adata.n_obs
    perm = rng.permutation(n)
    n_train = int(n * train_frac)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train = adata[train_idx].copy()
    test = adata[test_idx].copy()

    train_path = out_dir / "tonsil_train.h5ad"
    test_path = out_dir / "tonsil_finetune.h5ad"
    train.write_h5ad(train_path)
    test.write_h5ad(test_path)

    meta = out_dir / "split_meta.json"
    meta.write_text(
        f'{{"seed": {seed}, "train_frac": {train_frac}, '
        f'"n_train": {train.n_obs}, "n_finetune": {test.n_obs}}}\n'
    )
    return train_path, test_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--h5ad",
        type=Path,
        default=Path("../../data/h5ad/SlideTags_human_tonsil.h5ad"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.5)
    args = p.parse_args()
    train_path, test_path = split_h5ad(args.h5ad, args.out_dir, args.seed, args.train_frac)
    print(f"train: {train_path}  finetune: {test_path}")


if __name__ == "__main__":
    main()
