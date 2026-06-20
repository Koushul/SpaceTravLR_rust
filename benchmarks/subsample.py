"""Subsample the atera_human_cervix.h5ad source into a smaller benchmark .h5ad.

Selection strategy
------------------
- Stratified by `obs[cell_type]` so all clusters are represented (proportional sampling).
- Deterministic via numpy RandomState(seed + n).
- The result has `X = layers['normalized_count']` (un-logged, normalized counts), no
  `uns['log1p']`, no `obsm['X_pca']` / `obsm['X_umap']`, no `obsp/*`. This way both the
  Rust pure-Rust and the scanpy preprocess paths run identical work for a fair benchmark.
- We also keep `obsm['spatial']` so that downstream rust preprocess does not warn.

For the largest size (700k) the source is essentially the whole dataset; we still go
through the same path so timings stay consistent.

Usage
-----
python subsample.py --n 7000 --out benchmarks/data/atera_n7000.h5ad [--source PATH]
"""
import argparse
import json
import os
import sys
import time
import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument(
        "--source",
        type=str,
        default="/ix/djishnu/shared/djishnu_kor11/training_data_revision/atera_human_cervix.h5ad",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--stratify", type=str, default="cell_type")
    return p.parse_args()


def pick_indices(src: str, n_target: int, seed: int, stratify_col: str) -> np.ndarray:
    """Pick a stratified subsample of row indices."""
    with h5py.File(src, "r") as f:
        total = f["X"].shape[0]
        n_target = min(n_target, total)
        codes_path = f"obs/{stratify_col}/codes"
        if codes_path in f:
            codes = f[codes_path][...]
        else:
            codes = np.zeros(total, dtype=np.int32)

    rng = np.random.RandomState(seed)
    if n_target == total:
        idx = np.arange(total)
        rng.shuffle(idx)
        return np.sort(idx)

    uniq, counts = np.unique(codes, return_counts=True)
    keep_per = np.maximum(1, np.round(counts / total * n_target).astype(np.int64))
    deficit = n_target - keep_per.sum()
    if deficit != 0:
        order = np.argsort(-counts)
        i = 0
        while deficit != 0:
            j = order[i % len(order)]
            if deficit > 0:
                keep_per[j] += 1
                deficit -= 1
            else:
                if keep_per[j] > 1:
                    keep_per[j] -= 1
                    deficit += 1
            i += 1

    picks = []
    for code, k in zip(uniq, keep_per):
        mask_idx = np.where(codes == code)[0]
        if k >= len(mask_idx):
            picks.append(mask_idx)
        else:
            chosen = rng.choice(mask_idx, size=int(k), replace=False)
            picks.append(chosen)
    out = np.sort(np.concatenate(picks))
    if len(out) > n_target:
        out = np.sort(rng.choice(out, size=n_target, replace=False))
    return out


def write_subsample(src: str, idx: np.ndarray, out_path: str) -> dict:
    """Write a slim h5ad with X = normalized_count, spatial, basic obs/var."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t0 = time.time()
    with h5py.File(src, "r") as fi, h5py.File(out_path, "w") as fo:
        n_genes = fi["X"].shape[1]
        n = len(idx)

        nc_src = fi["layers/normalized_count"]
        x_out = fo.create_dataset("X", shape=(n, n_genes), dtype="float32")
        x_out.attrs["encoding-type"] = "array"
        x_out.attrs["encoding-version"] = "0.2.0"

        chunk = 4096
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            rows = idx[s:e]
            sorted_idx = np.argsort(rows)
            row_sorted = rows[sorted_idx]
            block = nc_src[list(row_sorted), :].astype(np.float32)
            inv = np.empty_like(sorted_idx)
            inv[sorted_idx] = np.arange(len(sorted_idx))
            x_out[s:e, :] = block[inv]

        layers = fo.create_group("layers")
        layers.attrs["encoding-type"] = "dict"
        layers.attrs["encoding-version"] = "0.1.0"

        obs_g = fo.create_group("obs")
        obs_g.attrs["encoding-type"] = "dataframe"
        obs_g.attrs["encoding-version"] = "0.2.0"
        obs_g.attrs["_index"] = "_index"
        obs_g.attrs["column-order"] = np.array(
            ["x_centroid", "y_centroid", "cell_type", "leiden"], dtype=object
        )

        src_index = fi["obs/_index"][...]
        sub_index = src_index[idx]
        obs_g.create_dataset("_index", data=sub_index)

        for col in ("x_centroid", "y_centroid"):
            if f"obs/{col}" in fi:
                obs_g.create_dataset(col, data=fi[f"obs/{col}"][...][idx])
        for cat_col in ("cell_type", "leiden"):
            cat_path = f"obs/{cat_col}"
            if cat_path not in fi:
                continue
            cats = fi[f"{cat_path}/categories"][...]
            codes = fi[f"{cat_path}/codes"][...][idx]
            g = obs_g.create_group(cat_col)
            g.attrs["encoding-type"] = "categorical"
            g.attrs["encoding-version"] = "0.2.0"
            g.attrs["ordered"] = False
            g.create_dataset("categories", data=cats)
            g.create_dataset("codes", data=codes.astype(np.int8))

        var_g = fo.create_group("var")
        var_g.attrs["encoding-type"] = "dataframe"
        var_g.attrs["encoding-version"] = "0.2.0"
        var_g.attrs["_index"] = "_index"
        var_g.attrs["column-order"] = np.array(["gene_ids"], dtype=object)
        var_index = fi["var/_index"][...]
        var_g.create_dataset("_index", data=var_index)
        if "var/gene_ids" in fi:
            var_g.create_dataset("gene_ids", data=fi["var/gene_ids"][...])

        obsm_g = fo.create_group("obsm")
        obsm_g.attrs["encoding-type"] = "dict"
        obsm_g.attrs["encoding-version"] = "0.1.0"
        if "obsm/spatial" in fi:
            obsm_g.create_dataset("spatial", data=fi["obsm/spatial"][...][idx, :])

        fo.create_group("uns").attrs["encoding-type"] = "dict"
        fo["uns"].attrs["encoding-version"] = "0.1.0"

    elapsed = time.time() - t0
    size_gb = os.path.getsize(out_path) / 1e9
    return {"n_cells": int(len(idx)), "elapsed_s": elapsed, "size_gb": size_gb, "out": out_path}


def main():
    args = parse_args()
    if os.path.exists(args.out):
        try:
            with h5py.File(args.out, "r") as f:
                if f["X"].shape[0] == args.n:
                    info = {"n_cells": args.n, "elapsed_s": 0.0, "size_gb": os.path.getsize(args.out) / 1e9, "out": args.out, "cached": True}
                    print(json.dumps(info))
                    return 0
        except Exception:
            pass
    idx = pick_indices(args.source, args.n, args.seed + args.n, args.stratify)
    info = write_subsample(args.source, idx, args.out)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
