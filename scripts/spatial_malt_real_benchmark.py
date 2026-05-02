#!/usr/bin/env python3
"""Real-data benchmark for spatial MALT label transfer.

Downloads the public Open Problems seqFISH Mouse Organogenesis AnnData,
creates a stratified reference/query split, writes lightweight seed-style
betadata features from each cell's local spatial neighborhood, runs
`spacetravlr --map-labels --map-labels-spatial`, and prints metrics.

Recommended invocation from the repo root:

  uv run --isolated --with 'numpy<2' --with 'pandas>=2.2' \
    --with 'anndata>=0.11' --with scanpy --with scikit-learn \
    --with pyarrow --with requests \
    python scripts/spatial_malt_real_benchmark.py \
      --spacetravlr ./target/debug/spacetravlr
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import requests
import scanpy as sc
from sklearn.neighbors import NearestNeighbors


DATASET_URL = (
    "https://openproblems-data.s3.amazonaws.com/resources/datasets/"
    "zenodo_spatial/seqfish/mouse_organogenesis_seqfish/dataset.h5ad"
)


def download_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 1_000_000:
        return
    with requests.get(DATASET_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)


def prepare_split(
    dataset_path: Path,
    out_dir: Path,
    *,
    label_key: str,
    cells_per_type: int | None,
    n_types: int | None,
    train_fraction: float,
    seed: int,
) -> tuple[Path, Path, list[str]]:
    rng = np.random.default_rng(seed)
    a = sc.read_h5ad(dataset_path)
    if "counts" in a.layers:
        a.X = a.layers["counts"].copy()
    a.var_names = a.var["feature_name"].astype(str).values if "feature_name" in a.var else a.var_names
    a.var_names_make_unique()
    a.obs[label_key] = a.obs[label_key].astype(str)
    a = a[a.obs[label_key] != "Low quality"].copy()

    counts = a.obs[label_key].value_counts()
    min_cells = 2 if cells_per_type is None else max(2, cells_per_type)
    chosen = counts[counts >= min_cells].index.tolist()
    if n_types is not None:
        chosen = chosen[:n_types]
    keep_idx: list[int] = []
    for ct in chosen:
        idx = np.flatnonzero(a.obs[label_key].values == ct)
        if cells_per_type is None:
            take = idx
        else:
            take = rng.choice(idx, size=cells_per_type, replace=False)
        keep_idx.extend(take.tolist())
    keep_idx = np.array(sorted(keep_idx))
    a = a[keep_idx].copy()

    train_idx: list[int] = []
    test_idx: list[int] = []
    labels = a.obs[label_key].astype(str).values
    for ct in chosen:
        idx = np.flatnonzero(labels == ct)
        rng.shuffle(idx)
        n_train = int(round(len(idx) * train_fraction))
        n_train = min(max(1, n_train), len(idx) - 1)
        train_idx.extend(idx[:n_train].tolist())
        test_idx.extend(idx[n_train:].tolist())

    ref = a[np.array(sorted(train_idx))].copy()
    query = a[np.array(sorted(test_idx))].copy()
    query.obs["truth"] = query.obs[label_key].astype(str)

    for x in (ref, query):
        x.obs["cell_type"] = x.obs[label_key].astype(str)
        if "counts" not in x.layers:
            x.layers["counts"] = x.X.copy()

    ref_path = out_dir / "seqfish_reference_train.h5ad"
    query_path = out_dir / "seqfish_query_test.h5ad"
    ref.write_h5ad(ref_path)
    query.write_h5ad(query_path)
    return ref_path, query_path, chosen


def top_marker_genes(ref_path: Path, *, label_key: str, genes_per_type: int) -> list[str]:
    ref = sc.read_h5ad(ref_path)
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)
    sc.tl.rank_genes_groups(ref, groupby=label_key, method="wilcoxon", n_genes=50, use_raw=False)
    genes: list[str] = []
    seen: set[str] = set()
    for ct in ref.obs[label_key].astype(str).unique():
        df = sc.get.rank_genes_groups_df(ref, group=ct)
        for g in df["names"].astype(str).head(genes_per_type):
            if g not in seen and g in ref.var_names:
                seen.add(g)
                genes.append(g)
    return genes


def write_local_betadata(
    h5ad_path: Path,
    out_dir: Path,
    genes: list[str],
    *,
    id_col: str,
    label_key: str,
    k: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    a = sc.read_h5ad(h5ad_path)
    X = a.layers["counts"] if "counts" in a.layers else a.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    E = np.asarray(a.X.toarray() if hasattr(a.X, "toarray") else a.X, dtype=np.float32)
    xy = np.asarray(a.obsm["spatial"], dtype=np.float32)[:, :2]
    nn = NearestNeighbors(n_neighbors=min(k + 1, a.n_obs), metric="euclidean")
    nn.fit(xy)
    dist, idx = nn.kneighbors(xy)
    neigh = idx[:, 1:]
    dist = dist[:, 1:]
    sig = float(np.median(dist[dist > 0])) if np.any(dist > 0) else 1.0
    w = np.exp(-(dist**2) / (2.0 * max(sig, 1e-6) ** 2)).astype(np.float32)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-8)

    ids = a.obs_names.astype(str) if id_col == "CellID" else a.obs[label_key].astype(str).values
    for gene in genes:
        gi = int(a.var_names.get_loc(gene))
        local = (w * E[neigh, gi]).sum(1)
        own = E[:, gi]
        rows = pd.DataFrame(
            {
                id_col: ids,
                "beta0": np.zeros(a.n_obs, dtype=np.float32),
                "beta_local_expr": own,
                "beta_spatial_neighbor": local,
                "beta_spatial_delta": local - own,
            }
        )
        if id_col != "CellID":
            rows = rows.groupby(id_col, as_index=False).mean(numeric_only=True)
        rows.to_feather(out_dir / f"{gene}_betadata.feather")


def run_spacetravlr(args: argparse.Namespace, ref_path: Path, query_path: Path, ref_beta: Path, query_beta: Path) -> None:
    if str(args.spacetravlr) == "python":
        cmd = [
            sys.executable,
            "scripts/malt_label_transfer.py",
            "--spatial",
            "--reference",
            str(ref_path),
            "--query",
            str(query_path),
            "--outdir",
            str(args.out_dir / "malt_out"),
            "--groupby",
            "cell_type",
            "--reference-betadata-dir",
            str(ref_beta),
            "--query-betadata-dir",
            str(query_beta),
            "--benchmark-truth",
            "truth",
            "--no-leiden-map",
            "--expression-mode",
            "counts",
            "--counts-layer",
            "counts",
        ]
    else:
        cmd = [
            str(args.spacetravlr),
            "--map-labels",
            "--map-labels-spatial",
            "--reference",
            str(ref_path),
            "--query",
            str(query_path),
            "--map-labels-outdir",
            str(args.out_dir / "malt_out"),
            "--map-labels-groupby",
            "cell_type",
            "--map-labels-reference-betadata-dir",
            str(ref_beta),
            "--map-labels-query-betadata-dir",
            str(query_beta),
            "--map-labels-benchmark-truth",
            "truth",
            "--map-labels-no-leiden",
            "--map-labels-expression-mode",
            "counts",
            "--map-labels-counts-layer",
            "counts",
        ]
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/spacetravlr_real_spatial_benchmark"))
    p.add_argument("--spacetravlr", type=Path, default=Path("./target/debug/spacetravlr"))
    p.add_argument("--cells-per-type", type=int, default=0, help="0 uses every cell per retained type")
    p.add_argument("--n-types", type=int, default=0, help="0 uses every non-low-quality annotated type")
    p.add_argument("--train-fraction", type=float, default=0.5)
    p.add_argument("--genes-per-type", type=int, default=4)
    p.add_argument("--neighbor-k", type=int, default=8)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.out_dir / "mouse_organogenesis_seqfish.h5ad"
    download_dataset(dataset)
    ref_path, query_path, cell_types = prepare_split(
        dataset,
        args.out_dir,
        label_key="celltype_mapped_refined",
        cells_per_type=None if args.cells_per_type == 0 else args.cells_per_type,
        n_types=None if args.n_types == 0 else args.n_types,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    genes = top_marker_genes(ref_path, label_key="cell_type", genes_per_type=args.genes_per_type)
    (args.out_dir / "benchmark_genes.txt").write_text("\n".join(genes) + "\n")
    ref_beta = args.out_dir / "reference_betadata"
    query_beta = args.out_dir / "query_betadata"
    write_local_betadata(ref_path, ref_beta, genes, id_col="CellID", label_key="cell_type", k=args.neighbor_k)
    write_local_betadata(query_path, query_beta, genes, id_col="CellID", label_key="cell_type", k=args.neighbor_k)
    run_spacetravlr(args, ref_path, query_path, ref_beta, query_beta)

    meta = json.loads((args.out_dir / "malt_out" / "run_meta.json").read_text())
    bench = meta["per_group"][0]["spatial_malt"]["benchmark"]
    rows = []
    for method, vals in bench.items():
        rows.append(
            {
                "method": method,
                "accuracy": vals.get("accuracy"),
                "balanced_accuracy": vals.get("balanced_accuracy"),
                "ari": vals.get("ari"),
                "dotplot_mean_r2": vals.get("dotplot_mean_r2"),
            }
        )
    df = pd.DataFrame(rows).sort_values("method")
    df.to_csv(args.out_dir / "metrics.csv", index=False)
    print("Dataset: Open Problems seqFISH Mouse Organogenesis")
    print(f"Cell types: {', '.join(cell_types)}")
    print(f"Reference: {ref_path}")
    print(f"Query: {query_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
