#!/usr/bin/env python3
"""
PBMC 3k benchmark for MALT with optional GRN (seed-only TF betas from SpaceTravLR).

Ground truth: Scanpy `pbmc3k_processed` (canonical tutorial object; obs labels from Louvain).

Workflow:
  1) Split cells into reference / query.
  2) Add fake 2D `obsm['spatial']`; reference `obs['cell_type']` = Louvain; query drops labels.
  3) `obs['cluster_train']`: reference = cell_type; query = Leiden on query PCA.
  4) Optionally run `spacetravlr` twice (`--training-mode seed`, TF-only modulators) using a patched
     config (`layer = "X"`, `cluster_annot = "cluster_train"`).
  5) Run `scripts/malt_label_transfer.py` baseline vs GRN-weighted; compare accuracy vs Louvain.

Requires: scanpy, anndata, numpy, scipy, torch, pyarrow, pandas, h5py.

Example:
  uv run --with scanpy --with anndata --with numpy --with scipy \\
    --with torch --with pyarrow --with pandas --with h5py \\
    examples/benchmark_malt_pbmc3k_grn.py --outdir /tmp/pbmc3k_malt_grn \\
    --train-spacetravlr --spacetravlr-bin spacetravlr \\
    --config /path/to/spaceship_config.toml --network-data-dir /path/with/human_network.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys

import numpy as np
import scanpy as sc


def _repo_scripts_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


def _patch_spaceship_train_toml(
    src: str,
    dst: str,
    *,
    cluster_annot: str,
    layer: str,
    train_modulators: str,
) -> None:
    text = open(src, encoding="utf-8").read()
    text = re.sub(
        r"(?m)^\s*cluster_annot\s*=.*$",
        f'cluster_annot = "{cluster_annot}"',
        text,
        count=1,
    )
    text = re.sub(
        r"(?m)^\s*layer\s*=.*$",
        f'layer = "{layer}"',
        text,
        count=1,
    )
    if re.search(r"(?m)^\s*train_modulators\s*=", text):
        text = re.sub(
            r"(?m)^\s*train_modulators\s*=.*$",
            f'train_modulators = "{train_modulators}"',
            text,
            count=1,
        )
    else:
        text = re.sub(
            r"(\[grn\]\s*\n)",
            rf'\1train_modulators = "{train_modulators}"\n',
            text,
            count=1,
        )
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)


def _accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    p = np.asarray(pred).astype(str)
    t = np.asarray(true).astype(str)
    return float((p == t).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ref-fraction", type=float, default=0.67)
    ap.add_argument(
        "--train-spacetravlr",
        action="store_true",
        help="Run spacetravlr --training-mode seed on ref and query (needs binary + GRN parquets)",
    )
    ap.add_argument("--spacetravlr-bin", default="spacetravlr")
    ap.add_argument("--config", default=None, help="spaceship_config.toml (must set grn paths)")
    ap.add_argument(
        "--network-data-dir",
        default=None,
        help="Override [grn].network_data_dir for training (human_network.parquet)",
    )
    ap.add_argument("--max-genes-train", type=int, default=80, help="Cap genes per training run")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--grn-weight", type=float, default=2.0)
    ap.add_argument(
        "--skip-malt",
        action="store_true",
        help="Only prepare h5ad + optional training; skip MALT subprocess",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    ad = sc.datasets.pbmc3k_processed()
    if "louvain" not in ad.obs.columns:
        raise KeyError("pbmc3k_processed missing obs['louvain']")
    ad.obs["cell_type"] = ad.obs["louvain"].astype(str)
    label_col = "cell_type"
    n = ad.n_obs
    ix = np.arange(n)
    rng.shuffle(ix)
    n_ref = int(n * args.ref_fraction)
    ref_ix = ix[:n_ref]
    q_ix = ix[n_ref:]

    def _tag_spatial(a: sc.AnnData) -> None:
        xy = np.column_stack(
            [
                np.random.RandomState(args.seed).randn(a.n_obs),
                np.random.RandomState(args.seed + 1).randn(a.n_obs),
            ]
        ).astype(np.float32)
        a.obsm["spatial"] = xy

    ref = ad[ref_ix].copy()
    query_full = ad[q_ix].copy()
    _tag_spatial(ref)
    _tag_spatial(query_full)

    ref.obs["cluster_train"] = ref.obs[label_col].astype(str)
    q_tmp = query_full.copy()
    sc.pp.neighbors(q_tmp, n_neighbors=15, use_rep="X_pca")
    sc.tl.leiden(q_tmp, resolution=0.6, key_added="leiden")
    query_full.obs["leiden"] = q_tmp.obs["leiden"].astype(str)
    query_full.obs["cluster_train"] = query_full.obs["leiden"]

    true_labels = query_full.obs[label_col].to_numpy()
    query = query_full.copy()
    query.obs.drop(columns=[label_col], inplace=True, errors="ignore")

    ref_path = os.path.join(args.outdir, "reference.h5ad")
    q_path = os.path.join(args.outdir, "query.h5ad")
    ref.write_h5ad(ref_path)
    query.write_h5ad(q_path)

    marker_union: list[str] = []
    ref_tmp = ref.copy()
    sc.tl.rank_genes_groups(ref_tmp, groupby=label_col, method="wilcoxon", n_genes=50, use_raw=False)
    cats = ref_tmp.obs[label_col].astype("category").cat.categories
    for ct in cats:
        df = sc.get.rank_genes_groups_df(ref_tmp, group=str(ct))
        sig = df[(df["pvals_adj"] < 0.05) & (df["logfoldchanges"] > 0.25)]
        marker_union.extend(sig.head(15)["names"].astype(str).tolist())
    marker_union = sorted(set(marker_union) & set(ref.var_names) & set(query.var_names))
    genes_file = os.path.join(args.outdir, "malt_marker_genes_train.txt")
    with open(genes_file, "w") as f:
        f.write("\n".join(marker_union) + "\n")

    meta = {
        "dataset": "scanpy.datasets.pbmc3k_processed",
        "label_column": label_col,
        "training_cluster_obs": "cluster_train",
        "note": "Patched TOML sets [data].cluster_annot = cluster_train, layer = X, train_modulators = tf.",
        "n_ref": int(ref.n_obs),
        "n_query": int(query.n_obs),
        "n_marker_genes_for_train": len(marker_union),
        "marker_genes_file": genes_file,
    }
    with open(os.path.join(args.outdir, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    ref_run = os.path.join(args.outdir, "train_ref")
    q_run = os.path.join(args.outdir, "train_query")
    malt_base = os.path.join(args.outdir, "malt_baseline")
    malt_grn = os.path.join(args.outdir, "malt_grn")

    if args.train_spacetravlr:
        if not args.config or not os.path.isfile(args.config):
            raise SystemExit("--train-spacetravlr requires existing --config spaceship_config.toml")
        train_cfg = os.path.join(args.outdir, "spaceship_train_benchmark.toml")
        _patch_spaceship_train_toml(
            args.config,
            train_cfg,
            cluster_annot="cluster_train",
            layer="X",
            train_modulators="tf",
        )
        glist = ",".join(marker_union[: args.max_genes_train])
        common = [
            args.spacetravlr_bin,
            "--plain",
            "--training-mode",
            "seed",
            "--config",
            train_cfg,
            "--h5ad",
            ref_path,
            "--output-dir",
            ref_run,
            "--genes",
            glist,
            "--parallel",
            str(args.parallel),
        ]
        env = os.environ.copy()
        if args.network_data_dir:
            env.setdefault("SPACETRAVLR_DATA_DIR", args.network_data_dir)
        subprocess.run(common, check=True, env=env)
        subprocess.run(
            [
                args.spacetravlr_bin,
                "--plain",
                "--training-mode",
                "seed",
                "--config",
                train_cfg,
                "--h5ad",
                q_path,
                "--output-dir",
                q_run,
                "--genes",
                glist,
                "--parallel",
                str(args.parallel),
            ],
            check=True,
            env=env,
        )

    if args.skip_malt:
        print(f"Prepared {ref_path} and {q_path}. Exiting (--skip-malt).")
        return

    malt_py = os.path.join(_repo_scripts_dir(), "malt_label_transfer.py")
    if not os.path.isfile(malt_py):
        raise FileNotFoundError(malt_py)

    def _run_malt(out: str, extra: list[str]) -> None:
        os.makedirs(out, exist_ok=True)
        cmd = [
            sys.executable,
            malt_py,
            "--reference",
            ref_path,
            "--query",
            q_path,
            "--outdir",
            out,
            "-g",
            label_col,
            "--expression-mode",
            "lognorm",
            *extra,
        ]
        subprocess.run(cmd, check=True)

    shutil.rmtree(malt_base, ignore_errors=True)
    shutil.rmtree(malt_grn, ignore_errors=True)
    _run_malt(malt_base, [])

    if args.train_spacetravlr:
        cl = "cluster_train"
        if cl not in query.obs.columns:
            query.obs[cl] = query_full.obs["cluster_train"].values
            query.write_h5ad(q_path)
        grn_extra = [
            "--ref-betadata-dir",
            ref_run,
            "--query-betadata-dir",
            q_run,
            "--query-grn-cluster-obs",
            cl,
            "--grn-loss-weight",
            str(args.grn_weight),
        ]
        _run_malt(malt_grn, grn_extra)
    else:
        print("Skipping GRN MALT (--train-spacetravlr not set).")

    def _acc_from_out(d: str) -> float | None:
        p = os.path.join(d, "query_labeled.h5ad")
        if not os.path.isfile(p):
            return None
        qo = sc.read_h5ad(p)
        pred = qo.obs["malt_label"].astype(str).values
        return _accuracy(pred, true_labels)

    acc_base = _acc_from_out(malt_base)
    results = {"accuracy_malt_baseline_vs_louvain": acc_base}
    if args.train_spacetravlr:
        results["accuracy_malt_grn_vs_louvain"] = _acc_from_out(malt_grn)
    with open(os.path.join(args.outdir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
