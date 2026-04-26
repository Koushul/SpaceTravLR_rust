#!/usr/bin/env python3
"""
MALT + optional GRN benchmark: MERFISH mouse cortex (spatial) vs Allen scRNA reference.

Spatial data: squidpy ``merfish()`` — MERFISH cells with ``obs['Cell_class']`` annotations
(Moffit et al.; bundled in squidpy). Coordinates in ``obsm['spatial']``.

Reference: squidpy ``sc_mouse_cortex()`` — Allen Brain Atlas single-cell (mouse), with
``obs['cell_class']`` / ``obs['cell_subclass']``. Only neurons used for reference labels:
``Glutamatergic`` → ``Excitatory``, ``GABAergic`` → ``Inhibitory`` to align with the
binary spatial task (spatial subset: ``Cell_class`` in {Excitatory, Inhibitory}).

Ground truth: spatial ``Cell_class`` on the held-out query. MALT predicts ``cell_type``;
accuracy = fraction of query cells where ``malt_label`` matches that ground truth.

Requires: squidpy, scanpy, anndata, numpy, scipy, torch, pyarrow, pandas, h5py, igraph, leidenalg.

Example:
  uv run --with squidpy --with scanpy --with anndata --with numpy --with scipy \\
    --with torch --with pyarrow --with pandas --with h5py --with igraph --with leidenalg \\
    examples/benchmark_malt_merfish_grn.py --outdir /tmp/merfish_malt_grn \\
    --train-spacetravlr --spacetravlr-bin spacetravlr \\
    --config /path/to/spaceship_config.toml --network-data-dir /path/with/mouse_network.parquet
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
import squidpy as sq


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


def _ref_neuron_label(cell_class: str) -> str | None:
    cc = str(cell_class)
    if cc == "Glutamatergic":
        return "Excitatory"
    if cc == "GABAergic":
        return "Inhibitory"
    return None


def _lognorm_layers(adata: sc.AnnData, *, copy_raw_to_layer: bool = False) -> None:
    """Fill ``layers['ln']`` with log1p(normalize_total(.X)) for MALT lognorm mode."""
    ad = adata
    if copy_raw_to_layer:
        import scipy.sparse as sp

        x0 = ad.X
        if sp.issparse(x0):
            ad.layers["counts"] = x0.copy()
        else:
            ad.layers["counts"] = np.asarray(x0, dtype=np.float32).copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    import scipy.sparse as sp

    if sp.issparse(ad.X):
        ad.layers["ln"] = ad.X.copy()
    else:
        ad.layers["ln"] = np.asarray(ad.X, dtype=np.float32).copy()


def _subsample_obs(ad: sc.AnnData, n_max: int, seed: int) -> sc.AnnData:
    if ad.n_obs <= n_max:
        return ad
    rng = np.random.default_rng(seed)
    ix = rng.choice(ad.n_obs, size=n_max, replace=False)
    return ad[ix].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ref-fraction", type=float, default=0.65)
    ap.add_argument("--max-spatial-cells", type=int, default=10000, help="Cap MERFISH query+pool size")
    ap.add_argument("--max-ref-cells", type=int, default=15000, help="Cap Allen reference cells")
    ap.add_argument(
        "--train-spacetravlr",
        action="store_true",
        help="Run spacetravlr --training-mode seed on ref and query (needs binary + mouse GRN)",
    )
    ap.add_argument("--spacetravlr-bin", default="spacetravlr")
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--network-data-dir",
        default=None,
        help="Directory with mouse_network.parquet (and optional human_network.parquet)",
    )
    ap.add_argument("--max-genes-train", type=int, default=70)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--grn-weight", type=float, default=2.0)
    ap.add_argument("--skip-malt", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    print("Loading squidpy.datasets.merfish() …")
    sp_full = sq.datasets.merfish()
    mask_sp = sp_full.obs["Cell_class"].astype(str).isin(["Excitatory", "Inhibitory"])
    spatial_src = sp_full[mask_sp].copy()
    spatial_src = _subsample_obs(spatial_src, args.max_spatial_cells, args.seed)

    print("Loading squidpy.datasets.sc_mouse_cortex() …")
    scr = sq.datasets.sc_mouse_cortex()
    labels_raw = scr.obs["cell_class"].astype(str).map(_ref_neuron_label)
    scr = scr[labels_raw.notna()].copy()
    scr.obs["cell_type"] = labels_raw[labels_raw.notna()].astype(str)
    scr = _subsample_obs(scr, args.max_ref_cells, args.seed + 1)

    shared = sorted(set(spatial_src.var_names) & set(scr.var_names))
    if len(shared) < 20:
        raise SystemExit(f"Too few shared genes ({len(shared)}); check datasets.")

    spatial_src = spatial_src[:, shared].copy()
    scr = scr[:, shared].copy()

    label_col = "cell_type"
    n = spatial_src.n_obs
    ix = np.arange(n)
    rng.shuffle(ix)
    n_ref = int(n * args.ref_fraction)
    ref_spatial_ix = ix[:n_ref]
    q_ix = ix[n_ref:]

    ref_sp = spatial_src[ref_spatial_ix].copy()
    query_full = spatial_src[q_ix].copy()

    ref = ref_sp.copy()
    query_full = query_full.copy()

    _lognorm_layers(ref, copy_raw_to_layer=True)
    ref.X = ref.layers["ln"].copy()
    _lognorm_layers(query_full, copy_raw_to_layer=True)
    query_full.X = query_full.layers["ln"].copy()

    ref_sc = scr.copy()
    _lognorm_layers(ref_sc, copy_raw_to_layer=True)
    ref_sc.X = ref_sc.layers["ln"].copy()

    ref.obs[label_col] = ref.obs["Cell_class"].astype(str)
    ref.obs["cluster_train"] = ref.obs[label_col].astype(str)

    q_tmp = query_full.copy()
    sc.pp.pca(q_tmp, n_comps=min(30, q_tmp.n_vars - 1), random_state=args.seed)
    sc.pp.neighbors(q_tmp, n_neighbors=15, use_rep="X_pca")
    sc.tl.leiden(q_tmp, resolution=0.5, key_added="leiden", random_state=args.seed)
    query_full.obs["leiden"] = q_tmp.obs["leiden"].astype(str)
    query_full.obs["cluster_train"] = query_full.obs["leiden"]

    true_labels = query_full.obs["Cell_class"].astype(str).to_numpy()
    query = query_full.copy()
    query.obs.drop(columns=["Cell_class"], inplace=True, errors="ignore")

    ref_combined = sc.concat(
        [ref, ref_sc],
        axis=0,
        join="outer",
        label="batch",
        keys=["spatial", "scrna"],
    )
    ref_combined.obs[label_col] = np.concatenate(
        [
            ref.obs[label_col].astype(str).values,
            ref_sc.obs[label_col].astype(str).values,
        ]
    )
    ref_combined.obs["cluster_train"] = np.concatenate(
        [
            ref.obs["cluster_train"].astype(str).values,
            ref_sc.obs[label_col].astype(str).values,
        ]
    )

    ref_path = os.path.join(args.outdir, "reference.h5ad")
    q_path = os.path.join(args.outdir, "query.h5ad")
    ref_combined.write_h5ad(ref_path)
    query.write_h5ad(q_path)

    marker_union: list[str] = []
    ref_tmp = ref_combined[ref_combined.obs["batch"] == "scrna"].copy()
    sc.tl.rank_genes_groups(ref_tmp, groupby=label_col, method="wilcoxon", n_genes=50, use_raw=False)
    cats = ref_tmp.obs[label_col].astype("category").cat.categories
    for ct in cats:
        df = sc.get.rank_genes_groups_df(ref_tmp, group=str(ct))
        sig = df[(df["pvals_adj"] < 0.05) & (df["logfoldchanges"] > 0.2)]
        marker_union.extend(sig.head(20)["names"].astype(str).tolist())
    marker_union = sorted(set(marker_union) & set(ref_combined.var_names) & set(query.var_names))
    genes_file = os.path.join(args.outdir, "malt_marker_genes_train.txt")
    with open(genes_file, "w") as f:
        f.write("\n".join(marker_union) + "\n")

    meta = {
        "spatial_dataset": "squidpy.datasets.merfish (Cell_class Excitatory|Inhibitory)",
        "reference_dataset": "squidpy.datasets.sc_mouse_cortex (Glutamatergic→Excitatory, GABAergic→Inhibitory)",
        "label_column": label_col,
        "ground_truth_column": "Cell_class",
        "training_cluster_obs": "cluster_train",
        "n_spatial_ref_cells": int(ref_sp.n_obs),
        "n_scrna_ref_cells": int(ref_sc.n_obs),
        "n_query": int(query.n_obs),
        "n_shared_genes": len(shared),
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
    results = {
        "accuracy_malt_baseline_vs_cell_class": acc_base,
        "dataset": "merfish_excitatory_inhibitory",
    }
    if args.train_spacetravlr:
        results["accuracy_malt_grn_vs_cell_class"] = _acc_from_out(malt_grn)
    with open(os.path.join(args.outdir, "benchmark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
