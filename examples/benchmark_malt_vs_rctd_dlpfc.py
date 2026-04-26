#!/usr/bin/env python3
"""
MALT vs SpaceTravLR built-in RCTD on the classic **Maynard et al. human DLPFC Visium** benchmark.

Spatial: 10x Visium filtered counts from the LieberInstitute **HumanPilot** S3 bucket
(`spatial-dlpfc`) plus `tissue_positions_list.txt` metadata from GitHub — same resource
used across RCTD / BayesSpace / ST tutorials.

Ground truth: manual cortical layer labels from `HumanPilot/10X/barcode_level_layer_map.tsv`
(Maynard et al.; layers L1–L6, WM, etc.).

Reference snRNA: **spatialLIBD `sce_DLPFC_annotated`** (Dropbox bundle: `se.rds` + `assays.h5`).
Default reference label column is **`layer_annotation`** (L1, L2, L3, WM, …) so MALT/RCTD
predictions live in the **same label space** as the Visium manual layer map. Override with
`--ref-label-col` (e.g. `cellType_layer`) if you want a different taxonomy (then layer
accuracy vs the puck map may not be meaningful without a custom mapping).

Writes `spatial.h5ad`, `reference.h5ad`, runs `scripts/malt_label_transfer.py` and
`spacetravlr --rctd`, then reports accuracy vs layer ground truth (exact label match for MALT;
RCTD: argmax of type weights).

Requires: scanpy, squidpy, anndata, numpy, scipy, pandas, h5py, rdata, torch, igraph, leidenalg.

Example:
  uv run --with scanpy --with squidpy --with anndata --with numpy --with scipy \\
    --with pandas --with h5py --with rdata --with torch --with igraph --with leidenalg \\
    examples/benchmark_malt_vs_rctd_dlpfc.py --outdir /tmp/dlpfc_malt_rctd \\
    --spacetravlr-bin spacetravlr --max-spots 1200 --max-ref-cells 8000

MALT uses Visium spot coordinates (merged from tissue_positions when using load_images=false)
and by default blends a spatially smoothed kNN prior; tune with --malt-spatial-knn-weight
and --malt-spatial-k-neighbors, or pass --malt-spatial-knn-weight 0 to disable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import squidpy as sq


def _repo_scripts_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return
    print(f"Downloading {url} → {dest}")
    urllib.request.urlretrieve(url, dest)


def _ensure_sce_zip(zip_path: str, default_url: str) -> None:
    if os.path.isfile(zip_path) and os.path.getsize(zip_path) > 1_000_000:
        return
    _download(default_url, zip_path)


def _extract_sce(zip_path: str, dest_root: str) -> tuple[str, str]:
    """
    Return (path_to_se.rds, path_to_assays.h5) inside dest_root.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        rds = next(n for n in names if n.endswith("se.rds") and not n.startswith("__MACOSX"))
        h5n = next(n for n in names if n.endswith("assays.h5") and not n.startswith("__MACOSX"))
        for n in (rds, h5n):
            out = os.path.join(dest_root, n)
            if not os.path.isfile(out):
                zf.extract(n, dest_root)
        return os.path.join(dest_root, rds), os.path.join(dest_root, h5n)


def _attach_visium_spatial_coords(ad: sc.AnnData, visium_root: str) -> None:
    """
    Squidpy's read_visium(..., load_images=False) returns before merging tissue positions,
    so obsm['spatial'] is missing. Layer transfer needs spot coordinates for a spatial prior.
    """
    root = Path(visium_root)
    tp = root / "spatial" / "tissue_positions.csv"
    if not tp.is_file():
        tp = root / "spatial" / "tissue_positions_list.csv"
    if not tp.is_file():
        return
    with open(tp) as f:
        first_cell = f.readline().split(",")[0].strip()
    has_header = first_cell.lower() == "barcode"
    coords = pd.read_csv(tp, header=0 if has_header else None, index_col=0)
    coords.columns = [
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_col_in_fullres",
        "pxl_row_in_fullres",
    ]
    coords.set_index(coords.index.astype(ad.obs.index.dtype), inplace=True)
    ad.obs = pd.merge(ad.obs, coords, how="left", left_index=True, right_index=True)
    ad.obsm["spatial"] = np.asarray(
        ad.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].values,
        dtype=np.float64,
    )
    ad.obs.drop(
        columns=["pxl_row_in_fullres", "pxl_col_in_fullres"],
        inplace=True,
        errors="ignore",
    )


def _load_visium_dlpfc(sample_id: str, work: str) -> sc.AnnData:
    h5_name = f"{sample_id}_filtered_feature_bc_matrix.h5"
    h5_url = f"https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/{h5_name}"
    base = f"https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/10X/{sample_id}"
    h5_path = os.path.join(work, h5_name)
    _download(h5_url, h5_path)
    spatial_dir = os.path.join(work, "spatial")
    os.makedirs(spatial_dir, exist_ok=True)
    for fn in ("tissue_positions_list.txt", "scalefactors_json.json"):
        _download(f"{base}/{fn}", os.path.join(spatial_dir, fn))
    tp_src = os.path.join(spatial_dir, "tissue_positions_list.txt")
    tp_dst = os.path.join(spatial_dir, "tissue_positions_list.csv")
    if not os.path.isfile(tp_dst):
        shutil.copy2(tp_src, tp_dst)
    ad = sq.read.visium(work, counts_file=h5_name, load_images=False)
    _attach_visium_spatial_coords(ad, work)
    ad.var_names_make_unique()
    return ad


def _load_layer_table(work: str) -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/10X/barcode_level_layer_map.tsv"
    path = os.path.join(work, "barcode_level_layer_map.tsv")
    _download(url, path)
    df = pd.read_csv(path, sep="\t", header=None, names=["barcode", "sample_id", "layer"])
    return df


def _annotate_layers(ad: sc.AnnData, sample_id: str, tab: pd.DataFrame) -> None:
    sub = tab[tab["sample_id"].astype(str) == str(sample_id)].copy()
    if sub.empty:
        raise SystemExit(f"No rows in layer map for sample_id={sample_id!r}")
    m = sub.set_index("barcode")["layer"].astype(str)
    obs_names = pd.Index(ad.obs_names.astype(str))
    ad.obs["layer_label"] = obs_names.map(lambda b: m.get(b, np.nan))
    ad.obs["layer_label"] = ad.obs["layer_label"].astype("string")


def _stratified_cell_indices(labels: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n = len(labels)
    types = pd.unique(labels)
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    per = max(1, max_n // max(1, len(types)))
    picks: list[int] = []
    for t in types:
        ix = np.where(labels == t)[0]
        k = min(len(ix), max(per, min(30, len(ix))))
        picks.extend(rng.choice(ix, size=k, replace=False).tolist())
    picks = np.unique(np.asarray(picks, dtype=np.int64))
    if len(picks) > max_n:
        picks = rng.choice(picks, size=max_n, replace=False)
    return np.sort(picks)


def _load_snrna_reference(
    rds_path: str,
    h5_path: str,
    *,
    label_col: str,
    max_cells: int,
    seed: int,
) -> sc.AnnData:
    import h5py
    import rdata

    sce = rdata.read_rda(rds_path)
    cd = sce.colData
    if label_col not in cd.listData:
        raise KeyError(f"{label_col!r} not in colData; keys include: {list(cd.listData.keys())[-8:]}")
    labels_raw = pd.Series(cd.listData[label_col])
    obs_names_full = np.asarray(cd.rownames, dtype=str)
    genes = np.asarray(sce.rowRanges.ranges.NAMES, dtype=str)
    s_lab = labels_raw.astype(str)
    valid_lab = ~(labels_raw.isna() | s_lab.str.lower().isin(["nan", "none", "<na>"]))
    valid_ix = np.where(valid_lab.to_numpy())[0]
    if valid_ix.size == 0:
        raise RuntimeError(f"No cells with finite {label_col!r} after dropping NaN labels")
    labels = s_lab.iloc[valid_ix].to_numpy()
    obs_names = obs_names_full[valid_ix]

    with h5py.File(h5_path, "r") as f:
        ds = f["assay001"]
        n_obs_h5, n_var_h5 = ds.shape
    if n_obs_h5 != len(obs_names_full) or n_var_h5 != len(genes):
        raise RuntimeError(
            f"HDF5 shape {(n_obs_h5, n_var_h5)} vs metadata ({len(obs_names_full)}, {len(genes)})"
        )

    keep_rel = _stratified_cell_indices(labels, max_cells, seed)
    keep_h5 = valid_ix[keep_rel]
    sub_lab = labels[keep_rel].copy()
    sub_obs = obs_names[keep_rel].copy()
    order = np.argsort(keep_h5, kind="mergesort")
    keep_s = keep_h5[order]
    sub_lab = sub_lab[order]
    sub_obs = sub_obs[order]
    n_sub = len(keep_s)
    indptr = np.zeros(n_sub + 1, dtype=np.int64)
    rows: list[np.ndarray] = []
    data: list[np.ndarray] = []
    pos = 0
    with h5py.File(h5_path, "r") as f:
        ds = f["assay001"]
        i = 0
        while i < n_sub:
            j = i + 1
            while j < n_sub and int(keep_s[j]) == int(keep_s[j - 1]) + 1:
                j += 1
            lo, hi = int(keep_s[i]), int(keep_s[j - 1]) + 1
            block = np.asarray(ds[lo:hi, :], dtype=np.float64)
            for k in range(j - i):
                row = block[k]
                nz = np.flatnonzero(row > 0)
                if nz.size:
                    rows.append(nz.astype(np.int32))
                    data.append(row[nz].astype(np.float32))
                    pos += nz.size
                indptr[i + k + 1] = pos
            i = j

    if pos == 0:
        raise RuntimeError("Reference subsample has no non-zero counts")
    rows_arr = np.concatenate(rows)
    data_arr = np.concatenate(data)
    x = sp.csr_matrix((data_arr, rows_arr, indptr), shape=(n_sub, len(genes)))
    ad = sc.AnnData(X=x)
    ad.obs_names = sub_obs
    ad.var_names = genes
    ad.obs[label_col] = pd.Categorical(sub_lab)
    return ad


def _accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    p = np.asarray(pred).astype(str)
    t = np.asarray(true).astype(str)
    m = ~(pd.Series(t).isin(["nan", "NaN", "None", ""]) | pd.Series(t).isna())
    if not m.any():
        return float("nan")
    return float((p[m.values] == t[m.values]).mean())


def _layer_token_set(s: str) -> set[str]:
    s = str(s).strip().replace("*", "")
    if not s or s.lower() in ("nan", "none", "<na>"):
        return set()
    parts = [x.strip() for x in s.split("/") if x.strip()]
    return set(parts) if parts else {s}


def _accuracy_layer_overlap(pred: np.ndarray, true: np.ndarray) -> float:
    """
    True if manual layer token(s) overlap reference-style composite labels, e.g.
    truth L3 vs pred L3/4 or L3/4* counts as correct.
    """
    p = np.asarray(pred).astype(str)
    t = np.asarray(true).astype(str)
    m = ~(pd.Series(t).isin(["nan", "NaN", "None", ""]) | pd.Series(t).isna())
    if not m.any():
        return float("nan")
    ok = 0
    tot = int(m.sum())
    for pi, ti in zip(p[m.values], t[m.values]):
        if pi == ti:
            ok += 1
            continue
        a, b = _layer_token_set(ti), _layer_token_set(pi)
        if a and b and (a & b):
            ok += 1
    return float(ok / tot)


def _read_rctd_weights_csv(path: str) -> tuple[np.ndarray, list[str]]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f"empty CSV header: {path}")
        type_cols = [c for c in r.fieldnames if c and c != "obs"]
        rows = []
        for row in r:
            vec = [float(row[c]) for c in type_cols]
            rows.append(vec)
    w = np.asarray(rows, dtype=np.float64)
    pred = np.array([type_cols[i] for i in np.argmax(w, axis=1)], dtype=object)
    return pred, type_cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample-id", default="151673", help="HumanPilot Visium sample id (default 151673)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-spots", type=int, default=0, help="0 = all in-tissue spots with a layer label")
    ap.add_argument("--max-ref-cells", type=int, default=8000)
    ap.add_argument(
        "--ref-label-col",
        default="layer_annotation",
        help="colData column in sce_DLPFC_annotated (default layer_annotation ≈ Visium manual layers)",
    )
    ap.add_argument(
        "--sce-zip",
        default=None,
        help="Local path to sce_DLPFC_annotated.zip (downloaded if missing)",
    )
    ap.add_argument(
        "--sce-zip-url",
        default="https://www.dropbox.com/s/5919zt00vm1ht8e/sce_DLPFC_annotated.zip?dl=1",
    )
    ap.add_argument("--spacetravlr-bin", default="spacetravlr")
    ap.add_argument("--rctd-k-val", type=int, default=100, help="Match spacexr default K_val where possible")
    ap.add_argument("--rctd-ref-cell-min", type=int, default=20)
    ap.add_argument("--rctd-batch-size", type=int, default=256)
    ap.add_argument("--skip-malt", action="store_true")
    ap.add_argument("--skip-rctd", action="store_true")
    ap.add_argument(
        "--malt-spatial-key",
        default="spatial",
        help="query.obsm key for Visium pixel coords (default spatial). Empty string disables.",
    )
    ap.add_argument(
        "--malt-spatial-knn-weight",
        type=float,
        default=0.78,
        help="Blend spatial smoothing into MALT's kNN prior; 0 disables (default 0.78 for cortical layers).",
    )
    ap.add_argument(
        "--malt-spatial-k-neighbors",
        type=int,
        default=18,
        help="Spatial neighbors for smoothing the expression kNN prior (default 18).",
    )
    ap.add_argument(
        "--malt-alpha-k",
        type=float,
        default=1.4,
        help="MALT KL weight toward the (spatially smoothed) kNN prior (default 1.4).",
    )
    ap.add_argument(
        "--malt-init-knn-mix",
        type=float,
        default=0.88,
        help="Init blend toward kNN prior vs marker softmax (default 0.88).",
    )
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    work = os.path.join(args.outdir, "_work")
    os.makedirs(work, exist_ok=True)

    zip_path = args.sce_zip or os.path.join(work, "sce_DLPFC_annotated.zip")
    _ensure_sce_zip(zip_path, args.sce_zip_url)
    sce_root = os.path.join(work, "sce_extract")
    os.makedirs(sce_root, exist_ok=True)
    rds_path, h5_path = _extract_sce(zip_path, sce_root)

    print("Loading Visium + layer labels…")
    vis_dir = os.path.join(work, f"visium_{args.sample_id}")
    os.makedirs(vis_dir, exist_ok=True)
    spatial = _load_visium_dlpfc(args.sample_id, vis_dir)
    layers = _load_layer_table(work)
    _annotate_layers(spatial, args.sample_id, layers)
    spatial = spatial[spatial.obs["layer_label"].notna()].copy()
    spatial = spatial[spatial.obs["layer_label"].astype(str).str.lower() != "nan"].copy()
    if args.max_spots and spatial.n_obs > args.max_spots:
        ix = np.random.default_rng(args.seed).choice(
            spatial.n_obs, size=args.max_spots, replace=False
        )
        spatial = spatial[ix].copy()

    print("Building snRNA reference from sce_DLPFC_annotated…")
    reference = _load_snrna_reference(
        rds_path,
        h5_path,
        label_col=args.ref_label_col,
        max_cells=args.max_ref_cells,
        seed=args.seed + 7,
    )
    reference.var_names_make_unique()

    q = spatial.copy()
    true_layers = q.obs["layer_label"].astype(str).to_numpy()
    q.obs.drop(columns=["layer_label"], inplace=True, errors="ignore")
    q.var_names_make_unique()

    ref_path = os.path.join(args.outdir, "reference.h5ad")
    q_path = os.path.join(args.outdir, "spatial.h5ad")
    def _dense_f64_for_rctd(ad: sc.AnnData) -> sc.AnnData:
        x = ad.X
        if sp.issparse(x):
            ad.X = np.asarray(x.toarray(), dtype=np.float64)
        else:
            ad.X = np.asarray(x, dtype=np.float64)
        return ad

    _dense_f64_for_rctd(reference).write_h5ad(ref_path)
    _dense_f64_for_rctd(q.copy()).write_h5ad(q_path)

    meta = {
        "spatial_sample": args.sample_id,
        "n_spots": int(q.n_obs),
        "n_ref_cells": int(reference.n_obs),
        "n_genes_shared": int(len(set(reference.var_names) & set(q.var_names))),
        "ref_label_col": args.ref_label_col,
        "reference_types": sorted(pd.unique(reference.obs[args.ref_label_col].astype(str))),
    }
    meta["n_genes_ref"] = int(reference.n_vars)
    meta["n_genes_query"] = int(q.n_vars)
    with open(os.path.join(args.outdir, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    results: dict = {}

    malt_py = os.path.join(_repo_scripts_dir(), "malt_label_transfer.py")
    if not os.path.isfile(malt_py):
        raise FileNotFoundError(malt_py)
    malt_out = os.path.join(args.outdir, "malt_out")

    if not args.skip_malt:
        shutil.rmtree(malt_out, ignore_errors=True)
        cmd = [
            sys.executable,
            malt_py,
            "--reference",
            ref_path,
            "--query",
            q_path,
            "--outdir",
            malt_out,
            "-g",
            args.ref_label_col,
            "--expression-mode",
            "auto",
        ]
        sk = (args.malt_spatial_key or "").strip()
        if sk and args.malt_spatial_knn_weight > 0.0:
            cmd.extend(
                [
                    "--spatial-key",
                    sk,
                    "--spatial-knn-weight",
                    str(args.malt_spatial_knn_weight),
                    "--spatial-k-neighbors",
                    str(args.malt_spatial_k_neighbors),
                ]
            )
        cmd.extend(
            [
                "--alpha-k",
                str(args.malt_alpha_k),
                "--init-knn-mix",
                str(args.malt_init_knn_mix),
            ]
        )
        print("Running MALT…")
        subprocess.run(cmd, check=True)
        ql = sc.read_h5ad(os.path.join(malt_out, "query_labeled.h5ad"))
        pred_m = ql.obs["malt_label"].astype(str).values
        results["accuracy_malt_vs_manual_layer"] = _accuracy(pred_m, true_layers)
        results["accuracy_malt_vs_manual_layer_overlap"] = _accuracy_layer_overlap(
            pred_m, true_layers
        )
    else:
        results["accuracy_malt_vs_manual_layer"] = None
        results["accuracy_malt_vs_manual_layer_overlap"] = None

    if not args.skip_rctd:
        rctd_prefix = os.path.join(args.outdir, "rctd_out")
        weights_csv = rctd_prefix + ".weights.csv"
        if os.path.isfile(weights_csv):
            os.remove(weights_csv)
        cmd = [
            args.spacetravlr_bin,
            "--rctd",
            "--h5ad",
            q_path,
            "--ref-adata",
            ref_path,
            "--cell-type-col",
            args.ref_label_col,
            "--rctd-output",
            rctd_prefix,
            "--rctd-k-val",
            str(args.rctd_k_val),
            "--ref-cell-min",
            str(args.rctd_ref_cell_min),
            "--rctd-batch-size",
            str(args.rctd_batch_size),
        ]
        print("Running RCTD (spacetravlr --rctd)…")
        subprocess.run(cmd, check=True)
        pred_r, _rctd_types = _read_rctd_weights_csv(weights_csv)
        if len(pred_r) != len(true_layers):
            raise RuntimeError(
                f"RCTD rows {len(pred_r)} != query rows {len(true_layers)}"
            )
        results["accuracy_rctd_vs_manual_layer"] = _accuracy(pred_r, true_layers)
        results["accuracy_rctd_vs_manual_layer_overlap"] = _accuracy_layer_overlap(
            pred_r, true_layers
        )
    else:
        results["accuracy_rctd_vs_manual_layer"] = None
        results["accuracy_rctd_vs_manual_layer_overlap"] = None

    out_json = os.path.join(args.outdir, "benchmark_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
