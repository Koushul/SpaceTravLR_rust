from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import magic
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


GENES = 160
GENES_PER_CELL = 120
N_CLUSTERS = 4
PY_TIMEOUT_SECONDS = int(os.environ.get("SCALE_BENCH_PY_TIMEOUT", "300"))
RUST_TIMEOUT_SECONDS = int(os.environ.get("SCALE_BENCH_RUST_TIMEOUT", "300"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def make_h5ad(path: Path, n_obs: int) -> dict:
    t0 = time.perf_counter()
    rng = np.random.default_rng(12345 + n_obs)
    indptr = np.arange(0, (n_obs + 1) * GENES_PER_CELL, GENES_PER_CELL, dtype=np.int64)
    panels = []
    for c in range(N_CLUSTERS):
        start = (c * 30) % GENES
        panel = (np.arange(start, start + GENES_PER_CELL, dtype=np.int32) % GENES).astype(np.int32)
        panels.append(panel)
    chunks = []
    chunk_rows = 50_000
    for lo in range(0, n_obs, chunk_rows):
        hi = min(n_obs, lo + chunk_rows)
        rows = np.arange(lo, hi, dtype=np.int64)
        cluster = rows % N_CLUSTERS
        idx = np.empty((hi - lo, GENES_PER_CELL), dtype=np.int32)
        for c in range(N_CLUSTERS):
            idx[cluster == c, :] = panels[c]
        chunks.append(idx.reshape(-1))
    indices = np.concatenate(chunks)
    data = rng.poisson(2.0, size=indices.size).astype(np.float32) + 1.0
    x = sp.csr_matrix((data, indices, indptr), shape=(n_obs, GENES))
    obs = pd.DataFrame(index=pd.Index([f"cell_{i}" for i in range(n_obs)], dtype=object))
    var = pd.DataFrame(index=pd.Index([f"G{i}" for i in range(GENES)], dtype=object))
    a = ad.AnnData(X=x, obs=obs, var=var)
    coords = np.empty((n_obs, 2), dtype=np.float32)
    rows = np.arange(n_obs)
    cluster = rows % N_CLUSTERS
    coords[:, 0] = cluster * 20.0 + rng.normal(0, 1, n_obs)
    coords[:, 1] = cluster * 8.0 + rng.normal(0, 1, n_obs)
    a.obsm["spatial"] = coords
    ad.settings.allow_write_nullable_strings = True
    a.write_h5ad(path)
    return {"seconds": time.perf_counter() - t0, "nnz": int(x.nnz)}


def timed(steps: dict[str, float], name: str, fn):
    t0 = time.perf_counter()
    result = fn()
    steps[name] = time.perf_counter() - t0
    print(f"<<< {name}: {steps[name]:.4f} s", flush=True)
    return result


def run_scanpy(input_h5ad: Path, output_h5ad: Path) -> dict:
    started = time.perf_counter()
    steps: dict[str, float] = {}
    ad.settings.allow_write_nullable_strings = True
    a = timed(steps, "read_h5ad", lambda: ad.read_h5ad(input_h5ad))
    timed(steps, "filter_cells(min_genes=100)", lambda: sc.pp.filter_cells(a, min_genes=100))
    timed(steps, "filter_genes(min_cells=3)", lambda: sc.pp.filter_genes(a, min_cells=3))
    a.layers["raw_count"] = a.X.copy()
    timed(steps, "normalize_total", lambda: sc.pp.normalize_total(a, target_sum=1e4))
    a.layers["normalized_count"] = a.X.copy()
    timed(steps, "log1p", lambda: sc.pp.log1p(a))
    timed(steps, "scale(max_value=10)", lambda: sc.pp.scale(a, max_value=10))
    timed(steps, "highly_variable_genes(120)", lambda: sc.pp.highly_variable_genes(a, n_top_genes=120))
    timed(steps, "pca(n_comps=50)", lambda: sc.pp.pca(a, n_comps=50))
    timed(steps, "neighbors(n_neighbors=10)", lambda: sc.pp.neighbors(a, n_neighbors=10))
    timed(steps, "umap", lambda: sc.tl.umap(a))
    timed(steps, "leiden", lambda: sc.tl.leiden(a, flavor="igraph", n_iterations=2))
    a.obs["cell_type"] = a.obs["leiden"].astype(str)

    def magic_clusterwise() -> None:
        nc = a.layers["normalized_count"]
        x = nc.toarray().astype(float) if sp.issparse(nc) else np.asarray(nc, dtype=float)
        out = x.copy()
        labels = np.asarray(a.obs["cell_type"].astype(str))
        for lab in np.unique(labels):
            mask = labels == lab
            n_sub = int(mask.sum())
            if n_sub < 2:
                continue
            sub = x[mask]
            active = sub.sum(axis=0) > 0
            n_genes = int(active.sum())
            if n_genes == 0:
                continue
            op = magic.MAGIC(
                knn=min(3, max(1, n_sub - 1)),
                knn_max=min(6, max(1, n_sub - 1)),
                decay=1,
                t=2,
                n_pca=min(12, max(1, min(n_sub, n_genes) - 1)),
                verbose=0,
            )
            imp = sub.copy()
            imp[:, active] = np.asarray(op.fit_transform(sub[:, active], genes="all_genes"), dtype=float)
            out[mask] = imp
        a.layers["imputed_count"] = out

    timed(steps, "MAGIC per cell_type", magic_clusterwise)

    def write() -> None:
        a.X = sp.csr_matrix(a.X)
        for key in list(a.layers.keys()):
            a.layers[key] = sp.csr_matrix(a.layers[key])
        a.write_h5ad(output_h5ad)

    timed(steps, "write_h5ad", write)
    return {
        "status": "ok",
        "wall_seconds": time.perf_counter() - started,
        "steps": steps,
        "output": str(output_h5ad),
        "scanpy_version": sc.__version__,
    }


def parse_rust_log(text: str) -> dict[str, float]:
    steps = {}
    for line in text.splitlines():
        m = re.match(r"<<< (.*): ([0-9.]+) s$", line.strip())
        if m:
            steps[m.group(1)] = float(m.group(2))
    return steps


def run_rust(input_h5ad: Path, output_dir: Path, binary: Path) -> dict:
    started = time.perf_counter()
    cmd = [
        str(binary),
        "--rust-process-h5ad",
        "--h5ad",
        str(input_h5ad),
        "--rust-n-top-hvg",
        "120",
        "--rust-n-neighbors",
        "10",
        "--process-output-dir",
        str(output_dir),
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=RUST_TIMEOUT_SECONDS)
    status = "ok" if proc.returncode == 0 else "failed"
    return {
        "status": status,
        "returncode": proc.returncode,
        "wall_seconds": time.perf_counter() - started,
        "steps": parse_rust_log(proc.stdout),
        "log_tail": proc.stdout.splitlines()[-40:],
    }


def run_one_size(n_obs: int, root: Path, binary: Path) -> dict:
    size_dir = root / f"scale_{n_obs}"
    size_dir.mkdir(parents=True, exist_ok=True)
    input_h5ad = size_dir / "input.h5ad"
    result = {"n_cells": n_obs, "n_genes": GENES, "genes_per_cell": GENES_PER_CELL}
    if not input_h5ad.exists():
        result["generate"] = make_h5ad(input_h5ad, n_obs)
    else:
        result["generate"] = {"seconds": 0.0, "reused": True}

    py_out = size_dir / "python_scanpy_processed.h5ad"
    try:
        result["python_scanpy"] = run_scanpy(input_h5ad, py_out)
    except subprocess.TimeoutExpired:
        result["python_scanpy"] = {"status": "timeout", "timeout_seconds": PY_TIMEOUT_SECONDS}
    except Exception as exc:
        result["python_scanpy"] = {"status": "failed", "error": repr(exc)}

    try:
        result["rust"] = run_rust(input_h5ad, size_dir, binary)
    except subprocess.TimeoutExpired:
        result["rust"] = {"status": "timeout", "timeout_seconds": RUST_TIMEOUT_SECONDS}
    except Exception as exc:
        result["rust"] = {"status": "failed", "error": repr(exc)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    parser.add_argument("--root", type=Path, default=Path("tmp_bench_h5"))
    parser.add_argument("--binary", type=Path, default=Path("/tmp/spacetravlr_target/release/spacetravlr"))
    parser.add_argument("--out", type=Path, default=Path("tmp_bench_h5/process_scale_benchmark.json"))
    parser.add_argument("--skip-million-full", action="store_true")
    args = parser.parse_args()

    all_results = {
        "config": {
            "genes": GENES,
            "genes_per_cell": GENES_PER_CELL,
            "python_timeout_seconds": PY_TIMEOUT_SECONDS,
            "rust_timeout_seconds": RUST_TIMEOUT_SECONDS,
        },
        "results": [],
    }
    if args.out.exists():
        all_results = json.loads(args.out.read_text())

    done = {row["n_cells"] for row in all_results.get("results", [])}
    for n_obs in args.sizes:
        if n_obs in done:
            continue
        if args.skip_million_full and n_obs >= 1_000_000:
            all_results["results"].append(
                {
                    "n_cells": n_obs,
                    "n_genes": GENES,
                    "genes_per_cell": GENES_PER_CELL,
                    "python_scanpy": {"status": "skipped_resource_guard", "reason": "full Scanpy UMAP/MAGIC on 1M cells exceeded benchmark guard"},
                    "rust": {"status": "skipped_resource_guard", "reason": "full Rust UMAP/MAGIC on 1M cells exceeded benchmark guard"},
                }
            )
            write_json(args.out, all_results)
            continue
        row = run_one_size(n_obs, args.root, args.binary)
        all_results["results"].append(row)
        write_json(args.out, all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
