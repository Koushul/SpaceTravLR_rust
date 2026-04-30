#!/usr/bin/env python3
"""
Synthetic AnnData benchmark: training (spacetravlr release binary) at 1k / 10k / 100k cells,
Rust --parallel vs Python multiprocess, and Rust vs Python AnnData preprocessing.

Writes JSON (default: benchmarks/rust_python_step_bench/results/bench_results.json).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class BenchResult:
    meta: dict[str, Any] = field(default_factory=dict)
    training: list[dict[str, Any]] = field(default_factory=list)
    preprocess: list[dict[str, Any]] = field(default_factory=list)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_minimal_mouse_network_parquet(path: Path) -> None:
    rows = []
    for i in range(10):
        rows.append((f"Reg{i}", f"Tg{i}", "grn", 1.0))
    for i in range(20):
        rows.append((f"L{i}", f"R{i}", "lr", 1.0))
    df = pd.DataFrame(rows, columns=["source", "target", "edge_type", "weight"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _make_synthetic_h5ad(
    path: Path,
    n_cells: int,
    n_genes: int,
    n_clusters: int,
    seed: int,
) -> None:
    import anndata as ad

    rng = np.random.default_rng(seed)
    names = [f"Tg{i}" for i in range(min(10, n_genes))] + [
        f"G{i}" for i in range(max(0, n_genes - 10))
    ]
    names = names[:n_genes]
    nnz_per_cell = min(n_genes, max(150, int(0.25 * n_genes)))
    cols = rng.integers(0, n_genes, size=(n_cells, nnz_per_cell), dtype=np.int32)
    for i in range(n_cells):
        if len(np.unique(cols[i])) < nnz_per_cell:
            cols[i] = rng.choice(n_genes, size=nnz_per_cell, replace=False)
    vals = np.exp(rng.normal(0.5, 0.5, size=(n_cells, nnz_per_cell))).astype(np.float32)
    rows = np.repeat(np.arange(n_cells, dtype=np.int32), nnz_per_cell)
    flat_c = cols.ravel()
    flat_d = vals.ravel()
    X = sp.csr_matrix((flat_d, (rows, flat_c)), shape=(n_cells, n_genes), dtype=np.float32)
    xy = rng.normal(size=(n_cells, 2)).astype(np.float32) * 500.0
    clusters = rng.integers(0, n_clusters, size=n_cells)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(clusters.astype(str)),
        },
        index=[f"cell{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=names)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = xy
    adata.layers["imputed_count"] = X.copy()
    if "log1p" in adata.uns:
        del adata.uns["log1p"]
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path, compression=None)


def _write_spaceship_config(
    path: Path,
    *,
    adata_path: Path,
    network_dir: Path,
    output_dir: Path,
) -> None:
    text = f"""
[data]
adata_path = "{adata_path.as_posix()}"
layer = "imputed_count"
cluster_annot = "cell_type"

[spatial]
radius = 200.0
spatial_dim = 24
contact_distance = 50.0
weighted_ligand_scale_factor = 1.0

[grn]
network_data_dir = "{network_dir.as_posix()}"
tf_ligand_cutoff = 0.05
max_ligands = 32
use_tf_modulators = true
use_lr_modulators = true
use_tfl_modulators = false

[cnn]
spatial_feature_radius = 200.0

[training]
mode = "seed"
epochs = 0
seed_only = true

[lasso]
n_iter = 80

[execution]
output_dir = "{output_dir.as_posix()}"
n_parallel = 1
write_minimal_repro_h5ad = false
"""
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> tuple[int, str, str, float]:
    t0 = time.perf_counter()
    merged = os.environ.copy()
    if env:
        merged.update(env)
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    dt = time.perf_counter() - t0
    return p.returncode, p.stdout, p.stderr, dt


def _parse_rust_preprocess_steps(stderr: str) -> dict[str, float]:
    steps: dict[str, float] = {}
    for line in stderr.splitlines():
        m = re.search(r"<<<\s*([^:]+):\s*([0-9.]+)\s*s\s*$", line)
        if m:
            steps[m.group(1).strip()] = float(m.group(2))
            continue
        mw = re.search(r"<<<\s*write_h5ad\s+[^:]+:\s*([0-9.]+)\s*s\s*$", line)
        if mw:
            steps["write_h5ad"] = float(mw.group(1))
            continue
        m2 = re.match(r"\s{2,}([^:]+):\s*([0-9.]+)\s*s\s*$", line)
        if m2 and "rust_preprocess" not in line and "TOTAL" not in line:
            steps[m2.group(1).strip()] = float(m2.group(2))
    return steps


def _rust_compute_sum_excl_write(steps: dict[str, Any]) -> float:
    s = 0.0
    for k, v in steps.items():
        if k == "write_h5ad" or k.startswith("write_h5ad_"):
            continue
        if isinstance(v, (int, float)):
            s += float(v)
    return s


def _train_one_gene_subprocess(args: tuple[Path, Path, Path, str, str]) -> dict[str, Any]:
    exe, base_cfg, h5ad, gene, force_cpu_s = args
    run_dir = Path(tempfile.mkdtemp(prefix="stlr_g_"))
    cfg_path = run_dir / "spaceship.toml"
    try:
        txt = Path(base_cfg).read_text(encoding="utf-8")
        txt_new, n = re.subn(
            r'output_dir\s*=\s*"[^"]*"',
            f'output_dir = "{run_dir.as_posix()}"',
            txt,
            count=1,
        )
        if n != 1:
            return {
                "gene": gene,
                "returncode": 99,
                "wall_s": 0.0,
                "stderr_tail": f"could not patch output_dir in config (replacements={n})",
            }
        cfg_path.write_text(txt_new, encoding="utf-8")
        cmd = [
            str(exe),
            "--plain",
            "--skip-auto-adata-prep",
            "--config",
            str(cfg_path),
            "--h5ad",
            str(h5ad),
            "--output-dir",
            str(run_dir),
            "--training-mode",
            "seed",
            "--epochs",
            "0",
            "--parallel",
            "1",
            "--genes",
            gene,
            "--max-ligands",
            "32",
            "--n-iter",
            "80",
        ]
        env: dict[str, str] = {}
        if force_cpu_s:
            env["SPACETRAVLR_FORCE_CPU"] = force_cpu_s
        run_env = os.environ.copy()
        run_env.update(env)
        code, _stdout, stderr, wall = _run_cmd(cmd, env=run_env, timeout=7200.0)
        return {
            "gene": gene,
            "returncode": code,
            "wall_s": wall,
            "stderr_tail": stderr[-4000:] if stderr else "",
        }
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def _python_mp_genes_wall(
    exe: Path,
    cfg: Path,
    h5ad: Path,
    genes: list[str],
    n_workers: int,
    force_cpu: bool,
) -> dict[str, Any]:
    n_workers = max(1, min(n_workers, len(genes)))
    t0 = time.perf_counter()
    errs: list[str] = []
    fc = "1" if force_cpu else ""
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = []
        for g in genes:
            futs.append(
                ex.submit(
                    _train_one_gene_subprocess,
                    (exe, cfg, h5ad, g, fc),
                )
            )
        for f in as_completed(futs):
            r = f.result()
            if r["returncode"] != 0:
                errs.append(f"{r['gene']}: rc={r['returncode']}\n{r.get('stderr_tail','')}")
    wall = time.perf_counter() - t0
    return {"wall_s": wall, "n_workers": n_workers, "errors": errs}


def _timed_python_preprocess_like_scanpy(h5ad_in: Path) -> dict[str, Any]:
    import anndata as ad
    import scanpy as sc

    sc.settings.verbosity = 0
    steps: dict[str, float] = {}
    t0 = time.perf_counter()
    adata = ad.read_h5ad(h5ad_in)
    steps["read_h5ad"] = time.perf_counter() - t0
    t = time.perf_counter()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    steps["filter_cells_and_genes"] = time.perf_counter() - t
    t = time.perf_counter()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    steps["normalize_total_log1p"] = time.perf_counter() - t
    t = time.perf_counter()
    nhvg = min(2000, max(2, int(adata.n_vars) - 1))
    sc.pp.highly_variable_genes(adata, n_top_genes=nhvg)
    steps["highly_variable_genes"] = time.perf_counter() - t
    t = time.perf_counter()
    adata = adata[:, adata.var["highly_variable"]].copy()
    steps["subset_hvg"] = time.perf_counter() - t
    t = time.perf_counter()
    sc.tl.pca(adata, n_comps=min(50, adata.n_obs - 1, adata.n_vars))
    steps["pca"] = time.perf_counter() - t
    t = time.perf_counter()
    no = int(adata.n_obs) - 1
    n_nb = min(15, max(2, no))
    sc.pp.neighbors(adata, n_neighbors=n_nb, use_rep="X_pca")
    steps["neighbors"] = time.perf_counter() - t
    t = time.perf_counter()
    sc.tl.umap(adata)
    steps["umap"] = time.perf_counter() - t
    total = sum(steps.values())
    steps["sum_steps_s"] = total
    return {"steps": steps, "note": "In-process Scanpy stages (no MAGIC); approximates core of rust_preprocess for step comparison."}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacetravlr", type=Path, default=None, help="Path to spacetravlr binary")
    ap.add_argument("--cells", default="1000,10000,100000", help="Comma-separated cell counts")
    ap.add_argument("--parallel", type=int, default=0, help="Rust/Python worker count (0 = CPU count)")
    ap.add_argument("--genes", default="Tg0", help="Comma-separated target genes for multi-gene runs")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument(
        "--process-h5ad-max-cells",
        type=int,
        default=10_000,
        help="Run spacetravlr --process-h5ad (uv Scanpy+MAGIC) only when n_cells <= this (default 10000).",
    )
    ap.add_argument(
        "--skip-process-h5ad",
        action="store_true",
        help="Skip spacetravlr --process-h5ad (uv Scanpy+MAGIC CLI).",
    )
    ap.add_argument("--force-cpu", action="store_true", default=True)
    ap.add_argument("--no-force-cpu", action="store_false", dest="force_cpu")
    args = ap.parse_args()

    repo = _repo_root()
    exe = args.spacetravlr or (repo / "target" / "release" / "spacetravlr")
    if not exe.is_file():
        print(f"Missing spacetravlr binary at {exe}; build with: cargo build --release --bin spacetravlr", file=sys.stderr)
        sys.exit(2)

    par = args.parallel or max(1, (os.cpu_count() or 4))
    out_json = args.out_json or (
        repo / "benchmarks" / "rust_python_step_bench" / "results" / "bench_results.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cell_grid = [int(x.strip()) for x in args.cells.split(",") if x.strip()]
    gene_list = [g.strip() for g in args.genes.split(",") if g.strip()]
    ten_genes = [f"Tg{i}" for i in range(10)]

    result = BenchResult(
        meta={
            "spacetravlr": str(exe.resolve()),
            "parallel_workers": par,
            "cell_counts": cell_grid,
            "single_gene": gene_list[0] if gene_list else "Tg0",
            "ten_genes": ten_genes,
            "force_cpu": bool(args.force_cpu),
            "process_h5ad_max_cells": args.process_h5ad_max_cells,
            "skip_process_h5ad_cli": bool(args.skip_process_h5ad),
            "synthetic_adata": "mouse_network.parquet stub; 500 genes; CSR X with ~150 nnz/cell (passes filter_cells min_genes=100); imputed_count=X; spatial in obsm; cell_type int clusters. Training uses --skip-auto-adata-prep.",
        }
    )

    tmp_root = Path(tempfile.mkdtemp(prefix="stlr_bench_", dir=str(out_json.parent)))

    try:
        net_dir = tmp_root / "network"
        _write_minimal_mouse_network_parquet(net_dir / "mouse_network.parquet")

        for n_cells in cell_grid:
            h5ad = tmp_root / f"synth_{n_cells}.h5ad"
            _make_synthetic_h5ad(h5ad, n_cells=n_cells, n_genes=500, n_clusters=5, seed=42)

            cfg = tmp_root / f"spaceship_{n_cells}.toml"
            out_train = tmp_root / f"run_{n_cells}"
            out_train.mkdir(parents=True, exist_ok=True)
            _write_spaceship_config(cfg, adata_path=h5ad, network_dir=net_dir, output_dir=out_train)

            env_base: dict[str, str] = {}
            if args.force_cpu:
                env_base["SPACETRAVLR_FORCE_CPU"] = "1"
            env_base["SPACETRAVLR_DATA_DIR"] = str(net_dir)
            os.environ["SPACETRAVLR_DATA_DIR"] = str(net_dir)
            if args.force_cpu:
                os.environ["SPACETRAVLR_FORCE_CPU"] = "1"

            def train_cmd(extra: list[str]) -> tuple[int, float, str]:
                cmd = [
                    str(exe),
                    "--plain",
                    "--skip-auto-adata-prep",
                    "--config",
                    str(cfg),
                    "--h5ad",
                    str(h5ad),
                    "--output-dir",
                    str(out_train),
                    "--training-mode",
                    "seed",
                    "--epochs",
                    "0",
                    "--max-ligands",
                    "32",
                    "--n-iter",
                    "80",
                    *extra,
                ]
                code, _o, err, wall = _run_cmd(cmd, env=env_base, timeout=7200.0)
                shutil.rmtree(out_train, ignore_errors=True)
                out_train.mkdir(parents=True, exist_ok=True)
                return code, wall, err

            g1 = gene_list[0] if gene_list else "Tg0"
            rc, w, err = train_cmd(
                ["--parallel", "1", "--genes", g1],
            )
            row = {
                "n_cells": n_cells,
                "scenario": "rust_train_1_gene_parallel_1",
                "returncode": rc,
                "wall_s": w,
            }
            if rc != 0:
                row["stderr_tail"] = err[-8000:]
            result.training.append(row)

            rc, w, err = train_cmd(
                ["--parallel", str(par), "--genes", g1],
            )
            result.training.append(
                {
                    "n_cells": n_cells,
                    "scenario": "rust_train_1_gene_parallel_N",
                    "parallel": par,
                    "returncode": rc,
                    "wall_s": w,
                    **({"stderr_tail": err[-8000:]} if rc != 0 else {}),
                }
            )

            genes_csv = ",".join(ten_genes)
            rc, w, err = train_cmd(
                ["--parallel", "1", "--genes", genes_csv],
            )
            result.training.append(
                {
                    "n_cells": n_cells,
                    "scenario": "rust_train_10_genes_parallel_1",
                    "returncode": rc,
                    "wall_s": w,
                    **({"stderr_tail": err[-8000:]} if rc != 0 else {}),
                }
            )

            rc, w, err = train_cmd(
                ["--parallel", str(par), "--genes", genes_csv],
            )
            result.training.append(
                {
                    "n_cells": n_cells,
                    "scenario": "rust_train_10_genes_parallel_N",
                    "parallel": par,
                    "returncode": rc,
                    "wall_s": w,
                    **({"stderr_tail": err[-8000:]} if rc != 0 else {}),
                }
            )

            mp = _python_mp_genes_wall(exe, cfg, h5ad, ten_genes, par, bool(args.force_cpu))
            result.training.append(
                {
                    "n_cells": n_cells,
                    "scenario": "python_mp_10_subprocesses_1_gene_each",
                    "parallel_workers": mp["n_workers"],
                    "wall_s": mp["wall_s"],
                    "errors": mp["errors"],
                }
            )

            proc_out = tmp_root / f"proc_rust_{n_cells}"
            proc_out.mkdir(parents=True, exist_ok=True)
            stem = h5ad.stem
            rust_out = proc_out / f"{stem}_rust_processed.h5ad"
            t0 = time.perf_counter()
            rc_r, _o, err_r, _ = _run_cmd(
                [
                    str(exe),
                    "--plain",
                    "--rust-process-h5ad",
                    "--h5ad",
                    str(h5ad),
                    "--process-output-dir",
                    str(proc_out),
                ],
                env=env_base,
                timeout=7200.0,
            )
            wall_rust = time.perf_counter() - t0
            rust_steps = _parse_rust_preprocess_steps(err_r)
            rust_write_ok = rc_r == 0
            if rust_out.is_file():
                rust_steps["write_h5ad_reported_s"] = rust_steps.get("write_h5ad", 0.0)
                if not rust_write_ok:
                    rust_steps["write_h5ad_note"] = (
                        "spacetravlr exited non-zero at HDF5 finalize; output file may still exist and be readable."
                    )
            pre_row: dict[str, Any] = {
                "n_cells": n_cells,
                "pipeline": "rust_process_h5ad",
                "returncode": rc_r,
                "wall_s": wall_rust,
                "steps_s": rust_steps,
            }
            if rust_out.is_file():
                pre_row["output_path"] = str(rust_out)
                pre_row["output_bytes"] = rust_out.stat().st_size
            if rc_r != 0:
                pre_row["stderr_tail"] = err_r[-1500:]
            pre_row["rust_compute_sum_steps_s"] = _rust_compute_sum_excl_write(rust_steps)
            result.preprocess.append(pre_row)

            try:
                py_steps = _timed_python_preprocess_like_scanpy(h5ad)
            except Exception as e:
                py_steps = {"error": repr(e)}
            result.preprocess.append(
                {
                    "n_cells": n_cells,
                    "pipeline": "python_scanpy_in_process_steps",
                    **py_steps,
                }
            )

            if (
                not args.skip_process_h5ad
                and n_cells <= args.process_h5ad_max_cells
            ):
                proc_py = tmp_root / f"proc_py_{n_cells}"
                proc_py.mkdir(parents=True, exist_ok=True)
                t1 = time.perf_counter()
                rc_p, _o, err_p, _ = _run_cmd(
                    [
                        str(exe),
                        "--plain",
                        "--process-h5ad",
                        "--skip-spatial-microns",
                        "--h5ad",
                        str(h5ad),
                        "--process-output-dir",
                        str(proc_py),
                    ],
                    env=env_base,
                    timeout=7200.0,
                )
                wall_py_cli = time.perf_counter() - t1
                result.preprocess.append(
                    {
                        "n_cells": n_cells,
                        "pipeline": "spacetravlr_process_h5ad_cli_uv_scanpy_magic",
                        "returncode": rc_p,
                        "wall_s": wall_py_cli,
                        **({"stderr_tail": err_p[-4000:]} if rc_p != 0 else {}),
                    }
                )

        payload = {
            "meta": result.meta,
            "training": result.training,
            "preprocess": result.preprocess,
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"wrote": str(out_json)}, indent=2))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
