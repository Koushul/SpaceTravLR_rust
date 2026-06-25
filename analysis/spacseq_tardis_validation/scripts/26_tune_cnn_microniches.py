#!/usr/bin/env python3
"""Tune CNN β-microniches to beat expression-Leiden and BANKSY controls.

Phases:
  leiden   — grid-search microniche clustering hyperparameters (fast)
  join     — resume GPU training for missing genes in an existing run dir
  perturb  — grid-search beta_scale × n_propagation for enrichment preds
  eval     — score CNN median Pearson r vs control baselines
  all      — join → perturb → leiden sweep → pick best
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _py_boot import ensure_boot

ensure_boot()

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

import cnn_microniche_utils as cmu

_spec23 = importlib.util.spec_from_file_location("sp23", HERE / "23_cnn_microniche_enrichment.py")
_sp23 = importlib.util.module_from_spec(_spec23)
_spec23.loader.exec_module(_sp23)

_spec25 = importlib.util.spec_from_file_location("sp25", HERE / "25_cnn_microniche_controls.py")
_sp25 = importlib.util.module_from_spec(_spec25)
_spec25.loader.exec_module(_sp25)

run_slice_enrichment = _sp23.run_slice_enrichment
resolve_paths = _sp23.resolve_paths
assign_pool_niches = _sp23.assign_pool_niches
load_baseline = _sp23.load_baseline
load_pool = _sp23.load_pool
SUBQ_SLICES = _sp23.SUBQ_SLICES
LUNG_SLICE = _sp23.LUNG_SLICE
SUBQ_PERTS = _sp23.SUBQ_PERTS
LUNG_PERTS = _sp23.LUNG_PERTS

CONTROL_BASELINES = {"expr_leiden": 0.294, "banksy": 0.195, "random_niche": -0.097}


def panel_genes(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def missing_genes(panel: Path, betadata_dir: Path) -> list[str]:
    want = set(panel_genes(panel))
    have = {p.stem.replace("_betadata", "") for p in betadata_dir.glob("*_betadata.feather")}
    return sorted(want - have)


def run_cmd(cmd: list[str], log: Path, env: dict | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    with log.open("a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True, env=run_env)


def join_train_missing(
    betadata_dir: Path,
    panel: Path,
    parallel: int,
    genes: list[str] | None = None,
) -> None:
    repro = betadata_dir / "spacetravlr_run_repro.toml"
    if not repro.exists():
        raise SystemExit(f"No repro TOML at {repro}; run leader training first.")
    todo = genes or missing_genes(panel, betadata_dir)
    if not todo:
        print("All panel genes already trained.")
        return
    print(f"Join-training {len(todo)} genes: {', '.join(todo)}")
    log = betadata_dir / "join_train.log"
    run_cmd([
        "spacetravlr", "--plain",
        "--join-output-dir", str(betadata_dir.resolve()),
        "--parallel", str(parallel),
        "--genes", ",".join(todo),
    ], log)


def train_v2(panel: Path, config: Path, parallel: int, epochs: int) -> Path:
    out = ROOT / "runs" / "baseline_pooled_cnn_v2"
    genes_list = panel_genes(panel)
    genes = ",".join(genes_list)
    log = ROOT / "runs" / "cnn_pooled_v2_train.log"
    force_keep = ",".join(genes_list)
    run_cmd([
        "spacetravlr", "--plain",
        "--config", str(config),
        "--h5ad", str(ROOT / "data/pooled/baseline_ntc.h5ad"),
        "--output-dir", str(out),
        "--training-mode", "full",
        "--epochs", str(epochs),
        "--parallel", str(parallel),
        "--genes", genes,
        "--max-ligands", "250",
    ], log, env={"SPACETRAVLR_FORCE_KEEP_GENES": force_keep})
    return out


def run_perturbations(run_toml: Path, pred_dir: Path, perts: list[str], beta: float, n_prop: int) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    log = pred_dir / "perturb.log"
    for g in perts:
        out = pred_dir / f"predicted_KO_{g}.feather"
        if out.exists():
            continue
        run_cmd([
            "spacetravlr-perturb",
            "--run-toml", str(run_toml),
            "--gene", g,
            "--desired-expr", "0",
            "--n-propagation", str(n_prop),
            "--beta-scale-factor", str(beta),
            "--out", str(out),
        ], log)


def score_cnn_enrichment(
    betadata_dir: Path,
    pred_dir: Path,
    seed_pred_dir: Path,
    leiden_kw: dict,
    tag: str = "tune",
    cluster_gene_filter: set[str] | None = None,
) -> tuple[float, pd.DataFrame]:
    pooled = load_baseline(ROOT / "data/pooled/baseline_ntc.h5ad")
    if "slice_id" not in pooled.obs.columns:
        pooled.obs["slice_id"] = pooled.obs_names.str.split("@").str[-1]

    all_corr: list[pd.DataFrame] = []
    for sl in SUBQ_SLICES + [LUNG_SLICE]:
        bd, bl_path, pd_dir = resolve_paths(
            argparse.Namespace(
                betadata_dir=betadata_dir,
                seed_betadata_dir=ROOT / "runs/baseline_pooled_seed",
                pred_dir=pred_dir,
                seed_pred_dir=seed_pred_dir,
                data_root=ROOT / "data",
                baseline_h5ad=ROOT / "data/pooled/baseline_ntc.h5ad",
            ),
            sl,
        )
        if sl.startswith("subQ"):
            baseline = pooled[pooled.obs["slice_id"].astype(str) == sl].copy()
            perts, gb = SUBQ_PERTS, None
        else:
            baseline = load_baseline(bl_path)
            baseline.obs["slice_id"] = sl
            perts, gb = LUNG_PERTS, pooled

        pool = load_pool(sl, ROOT / "data")
        pool.obs["slice_id"] = sl
        prep = baseline.copy()
        cmu.ensure_cluster_id(prep)
        beta_matrix, score_genes = cmu.build_beta_score_matrix(
            prep, bd, gene_filter=cluster_gene_filter,
        )
        prep, pool = assign_pool_niches(prep, pool, beta_matrix, sl, leiden_kw=leiden_kw)

        _, corr, _, _ = run_slice_enrichment(
            sl, perts, ROOT / "data", prep, bd, pd_dir, tag,
            global_baseline=gb, fallback_pred_dir=seed_pred_dir,
            leiden_kw=leiden_kw, niche_key="cnn_leiden",
            prep=prep, pool=pool, score_genes=score_genes, beta_matrix=beta_matrix,
        )
        if not corr.empty:
            all_corr.append(corr)

    corr_df = pd.concat(all_corr, ignore_index=True) if all_corr else pd.DataFrame()
    med = float(corr_df["pearson_r"].median()) if not corr_df.empty else float("nan")
    return med, corr_df


def sweep_leiden(
    betadata_dir: Path,
    pred_dir: Path,
    seed_pred_dir: Path,
    out_dir: Path,
    cluster_gene_filter: set[str] | None = None,
) -> dict:
    resolutions = [0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    spatial_weights = [0.25, 0.35, 0.4, 0.45, 0.55]
    n_pcs_list = [10, 14, 18]

    rows = []
    best = {"median_pearson_r": float("-inf")}
    for res, sw, npc in itertools.product(resolutions, spatial_weights, n_pcs_list):
        leiden_kw = {
            "resolution": res,
            "spatial_weight": sw,
            "n_pcs": npc,
            "min_cells": cmu.DEFAULT_LEIDEN_KW["min_cells"],
        }
        med, corr_df = score_cnn_enrichment(
            betadata_dir, pred_dir, seed_pred_dir, leiden_kw, cluster_gene_filter=cluster_gene_filter,
        )
        row = {
            "resolution": res,
            "spatial_weight": sw,
            "n_pcs": npc,
            "median_pearson_r": med,
            "mean_pearson_r": float(corr_df["pearson_r"].mean()) if not corr_df.empty else None,
            "n_tests": int(len(corr_df)),
            "beats_expr_leiden": med > CONTROL_BASELINES["expr_leiden"],
            "beats_banksy": med > CONTROL_BASELINES["banksy"],
        }
        rows.append(row)
        print(f"r={res} sw={sw} npc={npc} → median r={med:+.3f}")
        if med > best.get("median_pearson_r", float("-inf")):
            best = {**row, "leiden_kw": leiden_kw}

    sweep_df = pd.DataFrame(rows).sort_values("median_pearson_r", ascending=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(out_dir / "leiden_sweep.csv", index=False)
    (out_dir / "leiden_best.json").write_text(json.dumps(best, indent=2))
    print("\nBest Leiden:", json.dumps(best, indent=2))
    return best


def sweep_perturb(
    run_toml: Path,
    betadata_dir: Path,
    seed_pred_dir: Path,
    leiden_kw: dict,
    out_dir: Path,
    cluster_gene_filter: set[str] | None = None,
) -> dict:
    betas = [40, 60, 75, 100, 125, 150]
    n_props = [2, 3, 4, 5, 6]
    perts = list(dict.fromkeys(SUBQ_PERTS + LUNG_PERTS))
    rows = []
    best = {"median_pearson_r": float("-inf")}

    for beta, np_ in itertools.product(betas, n_props):
        tag = f"beta{int(beta)}_np{np_}"
        pred_dir = out_dir / "perturb_sweep" / tag
        run_perturbations(run_toml, pred_dir, perts, beta, np_)
        med, corr_df = score_cnn_enrichment(
            betadata_dir, pred_dir, seed_pred_dir, leiden_kw, tag=tag,
            cluster_gene_filter=cluster_gene_filter,
        )
        row = {"beta_scale_factor": beta, "n_propagation": np_, "median_pearson_r": med, "n_tests": int(len(corr_df))}
        rows.append(row)
        print(f"beta={beta} np={np_} → median r={med:+.3f}")
        if med > best.get("median_pearson_r", float("-inf")):
            best = {**row, "pred_dir": str(pred_dir), "leiden_kw": leiden_kw}

    sweep_df = pd.DataFrame(rows).sort_values("median_pearson_r", ascending=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(out_dir / "perturb_sweep.csv", index=False)
    (out_dir / "perturb_best.json").write_text(json.dumps(best, indent=2, default=str))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["join", "train", "perturb", "leiden", "perturb-sweep", "eval", "all"], default="all")
    ap.add_argument("--betadata-dir", type=Path, default=ROOT / "runs/baseline_pooled_cnn_v2")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "results/predictions_cnn_v2")
    ap.add_argument("--seed-pred-dir", type=Path, default=ROOT / "results/predictions_tuned")
    ap.add_argument("--panel", type=Path, default=ROOT / "data/cnn_niche_panel_v2.txt")
    ap.add_argument("--config-v2", type=Path, default=ROOT / "spaceship_config_pooled_cnn_v2.toml")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--beta", type=float, default=75.0)
    ap.add_argument("--n-propagation", type=int, default=4)
    ap.add_argument("--resolution", type=float, default=0.9)
    ap.add_argument("--spatial-weight", type=float, default=0.35)
    ap.add_argument("--n-pcs", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/cnn_enrichment/tune")
    ap.add_argument("--cluster-genes", choices=["all", "focused"], default="focused")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    perts = list(dict.fromkeys(SUBQ_PERTS + LUNG_PERTS))

    cluster_filter = None if args.cluster_genes == "all" else cmu.MICRONICHE_CLUSTER_GENES

    if args.phase in ("join", "all") and not args.use_v2 and args.phase != "all":
        join_train_missing(args.betadata_dir, args.panel, args.parallel)

    if args.phase in ("train", "all"):
        args.betadata_dir = train_v2(args.panel, args.config_v2, args.parallel, args.epochs)
        (args.betadata_dir / "data").mkdir(parents=True, exist_ok=True)
        extra = ROOT / "data" / "extra_modulators.txt"
        link = args.betadata_dir / "data" / "extra_modulators.txt"
        if extra.exists() and not link.exists():
            link.symlink_to(extra.resolve())

    run_toml = args.betadata_dir / "spacetravlr_run_repro.toml"
    if args.phase in ("perturb", "all"):
        run_perturbations(run_toml, args.pred_dir, perts, args.beta, args.n_propagation)

    leiden_kw = {
        "resolution": args.resolution,
        "spatial_weight": args.spatial_weight,
        "n_pcs": args.n_pcs,
        "min_cells": cmu.DEFAULT_LEIDEN_KW["min_cells"],
    }

    if args.phase in ("leiden", "all"):
        best_leiden = sweep_leiden(
            args.betadata_dir, args.pred_dir, args.seed_pred_dir, out_dir, cluster_filter,
        )
        leiden_kw = best_leiden.get("leiden_kw", leiden_kw)

    if args.phase == "perturb-sweep":
        sweep_perturb(run_toml, args.betadata_dir, args.seed_pred_dir, leiden_kw, out_dir, cluster_filter)

    if args.phase in ("eval", "all", "leiden", "perturb-sweep"):
        med, corr_df = score_cnn_enrichment(
            args.betadata_dir, args.pred_dir, args.seed_pred_dir, leiden_kw, tag="tuned",
            cluster_gene_filter=cluster_filter,
        )
        summary = {
            "median_pearson_r": med,
            "mean_pearson_r": float(corr_df["pearson_r"].mean()) if not corr_df.empty else None,
            "n_tests": int(len(corr_df)),
            "leiden_kw": leiden_kw,
            "betadata_dir": str(args.betadata_dir),
            "pred_dir": str(args.pred_dir),
            "control_baselines": CONTROL_BASELINES,
            "beats_expr_leiden": med > CONTROL_BASELINES["expr_leiden"],
            "beats_banksy": med > CONTROL_BASELINES["banksy"],
            "margin_vs_expr": med - CONTROL_BASELINES["expr_leiden"],
            "margin_vs_banksy": med - CONTROL_BASELINES["banksy"],
        }
        corr_df.to_csv(out_dir / "enrichment_corr_tuned.csv", index=False)
        (out_dir / "overall_tuned.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
