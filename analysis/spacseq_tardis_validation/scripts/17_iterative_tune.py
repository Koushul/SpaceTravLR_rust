#!/usr/bin/env python3
"""Iterative hyperparameter search for in-silico KO concordance.

Grid-searches beta_scale_factor × n_propagation on a trained run, scores
4-slice immune/myeloid concordance, writes tuned predictions, re-runs validation,
and emits a before/after dashboard.

Uses the Rust-bundled Python when available (scanpy + importlib_metadata).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _py_boot import PY, ensure_boot

ensure_boot()

import argparse
import importlib.util
import json
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = Path(os.environ.get("SPACETRAVLR_PYTHON", "/software/rhel9/manual/install/rust/1.89.0/python3.11/bin/python3"))

_spec08 = importlib.util.spec_from_file_location("ms08", HERE / "08_multislice_validation.py")
_ms08 = importlib.util.module_from_spec(_spec08)
_spec08.loader.exec_module(_ms08)
validate_slice = _ms08.validate_slice

VAL_GENES = ["Il4ra", "Cd83", "Cd74", "Bcam", "Cks1b", "Ptk6"]
SLICES = ["subQ-1", "subQ-2", "subQ-3", "subQ-4"]
IMMUNE_CT = {"immune", "myeloid", "fibroblast"}


def run_perturb(run_toml: Path, gene: str, out: Path, beta: float, n_prop: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "spacetravlr-perturb",
        "--run-toml", str(run_toml),
        "--gene", gene,
        "--desired-expr", "0.0",
        "--n-propagation", str(n_prop),
        "--beta-scale-factor", str(beta),
        "--out", str(out),
    ]
    subprocess.run(cmd, check=True)


def score_run(
    pred_dir: Path,
    baseline_h5ad: Path,
    data_root: Path,
    genes: list[str],
) -> dict:
    if baseline_h5ad.is_dir():
        baseline_h5ad = sorted(baseline_h5ad.glob("*.h5ad"))[0]
    baseline = sc.read_h5ad(baseline_h5ad)
    if "imputed_count" in baseline.layers:
        baseline.X = baseline.layers["imputed_count"]
    common = sorted(baseline.var_names.astype(str))
    rows = []
    for sl in SLICES:
        df, _ = validate_slice(sl, data_root, baseline, pred_dir, genes, common, n_perm=300)
        if not df.empty:
            rows.append(df)
    if not rows:
        return {"median_r_immune": np.nan, "median_r_myeloid": np.nan, "median_r_fibroblast": np.nan,
                "median_r_all": np.nan, "n_rows": 0}
    all_df = pd.concat(rows, ignore_index=True)
    out = {"n_rows": len(all_df)}
    for ct in ["immune", "myeloid", "fibroblast", "tumor"]:
        sub = all_df[all_df.cell_type == ct]
        out[f"median_r_{ct}"] = float(sub.pearson_r.median()) if len(sub) else float("nan")
    focus = all_df[all_df.cell_type.isin(IMMUNE_CT)]
    out["median_r_focus"] = float(focus.pearson_r.median()) if len(focus) else float("nan")
    out["median_r_all"] = float(all_df.pearson_r.median())
    return out


def grid_search(
    run_toml: Path,
    baseline_h5ad: Path,
    data_root: Path,
    betas: list[float],
    n_props: list[int],
    genes: list[str],
    sweep_root: Path,
    baseline_pred_dir: Path | None,
) -> pd.DataFrame:
    rows = []
    if baseline_pred_dir and baseline_pred_dir.exists():
        base = score_run(baseline_pred_dir, baseline_h5ad, data_root, genes)
        rows.append({
            "beta_scale_factor": "baseline",
            "n_propagation": "baseline",
            "pred_dir": str(baseline_pred_dir),
            **base,
        })
        print(f"baseline: focus r={base.get('median_r_focus', np.nan):+.3f}")

    for beta in betas:
        for n_prop in n_props:
            tag = f"beta{int(beta)}_np{n_prop}"
            pred_dir = sweep_root / tag
            print(f"\n=== {tag} ===")
            for gene in genes:
                out = pred_dir / f"predicted_KO_{gene}.feather"
                if not out.exists():
                    run_perturb(run_toml, gene, out, beta, n_prop)
            metrics = score_run(pred_dir, baseline_h5ad, data_root, genes)
            rows.append({
                "beta_scale_factor": beta,
                "n_propagation": n_prop,
                "pred_dir": str(pred_dir),
                **metrics,
            })
            print(f"  focus r={metrics.get('median_r_focus', np.nan):+.3f}  "
                  f"immune={metrics.get('median_r_immune', np.nan):+.3f}")
    return pd.DataFrame(rows)


def pick_best(sweep: pd.DataFrame) -> dict:
    sub = sweep[sweep.beta_scale_factor != "baseline"].copy()
    if sub.empty:
        return {"beta_scale_factor": 100.0, "n_propagation": 4}
    sub = sub.assign(
        beta_scale_factor=sub.beta_scale_factor.astype(float),
        n_propagation=sub.n_propagation.astype(int),
    )
    idx = sub.median_r_focus.idxmax()
    row = sub.loc[idx]
    return {
        "beta_scale_factor": float(row.beta_scale_factor),
        "n_propagation": int(row.n_propagation),
        "pred_dir": row.pred_dir,
        "median_r_focus": float(row.median_r_focus),
    }


def copy_tuned(best: dict, out_dir: Path, genes: list[str]) -> None:
    import shutil
    src = Path(best["pred_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for gene in genes:
        shutil.copy2(src / f"predicted_KO_{gene}.feather", out_dir / f"predicted_KO_{gene}.feather")
    meta = {k: v for k, v in best.items() if k != "pred_dir"}
    meta["source_pred_dir"] = best["pred_dir"]
    (out_dir / "tuning_meta.json").write_text(json.dumps(meta, indent=2))


def plot_dashboard(sweep: pd.DataFrame, fig_dir: Path, tag: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    sub = sweep[sweep.beta_scale_factor != "baseline"].copy()
    if sub.empty:
        return
    sub = sub.assign(
        beta_scale_factor=sub.beta_scale_factor.astype(float),
        n_propagation=sub.n_propagation.astype(int),
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    pivot = sub.pivot_table(index="n_propagation", columns="beta_scale_factor", values="median_r_focus")
    pivot.columns = pivot.columns.astype(float)
    pivot.index = pivot.index.astype(int)
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.02, vmax=max(0.2, float(np.nanmax(pivot.values))))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.0f}" for c in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{i}" for i in pivot.index])
    ax.set_xlabel("beta_scale_factor")
    ax.set_ylabel("n_propagation")
    ax.set_title("Focus median r\n(immune+myeloid+fibroblast)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    base = sweep[sweep.beta_scale_factor == "baseline"]
    best = sub.loc[sub.median_r_focus.idxmax()]
    comps = ["immune", "myeloid", "fibroblast"]
    x = np.arange(len(comps))
    w = 0.35
    if not base.empty:
        b = base.iloc[0]
        ax.bar(x - w / 2, [b.get(f"median_r_{c}", np.nan) for c in comps], width=w, label="baseline", color="#bbb")
    ax.bar(x + w / 2, [best.get(f"median_r_{c}", np.nan) for c in comps], width=w,
           label=f"tuned β={best.beta_scale_factor:.0f} np={best.n_propagation}", color="#2166ac")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(comps)
    ax.set_ylabel("Median Pearson r")
    ax.legend(fontsize=7)
    ax.set_title("Compartment concordance")

    ax = axes[2]
    if "median_r_all" in sub.columns:
        ax.scatter(sub.beta_scale_factor.astype(float), sub.median_r_focus, c=sub.n_propagation.astype(float),
                   cmap="viridis", s=60, edgecolors="k", lw=0.3)
        ax.set_xlabel("beta_scale_factor")
        ax.set_ylabel("Focus median r")
        ax.set_title("Sweep scatter")
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax)
        cb.set_label("n_propagation")

    fig.suptitle(f"Hyperparameter sweep — {tag}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig_iteration_dashboard_{tag}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def rerun_validation(baseline_h5ad: Path, betadata_dir: Path, pred_dir: Path, tag: str) -> None:
    cmds = [
        [str(PY), str(HERE / "08_multislice_validation.py"),
         "--baseline-h5ad", str(baseline_h5ad), "--pred-dir", str(pred_dir),
         "--out-dir", str(ROOT / "results/multislice"), "--fig-dir", str(ROOT / "figures/multislice"),
         "--tag", tag],
        [str(PY), str(HERE / "11_beta_leiden_microniches.py"),
         "--baseline-h5ad", str(baseline_h5ad), "--betadata-dir", str(betadata_dir),
         "--pred-dir", str(pred_dir), "--tag", tag],
        [str(PY), str(HERE / "12_beta_leiden_report_figures.py"),
         "--baseline-h5ad", str(baseline_h5ad), "--betadata-dir", str(betadata_dir),
         "--pred-dir", str(pred_dir), "--tag", tag],
        [str(PY), str(HERE / "10_sharpened_scorecard.py"), "--models", "pooled", tag],
    ]
    for cmd in cmds:
        print("+", " ".join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-toml", type=Path,
                    default=ROOT / "runs/baseline_pooled_seed/spacetravlr_run_repro.toml")
    ap.add_argument("--baseline-h5ad", type=Path,
                    default=ROOT / "runs/baseline_pooled_seed/spacetravlr_prep/baseline_ntc_0c6fbac5e6cd947c_fullprep.h5ad")
    ap.add_argument("--betadata-dir", type=Path, default=ROOT / "runs/baseline_pooled_seed")
    ap.add_argument("--baseline-pred-dir", type=Path, default=ROOT / "results/predictions_pooled")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--betas", nargs="+", type=float, default=[40, 75, 100, 125, 150, 200])
    ap.add_argument("--n-props", nargs="+", type=int, default=[3, 4, 5, 6])
    ap.add_argument("--sweep-root", type=Path, default=ROOT / "results/iteration_sweep")
    ap.add_argument("--tuned-dir", type=Path, default=ROOT / "results/predictions_tuned")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/iteration")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures/iteration")
    ap.add_argument("--tag", default="tuned")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    sweep_path = args.out_dir / "sweep_results.csv"
    if args.skip_sweep and sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
    else:
        sweep = grid_search(
            args.run_toml, args.baseline_h5ad, args.data_root,
            args.betas, args.n_props, VAL_GENES,
            args.sweep_root, args.baseline_pred_dir,
        )
        sweep.to_csv(sweep_path, index=False)

    best = pick_best(sweep)
    (args.out_dir / "best_params.json").write_text(json.dumps(best, indent=2))
    print("\nBEST:", json.dumps(best, indent=2))

    copy_tuned(best, args.tuned_dir, VAL_GENES)
    plot_dashboard(sweep, args.fig_dir, args.tag)

    if not args.skip_validation:
        rerun_validation(args.baseline_h5ad, args.betadata_dir, args.tuned_dir, args.tag)


if __name__ == "__main__":
    main()
