#!/usr/bin/env python3
"""Sweep [perturbation].beta_scale_factor and pick settings that maximize concordance.

Uses the existing trained run (default: baseline_pooled_seed). Ligand splash
during in-silico KO scales with beta_scale_factor; immune/ligand-heavy KOs
(Il4ra, Cd83, Cd74) often improve at non-default values.

Writes:
  results/beta_sweep/sweep_summary.csv
  results/beta_sweep/best_scales.json
  results/predictions_tuned/   (predictions at best per-gene or global scale)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_spec = importlib.util.spec_from_file_location("ms08", HERE / "08_multislice_validation.py")
_ms08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ms08)
validate_slice = _ms08.validate_slice

DEFAULT_GENES = ["Il4ra", "Cd83", "Cd74", "Cks1b", "Bcam", "Ptk6"]
DEFAULT_SCALES = [40.0, 75.0, 100.0, 125.0, 175.0]
DEFAULT_SLICES = ["subQ-1", "subQ-2", "subQ-3", "subQ-4"]


def run_perturb(run_toml: Path, gene: str, out: Path, beta_scale: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "spacetravlr-perturb",
        "--run-toml", str(run_toml),
        "--gene", gene,
        "--desired-expr", "0.0",
        "--n-propagation", "4",
        "--beta-scale-factor", str(beta_scale),
        "--out", str(out),
    ]
    subprocess.run(cmd, check=True)


def score_predictions(
    pred_dir: Path,
    baseline_h5ad: Path,
    data_root: Path,
    slices: list[str],
    genes: list[str],
) -> pd.DataFrame:
    import scanpy as sc
    if baseline_h5ad.is_dir():
        baseline_h5ad = sorted(baseline_h5ad.glob("*.h5ad"))[0]
    baseline = sc.read_h5ad(baseline_h5ad)
    if "imputed_count" in baseline.layers:
        baseline.X = baseline.layers["imputed_count"]
    common_genes = sorted(baseline.var_names.astype(str))
    rows = []
    for sl in slices:
        df, _ = validate_slice(sl, data_root, baseline, pred_dir, genes, common_genes, n_perm=500)
        if not df.empty:
            df["slice"] = sl
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-toml", type=Path,
                    default=ROOT / "runs/baseline_pooled_seed/spacetravlr_run_repro.toml")
    ap.add_argument("--baseline-h5ad", type=Path,
                    default=ROOT / "runs/baseline_pooled_seed/spacetravlr_prep")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--slices", nargs="+", default=DEFAULT_SLICES)
    ap.add_argument("--genes", nargs="+", default=DEFAULT_GENES)
    ap.add_argument("--scales", nargs="+", type=float, default=DEFAULT_SCALES)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/beta_sweep")
    ap.add_argument("--pred-root", type=Path, default=ROOT / "results/beta_sweep/predictions")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "figures/beta_sweep")
    ap.add_argument("--write-tuned", action="store_true",
                    help="Write best-scale predictions to results/predictions_tuned/")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for scale in args.scales:
        tag = f"beta{int(scale)}"
        pred_dir = args.pred_root / tag
        pred_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== beta_scale_factor = {scale} ===")
        for gene in args.genes:
            out = pred_dir / f"predicted_KO_{gene}.feather"
            if not out.exists():
                print(f"  perturb {gene}…")
                run_perturb(args.run_toml, gene, out, scale)
        scores = score_predictions(pred_dir, args.baseline_h5ad, args.data_root, args.slices, args.genes)
        if scores.empty:
            continue
        for (pert, ct), grp in scores.groupby(["perturbation", "cell_type"]):
            summary_rows.append({
                "beta_scale_factor": scale,
                "perturbation": pert,
                "cell_type": ct,
                "median_pearson_r": float(grp.pearson_r.median()),
                "mean_pearson_r": float(grp.pearson_r.mean()),
                "n_slices": int(grp.shape[0]),
            })
        med = scores.groupby("cell_type").pearson_r.median()
        summary_rows.append({
            "beta_scale_factor": scale,
            "perturbation": "_ALL_",
            "cell_type": "immune",
            "median_pearson_r": float(med.get("immune", np.nan)),
            "mean_pearson_r": float(scores[scores.cell_type == "immune"].pearson_r.mean()) if (scores.cell_type == "immune").any() else np.nan,
            "n_slices": 0,
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "sweep_summary.csv", index=False)

    best = {}
    sub = summary[summary.perturbation != "_ALL_"]
    if not sub.empty:
        for pert in args.genes:
            psub = sub[sub.perturbation == pert]
            if psub.empty:
                continue
            imm = psub[psub.cell_type.isin(["immune", "myeloid"])]
            target = imm if not imm.empty else psub
            by_scale = target.groupby("beta_scale_factor").median_pearson_r.mean()
            best[pert] = float(by_scale.idxmax())

        global_scores = sub.groupby("beta_scale_factor").median_pearson_r.mean()
        best["_global"] = float(global_scores.idxmax())

    (args.out_dir / "best_scales.json").write_text(json.dumps(best, indent=2))

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        show = sub[sub.cell_type.isin(["immune", "myeloid", "fibroblast"])]
        for ct in show.cell_type.unique():
            pivot = show[show.cell_type == ct].pivot_table(
                index="perturbation", columns="beta_scale_factor", values="median_pearson_r"
            )
            for pert in pivot.index:
                ax.plot(pivot.columns, pivot.loc[pert], marker="o", label=f"{pert}|{ct}" if ct == "immune" else None, alpha=0.7)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("beta_scale_factor")
        ax.set_ylabel("Median Pearson r (4 slices)")
        ax.set_title("Concordance vs beta_scale_factor (pooled seed model)")
        ax.legend(fontsize=6, ncol=2, loc="best")
        fig.tight_layout()
        fig.savefig(args.fig_dir / "beta_scale_sweep.png", dpi=180)
        plt.close(fig)

    if args.write_tuned:
        tuned_dir = ROOT / "results/predictions_tuned"
        tuned_dir.mkdir(parents=True, exist_ok=True)
        for gene in args.genes:
            scale = best.get(gene, best.get("_global", 100.0))
            out = tuned_dir / f"predicted_KO_{gene}.feather"
            run_perturb(args.run_toml, gene, out, scale)
        meta = {"best_scales": best, "run_toml": str(args.run_toml)}
        (tuned_dir / "tuning_meta.json").write_text(json.dumps(meta, indent=2))

    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
