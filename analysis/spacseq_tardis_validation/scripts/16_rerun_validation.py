#!/usr/bin/env python3
"""Re-run full validation stack for a model tag from config/validation_runs.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pooled_extra", choices=["pooled_seed", "pooled_extra", "pooled_tuned"])
    ap.add_argument("--tag", default=None, help="Output tag (default: model key)")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-perturb", action="store_true")
    ap.add_argument("--skip-analysis", action="store_true")
    args = ap.parse_args()

    cfg = json.loads((ROOT / "config/validation_runs.json").read_text())["models"][args.model]
    tag = args.tag or args.model.replace("pooled_", "")

    if not args.skip_train and args.model == "pooled_extra":
        genes = (ROOT / "data/target_genes.txt").read_text().strip().replace("\n", ",")
        run([
            "spacetravlr", "--plain", "--training-mode", "seed",
            "--config", "spaceship_config_pooled_extra.toml",
            "--h5ad", "data/pooled/baseline_ntc.h5ad",
            "--output-dir", cfg["run_dir"],
            "--max-ligands", "200",
            "--genes", genes,
            "--parallel", "8",
        ])

    run_toml = ROOT / cfg["run_toml"]
    pred_dir = ROOT / cfg["pred_dir"]
    pred_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_perturb:
        beta = cfg.get("beta_scale_factor", 100.0)
        per_gene_beta = None
        if isinstance(beta, str):
            meta = ROOT / "results/beta_sweep/best_scales.json"
            if meta.exists():
                per_gene_beta = json.loads(meta.read_text())
            beta = 100.0
        for gene in ["Bcam", "Cks1b", "Ptk6", "Cd83", "Il4ra", "Cd74"]:
            out = pred_dir / f"predicted_KO_{gene}.feather"
            b = float(per_gene_beta.get(gene, per_gene_beta.get("_global", beta))) if per_gene_beta else float(beta)
            cmd = [
                "spacetravlr-perturb",
                "--run-toml", str(run_toml),
                "--gene", gene,
                "--desired-expr", "0.0",
                "--n-propagation", "4",
                "--beta-scale-factor", str(b),
                "--out", str(out),
            ]
            run(cmd)

    if args.skip_analysis:
        return

    base = ROOT / cfg["baseline_h5ad"]
    if not Path(str(base)).exists():
        prep = ROOT / cfg["run_dir"] / "spacetravlr_prep"
        h5 = sorted(prep.glob("*.h5ad"))
        if not h5:
            sys.exit(f"No prep h5ad under {prep}")
        base = h5[0]

    run([
        sys.executable, "scripts/08_multislice_validation.py",
        "--baseline-h5ad", str(base),
        "--pred-dir", str(pred_dir),
        "--out-dir", f"results/multislice",
        "--fig-dir", "figures/multislice",
        "--tag", tag,
    ])
    run([
        sys.executable, "scripts/11_beta_leiden_microniches.py",
        "--baseline-h5ad", str(base),
        "--betadata-dir", str(ROOT / cfg["betadata_dir"]),
        "--pred-dir", str(pred_dir),
        "--tag", tag,
    ])
    run([
        sys.executable, "scripts/12_beta_leiden_report_figures.py",
        "--baseline-h5ad", str(base),
        "--betadata-dir", str(ROOT / cfg["betadata_dir"]),
        "--pred-dir", str(pred_dir),
        "--tag", tag,
    ])
    if (HERE / "13_niche_deg_ccc_analysis.py").exists():
        run([sys.executable, "scripts/13_niche_deg_ccc_analysis.py", "--tag", tag])


if __name__ == "__main__":
    main()
