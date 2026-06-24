#!/usr/bin/env python3
"""Run CNN full-mode training + perturbations for microniche enrichment pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "cnn_niche_panel.txt"
PERTS_POOLED = ["Il4ra", "Cd83", "Cd74", "Bcam", "Cks1b", "Ptk6", "Icam1", "Cd44", "Spp1"]
PERTS_LUNG = ["Icam1", "Bcam"]


def genes_csv(path: Path) -> str:
    return ",".join(g.strip() for g in path.read_text().splitlines() if g.strip())


def run(cmd: list[str], log: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    with log.open("a") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def train_pooled(args) -> None:
    log = ROOT / "runs" / "cnn_pooled_train.log"
    genes = genes_csv(PANEL)
    cmd = [
        "spacetravlr", "--plain",
        "--config", str(ROOT / "spaceship_config_pooled_full.toml"),
        "--h5ad", str(ROOT / "data/pooled/baseline_ntc.h5ad"),
        "--output-dir", str(ROOT / "runs/baseline_pooled_cnn"),
        "--training-mode", "full",
        "--epochs", str(args.epochs),
        "--parallel", str(args.parallel),
        "--genes", genes,
        "--max-ligands", "200",
    ]
    run(cmd, log)


def train_lung(args) -> None:
    log = ROOT / "runs" / "cnn_lung_train.log"
    genes = genes_csv(PANEL)
    cmd = [
        "spacetravlr", "--plain",
        "--config", str(ROOT / "spaceship_config_lung_cnn.toml"),
        "--h5ad", str(ROOT / "data/slices/Lung_Metastasis_M001/baseline_ntc.h5ad"),
        "--output-dir", str(ROOT / "runs/lung_m001_cnn"),
        "--training-mode", "full",
        "--epochs", str(max(6, args.epochs - 2)),
        "--parallel", str(min(2, args.parallel)),
        "--genes", genes,
        "--max-ligands", "200",
    ]
    run(cmd, log)


def perturb(run_toml: Path, pred_dir: Path, perts: list[str], beta: float, np: int) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    log = pred_dir / "perturb.log"
    for g in perts:
        out = pred_dir / f"predicted_KO_{g}.feather"
        if out.exists():
            continue
        cmd = [
            "spacetravlr-perturb",
            "--run-toml", str(run_toml),
            "--gene", g,
            "--desired-expr", "0",
            "--n-propagation", str(np),
            "--beta-scale-factor", str(beta),
            "--out", str(out),
        ]
        run(cmd, log)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-lung", action="store_true")
    ap.add_argument("--skip-perturb", action="store_true")
    ap.add_argument("--skip-enrichment", action="store_true")
    args = ap.parse_args()

    if not args.skip_train:
        train_pooled(args)
        if not args.skip_lung:
            train_lung(args)

    if not args.skip_perturb:
        pooled_toml = ROOT / "runs/baseline_pooled_cnn/spacetravlr_run_repro.toml"
        if not pooled_toml.exists():
            pooled_toml = ROOT / "runs/baseline_pooled_seed/spacetravlr_run_repro.toml"
        perturb(pooled_toml, ROOT / "results/predictions_cnn", PERTS_POOLED, beta=50, np=3)

    if not args.skip_enrichment:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/23_cnn_microniche_enrichment.py"), "--tag", "cnn"],
            cwd=ROOT, check=True,
        )


if __name__ == "__main__":
    main()
