#!/usr/bin/env python3
"""Run validation analyses once and write a cache manifest for notebooks.

Notebooks load CSV/JSON from the manifest and render plots via nb_viz.py instead
of re-running heavy scripts or displaying pre-baked PNGs only.

Usage:
  python3 scripts/cache_validation_results.py                    # full stack (tuned)
  python3 scripts/cache_validation_results.py --manifest-only    # index existing results
  python3 scripts/cache_validation_results.py --sections multislice,cnn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _py_boot import PY, ensure_boot
from nb_cache import DEFAULT_SECTIONS, build_manifest

ensure_boot()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(cmd: list[str]) -> None:
    import subprocess

    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def resolve_baseline(cfg: dict) -> Path:
    base = ROOT / cfg["baseline_h5ad"]
    if base.exists():
        return base
    prep = ROOT / cfg["run_dir"] / "spacetravlr_prep"
    h5 = sorted(prep.glob("*.h5ad"))
    if not h5:
        raise FileNotFoundError(f"No baseline h5ad under {prep}")
    return h5[0]


def run_section(
    section: str,
    tag: str,
    spatial_tag: str,
    cnn_tag: str,
    cfg: dict,
    baseline: Path,
    skip_beta_leiden: bool,
) -> None:
    pred = ROOT / cfg["pred_dir"]
    betadata = ROOT / cfg["betadata_dir"]

    if section == "multislice":
        run([
            str(PY), "scripts/08_multislice_validation.py",
            "--baseline-h5ad", str(baseline),
            "--pred-dir", str(pred),
            "--out-dir", "results/multislice",
            "--fig-dir", "figures/multislice",
            "--tag", tag,
        ])
    elif section == "spatial":
        run([
            str(PY), "scripts/09_spatial_validation.py",
            "--baseline-h5ad", str(baseline),
            "--pred-dir", str(pred),
            "--tag", tag,
        ])
    elif section == "scorecard":
        run([str(PY), "scripts/10_sharpened_scorecard.py", "--models", "pooled", tag])
    elif section == "beta_leiden":
        run([
            str(PY), "scripts/11_beta_leiden_microniches.py",
            "--baseline-h5ad", str(baseline),
            "--betadata-dir", str(betadata),
            "--pred-dir", str(pred),
            "--tag", tag,
        ])
        run([
            str(PY), "scripts/12_beta_leiden_report_figures.py",
            "--baseline-h5ad", str(baseline),
            "--betadata-dir", str(betadata),
            "--pred-dir", str(pred),
            "--tag", tag,
        ])
    elif section == "niche_deg":
        cmd = [
            str(PY), "scripts/13_niche_deg_ccc_analysis.py",
            "--baseline-h5ad", str(baseline),
            "--betadata-dir", str(betadata),
            "--pred-dir", str(pred),
            "--tag", spatial_tag,
        ]
        if skip_beta_leiden:
            cmd.append("--skip-beta-leiden")
        run(cmd)
    elif section == "niche_spp1":
        run([
            str(PY), "scripts/18_perturbation_niche_spp1.py",
            "--baseline-h5ad", str(baseline),
            "--pred-dir", str(pred),
            "--tag", tag,
            "--skip-spp1-perturb",
        ])
    elif section == "paper":
        run([
            str(PY), "scripts/19_paper_findings_validation.py",
            "--baseline-h5ad", str(baseline),
            "--pred-dir", str(pred),
            "--tag", tag,
        ])
    elif section == "extended_paper":
        run([
            str(PY), "scripts/20_extended_paper_validation.py",
            "--baseline-h5ad", str(baseline),
            "--pred-dir", str(pred),
            "--tag", tag,
        ])
    elif section == "dashboard":
        run([str(PY), "scripts/21_validation_dashboard.py", "--tag", spatial_tag])
    elif section == "cnn":
        cnn_betadata = ROOT / "runs/baseline_pooled_cnn"
        cnn_pred = ROOT / "results/predictions_cnn"
        if not cnn_betadata.exists():
            print(f"SKIP cnn: {cnn_betadata} not found")
            return
        run([
            str(PY), "scripts/23_cnn_microniche_enrichment.py",
            "--tag", cnn_tag,
            "--betadata-dir", str(cnn_betadata),
            "--pred-dir", str(cnn_pred),
            "--seed-pred-dir", str(pred),
            "--baseline-h5ad", str(ROOT / "data/pooled/baseline_ntc.h5ad"),
        ])
    else:
        raise ValueError(f"Unknown section: {section}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pooled_tuned", choices=["pooled_seed", "pooled_extra", "pooled_tuned"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--spatial-tag", default="spatial_v3")
    ap.add_argument("--cnn-tag", default="cnn")
    ap.add_argument(
        "--sections",
        default=",".join(DEFAULT_SECTIONS),
        help=f"Comma-separated sections: {','.join(DEFAULT_SECTIONS)}",
    )
    ap.add_argument("--manifest-only", action="store_true", help="Build manifest from existing artifacts only")
    ap.add_argument("--skip-beta-leiden", action="store_true", default=True)
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    args = ap.parse_args()

    cfg_path = ROOT / "config/validation_runs.json"
    cfg_all = json.loads(cfg_path.read_text())
    cfg = cfg_all["models"][args.model]
    if args.tag:
        tag = args.tag
    elif args.model == "pooled_tuned":
        tag = cfg_all.get("recommended_tag", "tuned")
    else:
        tag = args.model.replace("pooled_", "")

    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    cache_dir = args.cache_dir / tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.manifest_only:
        baseline = resolve_baseline(cfg)
        for sec in sections:
            print(f"\n=== section: {sec} ===", flush=True)
            run_section(sec, tag, args.spatial_tag, args.cnn_tag, cfg, baseline, args.skip_beta_leiden)

    manifest = build_manifest(tag, args.spatial_tag, args.cnn_tag, cfg, sections)
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {manifest_path}")
    print(f"Sections: {list(manifest['sections'].keys())}")
    if manifest["missing"]:
        print(f"Missing artifacts ({len(manifest['missing'])}):", *manifest["missing"][:8], sep="\n  ")
        if len(manifest["missing"]) > 8:
            print(f"  ... and {len(manifest['missing']) - 8} more")


if __name__ == "__main__":
    main()
