#!/usr/bin/env python3
"""Prepare all subQ Visium HD slices for SpaceTravLR validation.

For each slice (subQ-1 .. subQ-5):
  1. Download perturbation + segmentation + raw (skip 2 GB bin transcriptome)
  2. Annotate cells (00_annotate_slice.py)
  3. Assign sgRNAs to cells (01_assign_guides_to_cells.py)
  4. Build baseline_ntc.h5ad + perturbed_pool.h5ad (02_build_training_h5ad.py)

Outputs per slice under data/slices/<slice>/.
Also writes data/slices/slice_manifest.json summarising cell counts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent.parent
MC38 = REPO / "analysis" / "mc38_visiumhd"
SCRIPTS = HERE

DEFAULT_SLICES = ["subQ-1", "subQ-2", "subQ-3", "subQ-4", "subQ-5"]
PERTURB_GENES = ["Bcam", "Cks1b", "Ptk6", "Cd83", "Il4ra", "Cd74"]


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def slice_data_dir(slice_name: str, mc38_dir: Path) -> Path:
    return mc38_dir / slice_name


def slice_out_dir(slice_name: str, data_root: Path) -> Path:
    return data_root / "slices" / slice_name


def ensure_download(slice_name: str, mc38_dir: Path, skip_download: bool) -> Path:
    data_dir = slice_data_dir(slice_name, mc38_dir)
    need = data_dir / "perturbation" / "filtered_guide_bc_matrix.h5"
    if need.exists():
        print(f"[{slice_name}] data present at {data_dir}")
        return data_dir
    if skip_download:
        raise SystemExit(f"[{slice_name}] missing data and --skip-download set")
    data_dir.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable,
        str(MC38 / "download_spac_data.py"),
        "--name", slice_name,
        "--dataset-type", "2",
        "--out-dir", str(data_dir),
        "--components", "perturbation", "segmentation", "raw",
    ])
    return data_dir


def prepare_slice(
    slice_name: str,
    mc38_dir: Path,
    data_root: Path,
    skip_download: bool,
    perturb_genes: list[str],
) -> dict:
    t0 = time.time()
    data_dir = ensure_download(slice_name, mc38_dir, skip_download)
    out_dir = slice_out_dir(slice_name, data_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_path = data_dir / "processed" / f"{slice_name}_cells_annotated.h5ad"
    if not ann_path.exists():
        run([
            sys.executable, str(SCRIPTS / "00_annotate_slice.py"),
            "--data-dir", str(data_dir),
            "--slice-name", slice_name,
        ])
    else:
        print(f"[{slice_name}] annotated h5ad exists")

    guide_parquet = out_dir / "cell_guide_assignments.parquet"
    if not guide_parquet.exists():
        run([
            sys.executable, str(SCRIPTS / "01_assign_guides_to_cells.py"),
            "--data-dir", str(data_dir),
            "--out-dir", str(out_dir),
        ])
    else:
        print(f"[{slice_name}] guide assignments exist")

    baseline_path = out_dir / "baseline_ntc.h5ad"
    if not baseline_path.exists():
        run([
            sys.executable, str(SCRIPTS / "02_build_training_h5ad.py"),
            "--source-h5ad", str(ann_path),
            "--guide-parquet", str(guide_parquet),
            "--out-dir", str(out_dir),
            "--slice-id", slice_name,
            "--perturb-genes", *perturb_genes,
        ])
    else:
        print(f"[{slice_name}] h5ads exist")

    summary_path = out_dir / "guide_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    panel_path = out_dir / "gene_panel_summary.json"
    panel = json.loads(panel_path.read_text()) if panel_path.exists() else {}

    import scanpy as sc
    baseline = sc.read_h5ad(baseline_path)
    pert = sc.read_h5ad(out_dir / "perturbed_pool.h5ad")
    row = {
        "slice": slice_name,
        "n_cells_annotated": int(sc.read_h5ad(ann_path).n_obs) if ann_path.exists() else None,
        "n_cells_unambiguous": summary.get("n_cells_unambiguous"),
        "n_cells_ntc": summary.get("n_cells_ntc"),
        "n_baseline_ntc": int(baseline.n_obs),
        "n_perturbed_pool": int(pert.n_obs),
        "perturbation_cohorts": panel.get("perturbed_pool_cohorts", []),
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps(row, indent=2))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", nargs="+", default=DEFAULT_SLICES)
    ap.add_argument("--mc38-dir", type=Path, default=MC38)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--perturb-genes", nargs="+", default=PERTURB_GENES)
    args = ap.parse_args()

    rows = []
    for sl in args.slices:
        rows.append(prepare_slice(sl, args.mc38_dir, args.data_root, args.skip_download, args.perturb_genes))

    manifest = {"slices": rows, "perturb_genes": args.perturb_genes}
    manifest_path = args.data_root / "slices" / "slice_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {manifest_path}")


if __name__ == "__main__":
    main()
