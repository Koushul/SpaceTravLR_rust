"""Attach SPAC-seq H&E histology to AnnData for sc.pl.spatial."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import scanpy as sc
from matplotlib.image import imread

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent.parent
DEFAULT_MC38 = REPO / "analysis" / "mc38_visiumhd"
DOWNLOAD_SCRIPT = DEFAULT_MC38 / "download_spac_data.py"

LUNG_SLICES = {"Lung_Metastasis_M001", "Lung_Metastasis_M002", "Lung_Metastasis_M003"}


def dataset_type(slice_id: str) -> int:
    return 1 if slice_id in LUNG_SLICES or slice_id.startswith("Lung_") else 2


def raw_dir(mc38_dir: Path, slice_id: str) -> Path:
    return mc38_dir / slice_id / "raw" / "extracted"


def histology_paths(mc38_dir: Path, slice_id: str) -> tuple[Path, Path]:
    base = raw_dir(mc38_dir, slice_id)
    return base / "tissue_hires_image.png", base / "scalefactors_json.json"


def histology_ready(mc38_dir: Path, slice_id: str) -> bool:
    img, sf = histology_paths(mc38_dir, slice_id)
    return img.is_file() and sf.is_file()


def ensure_histology(
    slice_id: str,
    mc38_dir: Path = DEFAULT_MC38,
    *,
    skip_download: bool = False,
) -> Path:
    """Ensure tissue_hires_image.png + scalefactors exist; download raw zip if needed."""
    img, sf = histology_paths(mc38_dir, slice_id)
    if histology_ready(mc38_dir, slice_id):
        return img.parent
    if skip_download:
        raise FileNotFoundError(f"Histology missing for {slice_id} under {mc38_dir}")
    if not DOWNLOAD_SCRIPT.is_file():
        raise FileNotFoundError(f"Download script not found: {DOWNLOAD_SCRIPT}")
    out_dir = mc38_dir / slice_id
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(DOWNLOAD_SCRIPT),
            "--name",
            slice_id,
            "--dataset-type",
            str(dataset_type(slice_id)),
            "--out-dir",
            str(out_dir),
            "--components",
            "raw",
        ],
        check=True,
    )
    if not histology_ready(mc38_dir, slice_id):
        raise FileNotFoundError(f"Histology still missing after download for {slice_id}")
    return img.parent


def attach_histology(
    adata: sc.AnnData,
    slice_id: str,
    mc38_dir: Path = DEFAULT_MC38,
    *,
    library_id: str | None = None,
    skip_download: bool = False,
    img_key: str = "hires",
) -> sc.AnnData:
    """Populate adata.uns['spatial'] from SPAC portal raw_output bundle."""
    ensure_histology(slice_id, mc38_dir, skip_download=skip_download)
    img_path, sf_path = histology_paths(mc38_dir, slice_id)
    img = imread(img_path)
    sf = json.loads(sf_path.read_text())
    lib = library_id or slice_id
    adata.uns["spatial"] = {
        lib: {
            "images": {img_key: img},
            "scalefactors": {
                "tissue_hires_scalef": float(sf["tissue_hires_scalef"]),
                "spot_diameter_fullres": float(sf.get("spot_diameter_fullres", 50.0)),
            },
        }
    }
    return adata


def tumor_adata_from_parquet(parquet_path: Path, slice_id: str) -> sc.AnnData:
    """Minimal AnnData for sc.pl.spatial from exported spatial_tumor_*.parquet."""
    df = pd.read_parquet(parquet_path)
    obs = df[["cnn_leiden", "target_gene", "slice", "tag"]].copy()
    obs["cell_type"] = "tumor"
    adata = sc.AnnData(obs=obs)
    adata.obsm["spatial"] = df[["x", "y"]].to_numpy(dtype=np.float64)
    adata.obs["slice_id"] = slice_id
    return adata
