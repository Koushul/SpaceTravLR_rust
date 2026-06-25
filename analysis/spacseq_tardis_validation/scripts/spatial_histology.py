"""Attach SPAC-seq H&E histology to AnnData for sc.pl.spatial."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.image import imread

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_MC38 = (ROOT.parent / "mc38_visiumhd").resolve()
DOWNLOAD_SCRIPT = DEFAULT_MC38 / "download_spac_data.py"

LUNG_SLICES = {"Lung_Metastasis_M001", "Lung_Metastasis_M002", "Lung_Metastasis_M003"}


def dataset_type(slice_id: str) -> int:
    return 1 if slice_id in LUNG_SLICES or slice_id.startswith("Lung_") else 2


def raw_dir(mc38_dir: Path, slice_id: str) -> Path:
    return mc38_dir / slice_id / "raw" / "extracted"


def histology_paths(mc38_dir: Path, slice_id: str) -> tuple[Path, Path]:
    base = Path(mc38_dir).resolve() / slice_id / "raw" / "extracted"
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
    mc38_dir = Path(mc38_dir).resolve()
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
    mc38_dir = Path(mc38_dir).resolve()
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


def hires_coords(adata: sc.AnnData, library_id: str) -> np.ndarray:
    scalef = float(adata.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"])
    return adata.obsm["spatial"].astype(np.float64) * scalef


def default_spot_size(adata: sc.AnnData, library_id: str) -> float:
    sf = adata.uns["spatial"][library_id]["scalefactors"]
    scalef = float(sf["tissue_hires_scalef"])
    diam_hires = float(sf.get("spot_diameter_fullres", 50.0)) * scalef
    density = max(adata.n_obs, 1)
    base = max(diam_hires * 1.6, 2.5)
    scale = float(np.clip((2200.0 / density) ** 0.35, 0.45, 1.25))
    return float(np.clip((base * scale) ** 2, 10.0, 90.0))


def plot_microniche_on_he(
    adata: sc.AnnData,
    color_key: str,
    ax,
    library_id: str,
    palette: dict[str, object],
    *,
    img_alpha: float = 1.0,
    spot_size: float | None = None,
    spot_alpha: float = 0.78,
    edgecolor: str = "white",
    edge_width: float = 0.25,
    title: str = "",
    legend: bool = True,
    legend_fontsize: float = 6.0,
) -> None:
    """Draw H&E background with visible cell-level microniche overlay."""
    lib = adata.uns["spatial"][library_id]
    img = lib["images"]["hires"]
    xy = hires_coords(adata, library_id)
    size = spot_size if spot_size is not None else default_spot_size(adata, library_id)

    ax.imshow(img, origin="upper", interpolation="bilinear", alpha=img_alpha, zorder=0)
    labels = adata.obs[color_key].astype(str)
    for lab in labels.unique():
        if lab in ("nan", "unassigned"):
            continue
        mask = labels == lab
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=size,
            c=[palette.get(lab, "#888888")],
            alpha=spot_alpha,
            edgecolors=edgecolor,
            linewidths=edge_width,
            rasterized=True,
            zorder=2,
        )
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    if legend:
        handles = []
        for lab in sorted(l for l in labels.unique() if l not in ("nan", "unassigned")):
            handles.append(
                plt.Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor=palette.get(lab, "#888888"),
                    markersize=max(3.0, legend_fontsize * 0.75), label=lab,
                )
            )
        if handles:
            ax.legend(
                handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                fontsize=legend_fontsize, frameon=False, borderaxespad=0.0,
            )
