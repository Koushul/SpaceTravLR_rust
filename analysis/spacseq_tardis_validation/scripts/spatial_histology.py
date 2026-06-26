"""Attach SPAC-seq H&E histology to AnnData for sc.pl.spatial."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imread

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_MC38 = (ROOT.parent / "mc38_visiumhd").resolve()
DOWNLOAD_SCRIPT = DEFAULT_MC38 / "download_spac_data.py"

LUNG_SLICES = {"Lung_Metastasis_M001", "Lung_Metastasis_M002", "Lung_Metastasis_M003"}

FIGURE_PARAMS = {
    "dpi": 300,
    "bbox_inches": "tight",
    "format": "svg",
    "transparent": True,
}

PUBLICATION_RCPARAMS = {
    "text.usetex": False,
    "svg.fonttype": "none",
    "hatch.color": "black",
    "hatch.linewidth": 1.0,
}


def apply_publication_style() -> None:
    plt.rcParams.update(PUBLICATION_RCPARAMS)


def save_figure_png_svg(
    fig,
    path: Path | str,
    *,
    dpi: int = 200,
    transparent_png: bool = False,
) -> None:
    stem = Path(path).with_suffix("")
    png_kw: dict = {"dpi": dpi, "bbox_inches": "tight"}
    if transparent_png:
        png_kw["transparent"] = True
    fig.savefig(stem.with_suffix(".png"), **png_kw)
    apply_publication_style()
    fig.savefig(stem.with_suffix(".svg"), **FIGURE_PARAMS)


def segmentation_x_offset(mc38_dir: Path, slice_id: str) -> float:
    """Align StarDist polygon centroids with Visium fullres bin/image coordinates."""
    import geopandas as gpd

    mc38_dir = Path(mc38_dir).resolve()
    pos = pd.read_parquet(mc38_dir / slice_id / "raw" / "extracted" / "tissue_positions.parquet")
    geo = mc38_dir / slice_id / "segmentation/extracted/segmentation/graphclust_annotated_cell_segmentations.geojson"
    cells = gpd.read_file(geo)
    poly_x_min = float(cells.geometry.bounds["minx"].min())
    bin_x_min = float(pos.loc[pos.in_tissue == 1, "pxl_col_in_fullres"].min())
    return bin_x_min - poly_x_min


def tissue_hires_extent(mc38_dir: Path, slice_id: str) -> tuple[float, float, float, float]:
    mc38_dir = Path(mc38_dir).resolve()
    pos = pd.read_parquet(mc38_dir / slice_id / "raw" / "extracted" / "tissue_positions.parquet")
    sf = json.loads((mc38_dir / slice_id / "raw" / "extracted" / "scalefactors_json.json").read_text())
    scalef = float(sf["tissue_hires_scalef"])
    it = pos.loc[pos.in_tissue == 1]
    pad = 12.0
    xmin = float(it.pxl_col_in_fullres.min()) * scalef - pad
    xmax = float(it.pxl_col_in_fullres.max()) * scalef + pad
    ymin = float(it.pxl_row_in_fullres.min()) * scalef - pad
    ymax = float(it.pxl_row_in_fullres.max()) * scalef + pad
    return xmin, xmax, ymin, ymax


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
    x_offset = segmentation_x_offset(mc38_dir, slice_id)
    adata.uns["spatial"] = {
        lib: {
            "images": {img_key: img},
            "scalefactors": {
                "tissue_hires_scalef": float(sf["tissue_hires_scalef"]),
                "spot_diameter_fullres": float(sf.get("spot_diameter_fullres", 50.0)),
            },
            "segmentation_x_offset": x_offset,
            "mc38_dir": str(mc38_dir),
            "slice_id": slice_id,
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
    meta = adata.uns["spatial"][library_id]
    scalef = float(meta["scalefactors"]["tissue_hires_scalef"])
    xy = adata.obsm["spatial"].astype(np.float64).copy()
    x_offset = float(meta.get("segmentation_x_offset", 0.0))
    xy[:, 0] += x_offset
    return xy * scalef


def default_spot_size(adata: sc.AnnData, library_id: str) -> float:
    sf = adata.uns["spatial"][library_id]["scalefactors"]
    scalef = float(sf["tissue_hires_scalef"])
    diam_hires = float(sf.get("spot_diameter_fullres", 50.0)) * scalef
    density = max(adata.n_obs, 1)
    base = max(diam_hires * 1.2, 2.0)
    scale = float(np.clip((2200.0 / density) ** 0.35, 0.4, 1.05))
    return float(np.clip((base * scale) ** 2 * 0.65, 6.0, 55.0))


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
    rasterize: bool = False,
) -> None:
    """Draw H&E background with visible cell-level microniche overlay."""
    lib = adata.uns["spatial"][library_id]
    img = lib["images"]["hires"]
    xy = hires_coords(adata, library_id)
    size = spot_size if spot_size is not None else default_spot_size(adata, library_id)
    mc38_dir = Path(lib.get("mc38_dir", DEFAULT_MC38))
    sl = str(lib.get("slice_id", library_id))
    xmin, xmax, ymin, ymax = tissue_hires_extent(mc38_dir, sl)

    ax.imshow(
        img, origin="upper", interpolation="bilinear", alpha=img_alpha, zorder=0,
        extent=(0, img.shape[1], img.shape[0], 0), rasterized=rasterize,
    )
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
            rasterized=rasterize,
            zorder=2,
        )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
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


def _draw_he_background(
    ax, adata: sc.AnnData, library_id: str, *, img_alpha: float = 1.0, rasterize: bool = False,
) -> tuple[np.ndarray, float, tuple]:
    lib = adata.uns["spatial"][library_id]
    img = lib["images"]["hires"]
    xy = hires_coords(adata, library_id)
    size = default_spot_size(adata, library_id)
    mc38_dir = Path(lib.get("mc38_dir", DEFAULT_MC38))
    sl = str(lib.get("slice_id", library_id))
    xmin, xmax, ymin, ymax = tissue_hires_extent(mc38_dir, sl)
    ax.imshow(
        img, origin="upper", interpolation="bilinear", alpha=img_alpha, zorder=0,
        extent=(0, img.shape[1], img.shape[0], 0), rasterized=rasterize,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal")
    ax.axis("off")
    return xy, size, (xmin, xmax, ymin, ymax)


def plot_continuous_on_he(
    adata: sc.AnnData,
    color_key: str,
    ax,
    library_id: str,
    *,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = 0.0,
    spot_size: float | None = None,
    spot_alpha: float = 0.88,
    title: str = "",
    colorbar: bool = True,
    colorbar_label: str = "",
    rasterize: bool = False,
) -> matplotlib.cm.ScalarMappable | None:
    """H&E background with continuous score overlay (embedding-style)."""
    xy, size, _ = _draw_he_background(ax, adata, library_id, rasterize=rasterize)
    vals = pd.to_numeric(adata.obs[color_key], errors="coerce").to_numpy(dtype=float)
    if vmin is None:
        vmin = float(np.nanmin(vals)) if np.isfinite(vals).any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
    if vcenter is not None:
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=vmin, vmax=vmax)
    sc_plot = ax.scatter(
        xy[:, 0], xy[:, 1], c=vals, s=spot_size or size, cmap=cmap, norm=norm,
        alpha=spot_alpha, edgecolors="none", rasterized=rasterize, zorder=2,
    )
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=2)
    if colorbar:
        cb = plt.colorbar(sc_plot, ax=ax, fraction=0.05, pad=0.02, shrink=0.75)
        cb.ax.tick_params(labelsize=6)
        if colorbar_label:
            cb.set_label(colorbar_label, fontsize=7)
    return sc_plot


def plot_embedding_spatial(
    adata: sc.AnnData,
    color_key: str,
    ax,
    *,
    cmap=None,
    vmin: float | None = None,
    vmax: float | None = None,
    vcenter: float | None = None,
    categorical: bool = False,
    palette: dict[str, object] | None = None,
    title: str = "",
    size: float = 8.0,
    colorbar: bool = True,
    colorbar_label: str = "",
    rasterize: bool = True,
) -> matplotlib.cm.ScalarMappable | None:
    """sc.pl.embedding-style scatter on obsm['spatial'] without H&E."""
    xy = adata.obsm["spatial"]
    if categorical:
        labels = adata.obs[color_key].astype(str)
        palette = palette or {}
        for lab in sorted(labels.unique()):
            if lab in ("nan", "unassigned"):
                continue
            m = labels == lab
            ax.scatter(
                xy[m, 0], xy[m, 1], c=[palette.get(lab, "#888888")], s=size,
                alpha=0.85, edgecolors="none", rasterized=True,
            )
    else:
        vals = pd.to_numeric(adata.obs[color_key], errors="coerce").to_numpy(dtype=float)
        if vcenter is not None:
            if vmin is None or vmax is None:
                lim = max(0.35, float(np.nanpercentile(np.abs(vals - vcenter), 95)) if np.isfinite(vals).any() else 1.0)
                vmin, vmax = -lim, lim
            from matplotlib.colors import TwoSlopeNorm
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        else:
            from matplotlib.colors import Normalize
            if vmin is None:
                vmin = float(np.nanmin(vals)) if np.isfinite(vals).any() else 0.0
            if vmax is None:
                vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
            norm = Normalize(vmin=vmin, vmax=vmax)
        sc_plot = ax.scatter(
            xy[:, 0], xy[:, 1], c=vals, s=size, cmap=cmap or "RdBu_r", norm=norm,
            alpha=0.9, edgecolors="none", rasterized=rasterize,
        )
        if colorbar:
            cb = plt.colorbar(sc_plot, ax=ax, fraction=0.05, pad=0.02, shrink=0.75)
            cb.ax.tick_params(labelsize=6)
            if colorbar_label:
                cb.set_label(colorbar_label, fontsize=7)
        if title:
            ax.set_title(title, fontsize=9, fontweight="bold", pad=2)
        ax.set_aspect("equal")
        ax.axis("off")
        return sc_plot
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=2)
    return None
