#!/usr/bin/env python3
"""
Visium (H&E) -> DeepSpot virtual expression -> AnnData .h5ad (SpaceTravLR-friendly).

This script reproduces the flow from the DeepSpot example notebook
`GettingStartedWithDeepSpot_3.1_inference_pretrained_models.ipynb`, with an option
to use 10x Space Ranger `spatial/` spot coordinates instead of a synthetic grid.

Notes
-----
- DeepSpot pretrained heads predict a *panel* of highly variable genes shipped with
  each checkpoint (often ~5k genes), not every Ensembl/GENCODE gene. For broader
  "whole-transcriptome" coverage, see the paper's atlas-projection / DeepCell
  workflows; this script exports the model's native prediction matrix.
- `deepspot.utils.utils_image.predict_spot_spatial_transcriptomics_from_image_path`
  already applies `inverse_transform` on the current upstream DeepSpot main branch.
  Do not apply `inverse_transform` a second time unless you know your checkout
  differs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from deepspot.utils.utils_dataloader import compute_neighbors
from deepspot.utils.utils_image import compute_mini_tiles, detach_and_convert
from deepspot.utils.utils_image import (
    get_low_res_image,
    get_morphology_model_and_preprocess,
    predict_spot_spatial_transcriptomics_from_image_path,
)


def _has_pyvips() -> bool:
    try:
        import pyvips  # noqa: F401

        return True
    except Exception:
        return False


def crop_tile_np(img: np.ndarray, x_pixel: float, y_pixel: float, diameter: int) -> np.ndarray:
    """Crop square tile; x_pixel = row, y_pixel = col (DeepSpot convention)."""
    half = int(diameter // 2)
    r0 = int(x_pixel) - half
    c0 = int(y_pixel) - half
    tile = img[r0 : r0 + diameter, c0 : c0 + diameter, :3].astype(np.float32)
    return tile


def load_image_rgb_numpy(image_path: Path) -> np.ndarray:
    """Load H&E as H×W×3 float32 RGB in [0,255]-ish range for cropping."""
    return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)


def get_low_res_image_pil(image_path: Path, downsample_factor: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    small = img.resize((max(1, w // downsample_factor), max(1, h // downsample_factor)), Image.BILINEAR)
    return np.asarray(small)


def get_morphology_uni_timm_imagenet(device: torch.device):
    """UNI-compatible ViT-L/14 architecture with ImageNet-pretrained weights (not pathology FM).

    Use only when official UNI/H-optimus weights are unavailable. Expect domain shift vs paper.
    """
    import timm
    from torchvision import transforms

    morphology_model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        init_values=1e-5,
        num_classes=0,
        dynamic_img_size=True,
        pretrained=True,
    )
    morphology_model.eval()
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return morphology_model, preprocess, 1024


def predict_spot_spatial_transcriptomics_pil(
    image_path: Path,
    adata: ad.AnnData,
    spot_diameter: int,
    n_mini_tiles: int,
    preprocess,
    morphology_model,
    model_expression,
    device: torch.device,
    super_resolution: bool = False,
    neighbor_radius: int = 1,
) -> np.ndarray:
    """Same logic as DeepSpot ``predict_spot_spatial_transcriptomics_from_image_path`` using PIL/numpy (no pyvips)."""
    img_rgb = load_image_rgb_numpy(image_path)
    counts = []
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        with torch.inference_mode():
            for _, spot in tqdm(adata.obs.iterrows(), total=len(adata.obs), desc="DeepSpot inference"):
                neighbors_barcodes = compute_neighbors(spot, adata.obs, radius=neighbor_radius)
                neighbors_barcodes = [
                    b for b in neighbors_barcodes.split("___") if b
                ]

                X_spot = crop_tile_np(img_rgb, spot.x_pixel, spot.y_pixel, spot_diameter).astype(np.uint8)
                X_subspot = compute_mini_tiles(X_spot, n_mini_tiles, super_resolution)
                X_neighbors = []
                for _, spot_neighbor in adata.obs.query("barcode in @neighbors_barcodes").iterrows():
                    X_neighbors.append(
                        crop_tile_np(
                            img_rgb,
                            spot_neighbor.x_pixel,
                            spot_neighbor.y_pixel,
                            spot_diameter,
                        ).astype(np.uint8)
                    )

                if len(X_neighbors) == 0:
                    X_neighbors = [np.zeros_like(X_spot)]

                X_spot = preprocess(X_spot).to(device).float()
                X_subspot = torch.stack([preprocess(x) for x in X_subspot]).to(device).float()
                X_neighbors = torch.stack([preprocess(x).to(device) for x in X_neighbors]).to(device).float()

                X_spot = morphology_model(X_spot[None, ])
                X_subspot = morphology_model(X_subspot)
                X_neighbors = morphology_model(X_neighbors)

                X_spot = detach_and_convert(X_spot)
                X_subspot = detach_and_convert(X_subspot)
                X_neighbors = detach_and_convert(X_neighbors)

                X = [X_spot, X_subspot, X_neighbors]
                expr = model_expression(X)
                expr = expr.detach().cpu().numpy()
                expr = model_expression.inverse_transform(expr)
                assert np.isnan(expr).sum() == 0, spot
                counts.append(expr)
    counts = np.array(counts).squeeze()
    return counts


def _load_scalefactors(spatial_dir: Path) -> dict:
    p = spatial_dir / "scalefactors_json.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    with p.open() as f:
        return json.load(f)


def _read_tissue_positions(spatial_dir: Path) -> pd.DataFrame:
    for name in ("tissue_positions.csv", "tissue_positions_list.csv"):
        p = spatial_dir / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df.columns = [str(c).lower() for c in df.columns]
        if (
            "px_row_fullres" not in df.columns
            or "px_col_fullres" not in df.columns
        ):
            df = pd.read_csv(p, header=None, dtype={0: str})
            if df.shape[1] >= 6:
                df = df.iloc[:, :6]
                df.columns = [
                    "barcode",
                    "in_tissue",
                    "array_row",
                    "array_col",
                    "px_row_fullres",
                    "px_col_fullres",
                ]
            else:
                raise ValueError(f"Unexpected tissue positions layout in {p}")
        needed = [
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "px_row_fullres",
            "px_col_fullres",
        ]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{p} missing columns {missing}; got {list(df.columns)}")
        return df[needed].copy()
    raise FileNotFoundError(
        f"No tissue_positions*.csv under {spatial_dir} "
        "(expected tissue_positions.csv or tissue_positions_list.csv)."
    )


def build_adata_from_visium_spatial(
    spatial_dir: Path,
    genes_df: pd.DataFrame,
    selected_genes_bool: np.ndarray,
    sample_id: str,
    white_cutoff: float,
    use_pillow: bool,
) -> tuple[ad.AnnData, Path, int]:
    """Create an AnnData with spot layout + dummy counts for DeepSpot inference."""
    sf = _load_scalefactors(spatial_dir)
    pos = _read_tissue_positions(spatial_dir)
    pos["in_tissue"] = pos["in_tissue"].astype(int)
    pos = pos[pos["in_tissue"] == 1].copy()

    n_spots = len(pos)
    counts = np.zeros((n_spots, int(selected_genes_bool.sum())), dtype=np.float32)

    obs = pd.DataFrame(
        {
            "barcode": pos["barcode"].astype(str).values,
            "array_row": pos["array_row"].values,
            "array_col": pos["array_col"].values,
            # DeepSpot uses array indices for neighborhood queries on the grid.
            "x_array": pos["array_row"].values,
            "y_array": pos["array_col"].values,
            # Notebook naming: x_pixel is vertical / row, y_pixel is horizontal / col.
            "x_pixel": pos["px_row_fullres"].astype(float).values,
            "y_pixel": pos["px_col_fullres"].astype(float).values,
            "sampleID": sample_id,
        },
        index=pos["barcode"].astype(str).values,
    )
    obs["barcode"] = obs["barcode"].astype(str)

    adata = ad.AnnData(counts, obs=obs)
    adata.var.index = genes_df[selected_genes_bool]["gene_name"].astype(str).values

    # tissue whitening filter (from DeepSpot notebook)
    image_candidates = [
        spatial_dir / "tissue_fullres_image.tif",
        spatial_dir / "tissue_fullres_image.png",
        spatial_dir / "tissue_hires_image.png",
        spatial_dir / "detected_tissue_image.jpg",
    ]
    image_path = next((p for p in image_candidates if p.exists()), None)
    if image_path is None:
        raise FileNotFoundError(
            "Could not locate an H&E image under spatial/. "
            "Pass --image explicitly."
        )

    if use_pillow:
        img_rgb = load_image_rgb_numpy(image_path)
        spot_diameter = int(float(sf["spot_diameter_fullres"]))
        is_white = []
        for _, row in tqdm(obs.iterrows(), total=len(obs), desc="White-tissue filter"):
            main_tile = crop_tile_np(img_rgb, row.x_pixel, row.y_pixel, spot_diameter)
            is_white.append(float(np.mean(main_tile)))
    else:
        import pyvips

        from deepspot.utils.utils_image import crop_tile

        image = pyvips.Image.new_from_file(str(image_path))
        is_white = []
        spot_diameter = int(float(sf["spot_diameter_fullres"]))
        for _, row in tqdm(obs.iterrows(), total=len(obs), desc="White-tissue filter"):
            main_tile = crop_tile(image, row.x_pixel, row.y_pixel, spot_diameter)
            main_tile = main_tile[:, :, :3]
            is_white.append(float(np.mean(main_tile)))
    obs["is_white"] = is_white
    obs["is_white_bool"] = (obs["is_white"].values > white_cutoff).astype(int)
    adata.obs = obs
    adata = adata[adata.obs["is_white_bool"] == 0].copy()
    return adata, image_path, int(spot_diameter)


def build_adata_grid_from_image(
    image_path: Path,
    genes_df: pd.DataFrame,
    selected_genes_bool: np.ndarray,
    sample_id: str,
    spot_diameter: int,
    spot_distance: int,
    white_cutoff: float,
    use_pillow: bool,
    max_spots: Optional[int],
) -> ad.AnnData:
    """Synthetic square grid (DeepSpot toy notebook pattern)."""
    if use_pillow:
        image = load_image_rgb_numpy(image_path)
        height, width = image.shape[0], image.shape[1]
    else:
        import pyvips

        from deepspot.utils.utils_image import crop_tile

        vips_img = pyvips.Image.new_from_file(str(image_path))
        height, width = vips_img.height, vips_img.width

    coord = []
    for i, x in enumerate(range(spot_diameter + 1, height - spot_diameter - 1, spot_distance)):
        for j, y in enumerate(range(spot_diameter + 1, width - spot_diameter - 1, spot_distance)):
            coord.append([i, j, x, y])
            if max_spots is not None and len(coord) >= max_spots:
                break
        if max_spots is not None and len(coord) >= max_spots:
            break

    coord = pd.DataFrame(coord, columns=["x_array", "y_array", "x_pixel", "y_pixel"])
    coord.index = coord.index.astype(str)

    is_white = []
    if use_pillow:
        for _, row in tqdm(coord.iterrows(), total=len(coord), desc="White-tissue filter"):
            main_tile = crop_tile_np(image, row.x_pixel, row.y_pixel, spot_diameter)
            is_white.append(float(np.mean(main_tile)))
    else:
        for _, row in tqdm(coord.iterrows(), total=len(coord), desc="White-tissue filter"):
            main_tile = crop_tile(vips_img, row.x_pixel, row.y_pixel, spot_diameter)
            main_tile = main_tile[:, :, :3]
            is_white.append(float(np.mean(main_tile)))

    counts = np.zeros((len(coord), int(selected_genes_bool.sum())), dtype=np.float32)
    obs = coord.copy()
    obs["is_white"] = is_white
    obs["is_white_bool"] = (obs["is_white"].values > white_cutoff).astype(int)
    obs["sampleID"] = sample_id
    obs["barcode"] = obs.index.astype(str)

    adata = ad.AnnData(counts, obs=obs)
    adata.var.index = genes_df[selected_genes_bool]["gene_name"].astype(str).values
    adata = adata[adata.obs["is_white_bool"] == 0].copy()
    return adata


def attach_downsampled_hires_for_squidpy(
    adata: ad.AnnData,
    image_path: Path,
    downsample_factor: int,
    use_pillow: bool,
) -> None:
    """Attach a downsampled RGB preview + coordinates scaled to that preview.

    Does **not** modify ``obsm['spatial']`` (kept in full-resolution pixel space).
    Squidpy/Scanpy scatter can use ``spatial_key='spatial_hires'`` together with
    ``uns['spatial']['library_id']['images']['hires']``.
    """
    if use_pillow:
        img_rgb = get_low_res_image_pil(image_path, downsample_factor)
    else:
        img_rgb = get_low_res_image(str(image_path), downsample_factor)
    if img_rgb.shape[-1] != 3:
        raise ValueError("Expected an RGB H&E image with 3 channels.")

    spatial_xy = adata.obs[["y_pixel", "x_pixel"]].values.astype(float) / downsample_factor
    adata.obsm["spatial_hires"] = spatial_xy
    adata.uns["spatial"] = {"library_id": {"images": {"hires": np.asarray(img_rgb)}}}


def add_cell_type_placeholder(adata: ad.AnnData, column: str = "cell_type") -> None:
    """SpaceTravLR expects a cluster annotation column; use a placeholder if missing."""
    if column not in adata.obs.columns:
        adata.obs[column] = "spot"


def cluster_for_cell_type(adata: ad.AnnData, layer_key: str, column: str = "cell_type") -> None:
    """Cheap Leiden clusters on predicted expression for a real `cell_type` column."""
    x = adata.layers[layer_key] if layer_key in adata.layers else adata.X
    tmp = ad.AnnData(x.copy(), obs=adata.obs.copy(), var=adata.var.copy())
    sc.pp.pca(tmp, n_comps=min(50, tmp.n_vars - 1))
    sc.pp.neighbors(tmp)
    sc.tl.leiden(tmp, resolution=1.0, flavor="igraph", n_iterations=2)
    adata.obs[column] = tmp.obs["leiden"].astype(str).values


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="DeepSpot Visium H&E -> virtual ST .h5ad")
    p.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
        help="Unzipped Zenodo folder, e.g. DeepSpot_pretrained_model_weights/Colon_HEST1K",
    )
    p.add_argument(
        "--out-h5ad",
        type=Path,
        required=True,
        help="Output AnnData path (.h5ad)",
    )
    p.add_argument(
        "--spatial-dir",
        type=Path,
        default=None,
        help="10x `spatial/` directory (contains tissue_positions*.csv & scalefactors_json.json)",
    )
    p.add_argument(
        "--image",
        type=Path,
        default=None,
        help="H&E image path (full-res coordinates). Required for --mode grid; "
        "optional for Visium mode if a tissue image exists under spatial/",
    )
    p.add_argument(
        "--mode",
        choices=("visium", "grid"),
        default="visium",
        help="visium: use Space Ranger spot coordinates; grid: synthetic square grid (WSI toy)",
    )
    p.add_argument("--sample-id", type=str, default="sample1")
    p.add_argument(
        "--foundation-weights",
        type=Path,
        default=None,
        help="Path to pathology FM checkpoint (UNI/Phikon/H-optimus .bin). Not used with --foundation-timm-imagenet.",
    )
    p.add_argument(
        "--foundation-timm-imagenet",
        action="store_true",
        help="Use timm ImageNet ViT-L/14 (UNI architecture) instead of a local FM file. "
        "Use when HF-gated UNI weights are unavailable; expect domain shift vs the paper.",
    )
    p.add_argument(
        "--pillow",
        action="store_true",
        help="Load/crop H&E with Pillow instead of pyvips (no libvips required). "
        "Recommended when pyvips is not installed.",
    )
    p.add_argument(
        "--max-spots",
        type=int,
        default=None,
        help="Grid mode only: stop after this many candidate spots (useful for CPU demos).",
    )
    p.add_argument("--neighbor-radius", type=int, default=1)
    p.add_argument("--white-cutoff", type=float, default=200.0)
    p.add_argument("--downsample-factor", type=int, default=10)
    p.add_argument(
        "--skip-squidpy-image",
        action="store_true",
        help="Do not attach a downsampled H&E preview to uns['spatial'] / obsm['spatial_hires']",
    )
    p.add_argument(
        "--cluster-cell-type",
        action="store_true",
        help="Run PCA+Leiden on predictions and fill obs['cell_type']",
    )
    args = p.parse_args(argv)

    if args.foundation_timm_imagenet and args.foundation_weights is not None:
        print(
            "Ignoring --foundation-weights because --foundation-timm-imagenet was set.",
            file=sys.stderr,
        )
    if not args.foundation_timm_imagenet and args.foundation_weights is None:
        p.error("Either --foundation-weights or --foundation-timm-imagenet is required.")

    use_pillow = bool(args.pillow) or (not _has_pyvips())
    if use_pillow and not args.pillow and not _has_pyvips():
        print(
            "libvips not found; using Pillow for image I/O (pass --pillow to silence).",
            file=sys.stderr,
        )

    weights_dir: Path = args.weights_dir
    model_weights = weights_dir / "final_model.pkl"
    model_hparam_path = weights_dir / "top_param_overall.yaml"
    gene_path = weights_dir / "info_highly_variable_genes.csv"
    for req in (model_weights, model_hparam_path, gene_path):
        if not req.exists():
            raise FileNotFoundError(f"Missing expected file: {req}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(model_hparam_path, "r") as fh:
        model_hparam = yaml.safe_load(fh)
    image_feature_model = model_hparam["image_feature_model"]

    if args.foundation_timm_imagenet:
        if image_feature_model != "uni":
            raise ValueError(
                "--foundation-timm-imagenet only matches checkpoints trained with "
                f"`image_feature_model: uni` (this bundle uses `{image_feature_model}`)."
            )

    genes = pd.read_csv(gene_path)
    selected_genes_bool = genes["isPredicted"].values

    # Layout AnnData for inference
    if args.mode == "visium":
        if args.spatial_dir is None:
            raise ValueError("--spatial-dir is required for --mode visium")
        adata, resolved_image, spot_diameter = build_adata_from_visium_spatial(
            args.spatial_dir,
            genes,
            selected_genes_bool,
            args.sample_id,
            args.white_cutoff,
            use_pillow,
        )
        image_path = args.image if args.image is not None else resolved_image
    else:
        if args.image is None:
            raise ValueError("--image is required for --mode grid")
        image_path = args.image
        spot_diameter = int(model_hparam.get("spot_diameter", 100))
        spot_distance = int(model_hparam.get("spot_distance", spot_diameter))
        adata = build_adata_grid_from_image(
            image_path,
            genes,
            selected_genes_bool,
            args.sample_id,
            spot_diameter,
            spot_distance,
            args.white_cutoff,
            use_pillow,
            args.max_spots,
        )

    # Models
    model_expression = torch.load(model_weights, map_location=device, weights_only=False)
    model_expression.to(device)
    model_expression.eval()

    if args.foundation_timm_imagenet:
        morphology_model, preprocess, _feat_dim = get_morphology_uni_timm_imagenet(device)
        morphology_model.to(device)
        morphology_model.eval()
    else:
        morphology_model, preprocess, _feat_dim = get_morphology_model_and_preprocess(
            model_name=image_feature_model,
            device=device,
            model_path=str(args.foundation_weights),
        )
        morphology_model.to(device)
        morphology_model.eval()

    n_mini_tiles = 9
    if use_pillow:
        counts = predict_spot_spatial_transcriptomics_pil(
            Path(image_path),
            adata,
            spot_diameter,
            n_mini_tiles,
            preprocess,
            morphology_model,
            model_expression,
            device,
            super_resolution=False,
            neighbor_radius=args.neighbor_radius,
        )
    else:
        counts = predict_spot_spatial_transcriptomics_from_image_path(
            str(image_path),
            adata,
            spot_diameter,
            n_mini_tiles,
            preprocess,
            morphology_model,
            model_expression,
            device,
            super_resolution=False,
            neighbor_radius=args.neighbor_radius,
        )

    counts = np.asarray(counts, dtype=np.float32)
    counts[counts < 0] = 0.0

    adata.layers["imputed_count"] = counts
    adata.X = counts
    # Primary gene names from training manifest
    adata.var["gene_symbols"] = adata.var_names
    if args.foundation_timm_imagenet:
        adata.uns["morphology_backbone"] = "timm_vit_large_patch16_224_imagenet1k"
        adata.uns["morphology_note"] = (
            "ImageNet-pretrained ViT-L/14 (UNI architecture), not official UNI "
            "weights. Approve https://huggingface.co/MahmoodLab/UNI and use "
            "--foundation-weights for the pathology FM from the paper."
        )

    # SpaceTravLR contract: use the same units for spatial radii as obsm["spatial"] (here: full-res pixels).
    adata.obsm["spatial"] = adata.obs[["y_pixel", "x_pixel"]].values.astype(float)

    add_cell_type_placeholder(adata, column="cell_type")
    if args.cluster_cell_type:
        cluster_for_cell_type(adata, layer_key="imputed_count", column="cell_type")

    if not args.skip_squidpy_image:
        try:
            attach_downsampled_hires_for_squidpy(
                adata, Path(image_path), args.downsample_factor, use_pillow
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"Warning: could not attach hires preview ({exc}). "
                "Re-run with --skip-squidpy-image if you only need coordinates + expression.",
                file=sys.stderr,
            )

    args.out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out_h5ad)
    print(f"Wrote {args.out_h5ad} with shape {adata.shape}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
