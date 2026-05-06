#!/usr/bin/env python3
"""
Run DeepSpot's gene head on *random* morphology features with the correct shape.

This exists so the repo can produce a small `.h5ad` end-to-end when Hugging Face
gated weights (UNI / H-optimus-0 / …) are unavailable in the environment.

The resulting expression matrix is **not biologically meaningful** — it only
validates tensor shapes, checkpoint loading, and AnnData export.

Real inference must use `visium_he_to_h5ad.py` with downloaded FM checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from deepspot.utils.utils_dataloader import compute_neighbors


def crop_tile_np(img: np.ndarray, x_pixel: float, y_pixel: float, diameter: int) -> np.ndarray:
    """Crop a square tile; x_pixel=row, y_pixel=col (DeepSpot convention)."""
    half = int(diameter // 2)
    r0 = int(x_pixel) - half
    c0 = int(y_pixel) - half
    return img[r0 : r0 + diameter, c0 : c0 + diameter, :3]


def build_grid_adata(
    image_path: Path,
    genes: pd.DataFrame,
    selected_genes_bool: np.ndarray,
    sample_id: str,
    spot_diameter: int,
    spot_distance: int,
    white_cutoff: float,
    max_spots: int | None,
):
    from PIL import Image

    img = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = img.shape[0], img.shape[1]

    coord = []
    for i, x in enumerate(range(spot_diameter + 1, height - spot_diameter - 1, spot_distance)):
        for j, y in enumerate(range(spot_diameter + 1, width - spot_diameter - 1, spot_distance)):
            coord.append([i, j, x, y])
            if max_spots is not None and len(coord) >= max_spots:
                break
        if max_spots is not None and len(coord) >= max_spots:
            break

    coord = pd.DataFrame(
        coord, columns=["x_array", "y_array", "x_pixel", "y_pixel"]
    )
    coord.index = coord.index.astype(str)

    is_white = []
    for _, row in tqdm(coord.iterrows(), total=len(coord), desc="White-tissue filter"):
        main_tile = crop_tile_np(img, row.x_pixel, row.y_pixel, spot_diameter)
        is_white.append(float(np.mean(main_tile)))

    counts = np.zeros((len(coord), int(selected_genes_bool.sum())), dtype=np.float32)
    obs = coord.copy()
    obs["is_white"] = is_white
    obs["is_white_bool"] = (obs["is_white"].values > white_cutoff).astype(int)
    obs["sampleID"] = sample_id
    obs["barcode"] = obs.index.astype(str)

    adata = ad.AnnData(counts, obs=obs)
    adata.var.index = genes[selected_genes_bool]["gene_name"].astype(str).values
    adata = adata[adata.obs["is_white_bool"] == 0].copy()
    return adata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights-dir", type=Path, required=True)
    p.add_argument("--out-h5ad", type=Path, required=True)
    p.add_argument(
        "--image",
        type=Path,
        required=True,
        help="H&E image (DeepSpot toy example: ZEN38_without_fud.jpg)",
    )
    p.add_argument("--sample-id", type=str, default="ZEN38")
    p.add_argument("--max-spots", type=int, default=24)
    p.add_argument("--white-cutoff", type=float, default=200.0)
    args = p.parse_args()

    model_weights = args.weights_dir / "final_model.pkl"
    model_hparam_path = args.weights_dir / "top_param_overall.yaml"
    gene_path = args.weights_dir / "info_highly_variable_genes.csv"
    with open(model_hparam_path, "r") as fh:
        hparam = yaml.safe_load(fh)

    device = torch.device("cpu")
    model = torch.load(model_weights, map_location=device, weights_only=False)
    model.eval()
    input_size = int(model.hparams.input_size)

    genes = pd.read_csv(gene_path)
    selected_genes_bool = genes["isPredicted"].values

    spot_diameter = int(hparam.get("spot_diameter", 100))
    spot_distance = int(hparam.get("spot_distance", spot_diameter))
    adata = build_grid_adata(
        args.image,
        genes,
        selected_genes_bool,
        args.sample_id,
        spot_diameter,
        spot_distance,
        args.white_cutoff,
        args.max_spots,
    )

    counts = []
    for _, spot in tqdm(adata.obs.iterrows(), total=len(adata.obs), desc="Dummy FM"):
        nb_raw = compute_neighbors(spot, adata.obs, radius=1)
        nb = [b for b in nb_raw.split("___") if b]
        n_nb = max(1, len(nb))
        rng = torch.Generator(device=device)
        rng.manual_seed(42 + int(spot.barcode))
        x_spot = torch.randn(1, 1, input_size, device=device, generator=rng)
        x_sub = torch.randn(1, 9, input_size, device=device, generator=rng)
        x_nei = torch.randn(1, n_nb, input_size, device=device, generator=rng)
        with torch.inference_mode():
            expr = model([x_spot, x_sub, x_nei])
        expr = expr.detach().cpu().numpy()
        expr = model.inverse_transform(expr)
        counts.append(expr.reshape(-1))
    counts = np.asarray(counts, dtype=np.float32)
    counts[counts < 0] = 0.0

    adata.layers["imputed_count"] = counts
    adata.X = counts
    adata.obsm["spatial"] = adata.obs[["y_pixel", "x_pixel"]].values.astype(float)
    adata.obs["cell_type"] = "spot"
    adata.obs["dummy_fm_random_features"] = True
    adata.uns["deepspot_dummy_note"] = (
        "Predictions used random morphology features for CI/demo only. "
        "Replace with real foundation-model inference for science."
    )

    args.out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out_h5ad)
    print(f"Wrote {args.out_h5ad} shape={adata.shape} (dummy FM)")


if __name__ == "__main__":
    main()
