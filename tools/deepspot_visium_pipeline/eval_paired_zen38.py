#!/usr/bin/env python3
"""
Paired Visium H&E + measured expression: run DeepSpot and compare to ground truth.

Uses the public COAD **ZEN38** slice shipped with the DeepSpot repo
(`example_data/data/h5ad/ZEN38.h5ad` + `uns['spatial']` hires image) — the same
sample as the Colon/HEST DeepSpot release is intended for.

Outputs an AnnData with log-normalized measured expression and DeepSpot
`imputed_count` for the gene intersection, plus CSVs of per-gene Pearson r.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import yaml
from scipy.stats import pearsonr
from tqdm import tqdm

# Reuse pipeline helpers (same venv + PYTHONPATH=DeepSpot root)
from visium_he_to_h5ad import (
    get_morphology_uni_timm_imagenet,
    predict_spot_spatial_transcriptomics_pil,
)
from deepspot.utils.utils_image import get_morphology_model_and_preprocess


def _harvest_hires_image(adata: ad.AnnData) -> np.ndarray:
    return np.asarray(
        adata.uns["spatial"]["library_id"]["images"]["hires"],
        dtype=np.uint8,
    )


def _build_spot_table(adata: ad.AnnData) -> pd.DataFrame:
    sp = np.asarray(adata.obsm["spatial"])
    # Visium/Scanpy: column 0 ≈ x (col), column 1 ≈ y (row) — see ZEN38 corrs to pxl_col/row
    return pd.DataFrame(
        {
            "x_array": adata.obs["x_array"].values,
            "y_array": adata.obs["y_array"].values,
            "x_pixel": sp[:, 1].astype(float),
            "y_pixel": sp[:, 0].astype(float),
            "barcode": adata.obs_names.astype(str).values,
            "sampleID": "ZEN38",
        },
        index=adata.obs_names.astype(str),
    )


def _estimate_spot_diameter_hires(adata: ad.AnnData) -> int:
    from scipy.spatial import KDTree

    xy = np.asarray(adata.obsm["spatial"])
    tree = KDTree(xy)
    d, _ = tree.query(xy, k=2)
    nn = d[:, 1]
    return int(max(32, min(200, round(float(np.median(nn)) * 2.2))))


def _marker_table(marker_genes: list[str], names: set[str]) -> list[str]:
    return [g for g in marker_genes if g in names]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--measured-h5ad",
        type=Path,
        default=None,
        help="Paired Visium h5ad (default: DeepSpot example ZEN38.h5ad)",
    )
    p.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
        help="e.g. DeepSpot_pretrained_model_weights/Colon_HEST1K",
    )
    p.add_argument(
        "--out-h5ad",
        type=Path,
        required=True,
    )
    p.add_argument(
        "--out-corr-csv",
        type=Path,
        default=None,
        help="Per-gene Pearson r (all intersected genes)",
    )
    p.add_argument(
        "--out-markers-csv",
        type=Path,
        default=None,
        help="Subset of key marker correlations",
    )
    p.add_argument(
        "--foundation-weights",
        type=Path,
        default=None,
        help="UNI (or other FM) .bin; use with real gated weights",
    )
    p.add_argument(
        "--foundation-timm-imagenet",
        action="store_true",
        help="ImageNet ViT-L when UNI path unavailable (domain shift).",
    )
    p.add_argument(
        "--max-spots",
        type=int,
        default=None,
        help="Debug: only first N spots in obs order",
    )
    p.add_argument(
        "--white-cutoff",
        type=float,
        default=500.0,
        help="Skip only very bright empty tiles (tissue is usually < 500 mean RGB)",
    )
    args = p.parse_args()

    deepspot_root = Path(__file__).resolve().parent
    default_zen = deepspot_root / "zen38_source" / "ZEN38.h5ad"
    if args.measured_h5ad is None:
        # Prefer user-cloned DeepSpot path; else document
        for candidate in [
            default_zen,
            Path("/tmp/DeepSpot/example_data/data/h5ad/ZEN38.h5ad"),
        ]:
            if candidate.exists():
                args.measured_h5ad = candidate
                break
        if args.measured_h5ad is None:
            print(
                "Place ZEN38.h5ad at tools/deepspot_visium_pipeline/zen38_source/ "
                "or pass --measured-h5ad (copy from DeepSpot example_data).",
                file=sys.stderr,
            )
            return 1

    raw = sc.read_h5ad(args.measured_h5ad)
    if "spatial" not in raw.uns or "library_id" not in raw.uns["spatial"]:
        raise ValueError("Measured h5ad needs Scanpy Visium-style uns['spatial']['library_id']['images']['hires'].")

    if args.max_spots is not None:
        raw = raw[: args.max_spots].copy()

    hires_rgb = _harvest_hires_image(raw)
    tmp_img = deepspot_root / "_tmp_ZEN38_hires_eval.png"
    from PIL import Image

    Image.fromarray(hires_rgb).save(tmp_img)

    spot_diameter = _estimate_spot_diameter_hires(raw)

    weights_dir = args.weights_dir
    model_weights = weights_dir / "final_model.pkl"
    gene_csv = weights_dir / "info_highly_variable_genes.csv"
    model_yaml = weights_dir / "top_param_overall.yaml"

    genes_manifest = pd.read_csv(gene_csv)
    pred_mask = genes_manifest["isPredicted"].values.astype(bool)
    pred_genes = genes_manifest.loc[pred_mask, "gene_name"].astype(str).tolist()

    meas_genes = set(raw.var_names.astype(str))
    intersect_genes = [g for g in pred_genes if g in meas_genes]
    if len(intersect_genes) < 100:
        raise RuntimeError(f"Too few overlapping genes ({len(intersect_genes)}); check symbol conventions.")

    measured = raw[:, intersect_genes].copy()
    sc.pp.normalize_total(measured, target_sum=1e4)
    sc.pp.log1p(measured)

    obs_tbl = _build_spot_table(raw)
    n_spots = len(obs_tbl)
    n_genes_pred = int(pred_mask.sum())
    empty = np.zeros((n_spots, n_genes_pred), dtype=np.float32)
    adata_inf = ad.AnnData(empty, obs=obs_tbl, var=pd.DataFrame(index=genes_manifest.loc[pred_mask, "gene_name"].values))
    is_white = []
    for _, row in tqdm(obs_tbl.iterrows(), total=n_spots, desc="Tissue filter (hires)"):
        r0, c0 = int(row.x_pixel) - spot_diameter // 2, int(row.y_pixel) - spot_diameter // 2
        tile = hires_rgb[
            r0 : r0 + spot_diameter,
            c0 : c0 + spot_diameter,
            :3,
        ]
        is_white.append(float(np.mean(tile)) if tile.size else 255.0)
    adata_inf.obs["is_white"] = is_white
    adata_inf.obs["is_white_bool"] = (adata_inf.obs["is_white"] > args.white_cutoff).astype(int)
    adata_inf = adata_inf[adata_inf.obs["is_white_bool"] == 0].copy()

    # Align measured to filtered spots
    measured = measured[adata_inf.obs_names].copy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_expression = torch.load(model_weights, map_location=device, weights_only=False)
    model_expression.to(device)
    model_expression.eval()

    with open(model_yaml, "r") as fh:
        hparam = yaml.safe_load(fh)
    fm_name = hparam["image_feature_model"]

    if args.foundation_timm_imagenet:
        if fm_name != "uni":
            raise ValueError("--foundation-timm-imagenet requires a uni checkpoint.")
        morphology_model, preprocess, _ = get_morphology_uni_timm_imagenet(device)
        morphology_model.to(device)
    elif args.foundation_weights is not None:
        morphology_model, preprocess, _ = get_morphology_model_and_preprocess(
            model_name=fm_name,
            device=device,
            model_path=str(args.foundation_weights),
        )
        morphology_model.to(device)
    else:
        print("Need --foundation-timm-imagenet or --foundation-weights", file=sys.stderr)
        return 1

    counts = predict_spot_spatial_transcriptomics_pil(
        tmp_img,
        adata_inf,
        spot_diameter,
        9,
        preprocess,
        morphology_model,
        model_expression,
        device,
        super_resolution=False,
        neighbor_radius=int(hparam.get("neighbors", 1)),
    )
    counts = np.asarray(counts, dtype=np.float32)
    counts[counts < 0] = 0.0

    np.savez_compressed(
        args.out_h5ad.with_suffix(".counts.npz"),
        counts=counts,
        intersect_genes=np.array(intersect_genes, dtype=object),
        obs_names=adata_inf.obs_names.values.astype(str),
    )

    pred_var_names = genes_manifest.loc[pred_mask, "gene_name"].astype(str).values
    pred_col = {g: i for i, g in enumerate(pred_var_names)}
    pred_mat = counts[:, [pred_col[g] for g in intersect_genes]]
    meas_mat = measured.X
    if hasattr(meas_mat, "toarray"):
        meas_mat = meas_mat.toarray()
    meas_mat = np.asarray(meas_mat, dtype=np.float64)

    rows = []
    for i, g in enumerate(intersect_genes):
        x, y = meas_mat[:, i], pred_mat[:, i].astype(np.float64)
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            r = np.nan
        else:
            r, _ = pearsonr(x, y)
        rows.append({"gene": g, "pearson_r": r, "n_spots": meas_mat.shape[0]})
    corr_df = pd.DataFrame(rows).sort_values("pearson_r", ascending=False, na_position="last")

    markers = [
        "MUC2",
        "ITLN1",
        "CLCA1",
        "FCGBP",
        "EPCAM",
        "KRT20",
        "CDX2",
        "CD3D",
        "CD68",
        "COL1A1",
        "VIM",
        "MS4A1",
    ]
    mk_present = _marker_table(markers, set(intersect_genes))
    marker_df = corr_df[corr_df["gene"].isin(mk_present)].copy()

    out = ad.AnnData(
        meas_mat,
        obs=measured.obs.copy(),
        var=pd.DataFrame(index=intersect_genes),
    )
    out.layers["measured_log1p"] = meas_mat.copy()
    out.layers["imputed_count"] = pred_mat.astype(np.float32)
    out.obsm["spatial"] = np.asarray(measured.obsm["spatial"])
    out.uns["deepspot_eval"] = {
        "weights_dir": str(weights_dir.resolve()),
        "measured_h5ad": str(args.measured_h5ad.resolve()),
        "foundation": "timm_imagenet_vit_l" if args.foundation_timm_imagenet else str(args.foundation_weights),
        "fm_yaml": fm_name,
        "spot_diameter_hires": spot_diameter,
        "hires_image_shape": list(hires_rgb.shape),
        "note": (
            "Measured: normalize_total(1e4)+log1p on intersect genes. "
            "Predicted: DeepSpot Colon head; timm backbone unless UNI weights used."
        ),
    }
    out.uns["marker_pearson_json"] = marker_df.to_json(orient="records")
    out.uns["pearson_median_all_genes"] = float(np.nanmedian(corr_df["pearson_r"].values))
    top500 = corr_df.head(500)["pearson_r"].dropna().values
    out.uns["pearson_mean_top500"] = float(np.mean(top500)) if len(top500) else float("nan")

    args.out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out_h5ad)

    if args.out_corr_csv:
        args.out_corr_csv.parent.mkdir(parents=True, exist_ok=True)
        corr_df.to_csv(args.out_corr_csv, index=False)
    if args.out_markers_csv:
        args.out_markers_csv.parent.mkdir(parents=True, exist_ok=True)
        marker_df.to_csv(args.out_markers_csv, index=False)

    print(out.uns["deepspot_eval"])
    print("Median Pearson (all intersect genes):", out.uns["pearson_median_all_genes"])
    print("Mean Pearson (top 500 by r):", out.uns["pearson_mean_top500"])
    print("\nKey markers:\n", marker_df.to_string(index=False))
    print("\nWrote", args.out_h5ad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
