#!/usr/bin/env python3
"""
Compare perturbation vector fields on UMAP using the alignment score from
SpaceTravLR's VirtualTissue.compute_vector_alignment (same construction as
embeds2perturb.ipynb: CellRank-style transition probabilities + quiver field +
cosine alignment to pseudotime-gradient reference).

Requires: anndata, scanpy, numpy, scipy, pandas, matplotlib, scikit-learn,
velocyto, cellrank (same stack as SpaceTravLR plotting).

Set SPACE_TRAVLR_SRC to the `src` directory of the SpaceTravLR Python package
(default tries common paths next to this repo).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu


def _ensure_spacetravlr_path(extra: Sequence[str] | None) -> None:
    candidates = list(extra or [])
    env = os.environ.get("SPACE_TRAVLR_SRC", "").strip()
    if env:
        candidates.insert(0, env)
    here = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            str(here.parent / "SpaceTravLR" / "src"),
            "/ihome/ylee/kor11/tools/SpaceTravLR/src",
            "/ix1/ylee/kor11/tools/SpaceTravLR/src",
        ]
    )
    for p in candidates:
        if p and Path(p).is_dir() and (Path(p) / "SpaceTravLR").is_dir():
            if p not in sys.path:
                sys.path.insert(0, p)
            return
    raise RuntimeError(
        "Could not find SpaceTravLR Python package. Clone SpaceTravLR or set SPACE_TRAVLR_SRC "
        "to its `src` directory (contains SpaceTravLR/)."
    )


def _read_perturb_feather(path: Path, cell_ids: Iterable[str]) -> pd.DataFrame:
    df = pd.read_feather(path)
    if "CellID" not in df.columns:
        raise ValueError(f"{path}: expected column CellID")
    df = df.set_index("CellID")
    index = pd.Index(list(cell_ids))
    missing = index.difference(df.index)
    if len(missing) > 0:
        raise ValueError(f"{path}: missing {len(missing)} cells vs AnnData order")
    return df.reindex(index)


def _random_expression_like(baseline: pd.DataFrame, rng: np.random.Generator, lo=-55.0, hi=55.0):
    return pd.DataFrame(
        rng.uniform(lo, hi, size=baseline.shape),
        index=baseline.index,
        columns=baseline.columns,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", type=Path, required=True, help="Full h5ad (must match training cells)")
    ap.add_argument(
        "--spacetravlr-src",
        type=str,
        default="",
        help="Path to SpaceTravLR `src` (parent of SpaceTravLR/). Overrides SPACE_TRAVLR_SRC.",
    )
    ap.add_argument(
        "--annot",
        type=str,
        default="cell_type",
        help="obs column for cell types (Cartography / plot_arrows default uses cell_type too)",
    )
    ap.add_argument(
        "--alignment-annot",
        type=str,
        default="",
        help="obs column passed to compute_vector_alignment as `annot` (default: same as --annot)",
    )
    ap.add_argument(
        "--restrict-to",
        type=str,
        default="Naive CD4 T,T_follicular_helper,Th2,Th1",
        help="Comma-separated cell types for init_cartography restrict_to (empty = no restriction)",
    )
    ap.add_argument("--pseudotime-key", type=str, default="pseudotime", help="obs column for smoothing / ref flow")
    ap.add_argument("--layer", type=str, default="imputed_count", help="Expression layer in adata.layers")
    ap.add_argument("--umap-key", type=str, default="X_umap", help="obsm key for UMAP")
    ap.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV with columns: label,feather_path (feather = spacetravlr-perturb output, CellID + genes)",
    )
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=None, help="Optional full dump of per-cell alignment tables")
    ap.add_argument("--n-neighbors-quiver", type=int, default=200)
    ap.add_argument("--quiver-scale", type=float, default=4.0)
    ap.add_argument("--alignment-k-smooth", type=int, default=300)
    ap.add_argument("--random-seed", type=int, default=1)
    ap.add_argument("--random-low", type=float, default=-55.0)
    ap.add_argument("--random-high", type=float, default=55.0)
    args = ap.parse_args()

    extra_src = [args.spacetravlr_src] if args.spacetravlr_src.strip() else []
    _ensure_spacetravlr_path(extra_src)

    from SpaceTravLR.virtual_tissue import VirtualTissue

    adata = sc.read_h5ad(args.adata, backed="r")
    if args.layer not in adata.layers:
        raise SystemExit(f"Layer {args.layer!r} not in adata.layers")
    if args.umap_key not in adata.obsm:
        raise SystemExit(f"obsm {args.umap_key!r} missing")
    if args.annot not in adata.obs:
        raise SystemExit(f"obs {args.annot!r} missing")
    if args.pseudotime_key not in adata.obs:
        raise SystemExit(
            f"obs {args.pseudotime_key!r} missing; add pseudotime (e.g. DPT) or pass --pseudotime-key"
        )

    alignment_annot = args.alignment_annot.strip() or args.annot
    if alignment_annot not in adata.obs:
        raise SystemExit(f"obs {alignment_annot!r} missing")

    restrict = [x.strip() for x in args.restrict_to.split(",") if x.strip()]

    adata_full = adata.to_memory() if adata.isbacked else adata
    tonsil = VirtualTissue(adata_full, annot=args.annot, n_props=4)
    tonsil.init_cartography(adata_full, restrict_to=restrict or None)

    baseline = adata_full.to_df(layer=args.layer)
    rng = np.random.default_rng(args.random_seed)
    rand_df = _random_expression_like(baseline, rng, args.random_low, args.random_high)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_params = {
        "n_neighbors": args.n_neighbors_quiver,
        "remove_null": True,
        "scale": args.quiver_scale,
        "threshold": 0,
        "grey_out": False,
    }

    _, vector_field_rand = tonsil.plot_arrows(perturbed_df=rand_df, **plot_params)
    plt.close("all")

    manifest = pd.read_csv(args.manifest)
    for col in ("label", "feather_path"):
        if col not in manifest.columns:
            raise SystemExit(f"manifest CSV must include column {col!r}")

    rows_out = []
    json_blob: dict[str, object] = {}

    cell_order = list(adata_full.obs_names.astype(str))
    for _, rec in manifest.iterrows():
        label = str(rec["label"])
        fpath = Path(str(rec["feather_path"]))
        perturbed = _read_perturb_feather(fpath, cell_order)
        grid_points, vector_field = tonsil.plot_arrows(perturbed_df=perturbed, **plot_params)
        plt.close("all")

        alignment_df, df, alignment_df_rand, df_rand = tonsil.compute_vector_alignment(
            grid_points,
            vector_field,
            vector_field_rand,
            annot=alignment_annot,
            obs_key=args.pseudotime_key,
            k=args.alignment_k_smooth,
        )

        for ct in df[alignment_annot].unique():
            x = df.loc[df[alignment_annot] == ct, "alignment"].astype(float)
            y = df_rand.loc[df_rand[alignment_annot] == ct, "alignment"].astype(float)
            if len(x) < 3 or len(y) < 3:
                pval = float("nan")
                stat = float("nan")
            else:
                stat, pval = mannwhitneyu(x, y, alternative="two-sided")
            rows_out.append(
                {
                    "label": label,
                    "feather_path": str(fpath),
                    "cell_type": ct,
                    "mean_alignment": float(x.mean()),
                    "mean_alignment_rand": float(y.mean()),
                    "delta_mean_vs_rand": float(x.mean() - y.mean()),
                    "mannwhitney_p_two_sided": float(pval),
                    "mannwhitney_statistic": float(stat),
                }
            )

        json_blob[label] = {
            "alignment_per_cell_type": alignment_df.astype(float).to_dict(),
            "grid_shape": list(grid_points.shape),
        }

    out_df = pd.DataFrame(rows_out)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(json_blob, f, indent=2)


if __name__ == "__main__":
    main()
