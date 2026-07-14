#!/usr/bin/env python3
"""Compare ground-truth vs structure-inferred received ligands for atlas datasets.

For each SlideSeqV2 / VisiumHD / SlideTags atlas entry:

  truth[i,l]     = (1/N) Σ_j exp(-d(i,j)² / 2r²) · expr[j,l]     # spatial GT
  structure[i,l] = Σ_t Ŝ[type(i), t] · μ[t, l]                   # type-pooled Ŝ
  abundance[i,l] = Σ_t Â[type(i), t] · μ[t, l]                   # composition-only baseline
  type_mean_oracle[i,l] = Σ_t S[i, t] · μ[t, l]                  # cell niche × type means

Metrics are reported at cell level and after averaging to receiver type
(the natural non-spatial estimand). Results written under
results/neighborhood_atlas/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_neighborhood_atlas import (  # noqa: E402
    DATASETS,
    choose_radius,
    load_xy,
    stratified_subsample,
)
from load_neighborhood_atlas import get_entry, list_entries  # noqa: E402
from validate_structure_ligands import (  # noqa: E402
    abundance_baseline,
    build_structure_ref,
    gaussian_received,
    infer_from_structure,
    matrix_metrics,
    score,
    type_level_metrics,
    type_maps,
    type_mean_expr,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "neighborhood_atlas"


def pick_ligand_genes(X, genes: Sequence[str], n: int) -> List[str]:
    if sparse.issparse(X):
        means = np.asarray(X.mean(axis=0)).ravel()
        meansq = np.asarray(X.power(2).mean(axis=0)).ravel()
        var = np.maximum(meansq - means**2, 0.0)
    else:
        Xd = np.asarray(X, dtype=np.float64)
        means = Xd.mean(axis=0)
        var = Xd.var(axis=0)
    score_v = means * (1.0 + np.sqrt(var))
    order = np.argsort(-score_v)
    return [str(genes[i]) for i in order[:n]]


def load_harmonized(spec, max_cells: int, seed: int):
    a = ad.read_h5ad(spec.path)
    raw = a.obs[spec.obs_column].astype(str).to_numpy()
    missing = sorted(set(raw) - set(spec.mapping))
    if missing:
        raise ValueError(f"{spec.dataset_id}: unmapped labels {missing}")
    harm = np.array([spec.mapping[x] for x in raw], dtype=object)
    xy = load_xy(a)
    X = a.X
    genes = np.asarray(a.var_names.astype(str))
    sel = stratified_subsample(harm, max_cells, seed=seed)
    xy = xy[sel]
    harm = harm[sel]
    if sparse.issparse(X):
        X = X.tocsr()[sel]
    else:
        X = np.asarray(X, dtype=np.float64)[sel]
    return xy, harm, X, genes


def align_atlas_S(atlas_types: Sequence[str], atlas_S: np.ndarray, eval_types: Sequence[str]):
    """Reorder atlas Ŝ to evaluation type order; fail if type sets differ."""
    if sorted(atlas_types) != sorted(eval_types):
        raise ValueError(
            f"atlas types {sorted(atlas_types)} != eval types {sorted(eval_types)}"
        )
    idx = [atlas_types.index(t) for t in eval_types]
    return atlas_S[np.ix_(idx, idx)]


def evaluate_one(
    spec,
    *,
    max_cells: int,
    n_ligands: int,
    seed: int,
    radius: Optional[float],
) -> Dict:
    t0 = time.time()
    xy, labels, X, genes = load_harmonized(spec, max_cells, seed)
    labels_list = [str(x) for x in labels]
    names, codes = type_maps(labels_list)
    r = float(radius) if radius is not None else choose_radius(xy)

    lig_names = pick_ligand_genes(X, genes, n_ligands)
    gidx = {g: i for i, g in enumerate(genes)}
    cols = [gidx[g] for g in lig_names]
    if sparse.issparse(X):
        expr = np.asarray(X[:, cols].toarray(), dtype=np.float64)
    else:
        expr = np.asarray(X[:, cols], dtype=np.float64)

    truth = gaussian_received(xy, expr, r)
    ref = build_structure_ref(xy, labels_list, r)
    # Ensure type order matches names/codes
    assert ref["cell_types"] == names
    mu = type_mean_expr(expr, codes, len(names))

    cell_S = ref["_cell_weight"]
    oracle = np.einsum("it,tl->il", cell_S, mu)
    pooled = infer_from_structure(ref["mean_weight_mass"], codes, mu)
    abund = infer_from_structure(
        abundance_baseline(ref["mean_weight_mass"], ref["ref_type_counts"]), codes, mu
    )

    # Atlas-stored Ŝ (may differ slightly if radius/subsample differ)
    atlas = get_entry(spec.dataset_id)
    atlas_S = align_atlas_S(atlas["cell_types"], atlas["mean_weight_mass"], names)
    atlas_pred = infer_from_structure(atlas_S, codes, mu)

    soft_truth = ref["_cell_soft"]
    soft_pooled = ref["mean_soft_counts"][codes]

    rows = []
    for method, pred in [
        ("type_mean_oracle", oracle),
        ("structure_pooled", pooled),
        ("structure_atlas", atlas_pred),
        ("abundance_baseline", abund),
    ]:
        m = score(pred, truth, soft_pred=soft_pooled if method.startswith("structure") else None,
                  soft_truth=soft_truth if method.startswith("structure") else None)
        tl = type_level_metrics(pred, truth, codes)
        # Primary difference metrics vs GT
        mae, rmse, rel = matrix_metrics(pred, truth)
        delta = pred - truth
        rows.append(
            {
                "dataset_id": spec.dataset_id,
                "technology": spec.technology,
                "organ": spec.organ,
                "species": spec.species,
                "method": method,
                "n_cells": int(xy.shape[0]),
                "n_types": len(names),
                "n_ligands": int(expr.shape[1]),
                "radius": r,
                "cell_pearson": m.pearson_mean,
                "cell_spearman": m.spearman_mean,
                "cell_mae": mae,
                "cell_rmse": rmse,
                "cell_rel_mae": rel,
                "cell_mean_signed_error": float(delta.mean()),
                "type_pearson": tl["pearson_mean"],
                "type_mae": tl["mae"],
                "type_rmse": tl.get("rmse", np.nan),
                "type_rel_mae": tl.get("rel_mae", np.nan),
                "soft_neighbor_cosine": m.soft_cosine,
                "calib_slope": m.slope,
                "elapsed_sec": round(time.time() - t0, 2),
            }
        )
    return {
        "dataset_id": spec.dataset_id,
        "rows": rows,
        "cell_types": names,
        "ligands": lig_names,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Wide summary: structure vs GT difference per dataset."""
    piv = df[df["method"].isin(["structure_pooled", "abundance_baseline", "type_mean_oracle", "structure_atlas"])]
    out_rows = []
    for did, g in piv.groupby("dataset_id", sort=True):
        meta = g.iloc[0]
        by = {r["method"]: r for _, r in g.iterrows()}
        sp = by["structure_pooled"]
        ab = by["abundance_baseline"]
        orc = by["type_mean_oracle"]
        atl = by["structure_atlas"]
        out_rows.append(
            {
                "dataset_id": did,
                "technology": meta["technology"],
                "organ": meta["organ"],
                "species": meta["species"],
                "n_cells": int(meta["n_cells"]),
                "n_types": int(meta["n_types"]),
                "n_ligands": int(meta["n_ligands"]),
                "radius": float(meta["radius"]),
                # Primary: how far structure is from spatial GT
                "gt_vs_structure_type_pearson": float(sp["type_pearson"]),
                "gt_vs_structure_type_mae": float(sp["type_mae"]),
                "gt_vs_structure_cell_pearson": float(sp["cell_pearson"]),
                "gt_vs_structure_cell_mae": float(sp["cell_mae"]),
                "gt_vs_structure_cell_rel_mae": float(sp["cell_rel_mae"]),
                "gt_vs_structure_soft_cosine": float(sp["soft_neighbor_cosine"]),
                # Atlas disk Ŝ fidelity to same-sample GT
                "gt_vs_atlas_type_pearson": float(atl["type_pearson"]),
                "gt_vs_atlas_type_mae": float(atl["type_mae"]),
                # Baselines
                "gt_vs_oracle_type_pearson": float(orc["type_pearson"]),
                "gt_vs_abundance_type_pearson": float(ab["type_pearson"]),
                "structure_minus_abundance_type_pearson": float(
                    sp["type_pearson"] - ab["type_pearson"]
                ),
            }
        )
    return pd.DataFrame(out_rows)


def write_report(summary: pd.DataFrame, detail: pd.DataFrame) -> None:
    lines = [
        "# Ground-truth vs structure-inferred received ligands",
        "",
        "For each atlas dataset, spatial Gaussian received ligands (ground truth)",
        "are compared to predictions from the type-pooled neighborhood grammar `Ŝ`.",
        "",
        "```",
        "truth[i,l]     = (1/N) Σ_j exp(-d²/2r²) · expr[j,l]",
        "structure[i,l] = Σ_t Ŝ[type(i), t] · μ[t, l]",
        "```",
        "",
        "Primary metric: **type-level Pearson** (receiver-type means of pred vs truth).",
        "Cell-level Pearson is expected to be lower because `Ŝ` cannot recover",
        "within-type niche heterogeneity without coordinates.",
        "",
        "## Summary (structure_pooled vs GT)",
        "",
        "| dataset | tech | organ | type r | type MAE | cell r | cell MAE | soft cos | vs abund Δr |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.sort_values(["technology", "organ", "dataset_id"]).iterrows():
        lines.append(
            f"| `{r['dataset_id']}` | {r['technology']} | {r['organ']} | "
            f"{r['gt_vs_structure_type_pearson']:.3f} | {r['gt_vs_structure_type_mae']:.4g} | "
            f"{r['gt_vs_structure_cell_pearson']:.3f} | {r['gt_vs_structure_cell_mae']:.4g} | "
            f"{r['gt_vs_structure_soft_cosine']:.3f} | "
            f"{r['structure_minus_abundance_type_pearson']:+.3f} |"
        )
    lines += [
        "",
        "Files: `gt_vs_structure_summary.csv`, `gt_vs_structure_detail.csv`.",
        "",
    ]
    (OUT / "GT_VS_STRUCTURE.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-cells", type=int, default=12000)
    ap.add_argument("--n-ligands", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument(
        "--use-atlas-radius",
        action="store_true",
        help="Use radius stored in atlas entry instead of recomputing",
    )
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    specs = {s.dataset_id: s for s in DATASETS}
    ids = [e["dataset_id"] for e in list_entries()]
    if args.only:
        ids = [i for i in ids if i in set(args.only)]

    all_rows: List[Dict] = []
    meta = []
    for did in ids:
        if did not in specs:
            raise SystemExit(f"No DatasetSpec for atlas entry {did}")
        spec = specs[did]
        print(f"[gt-vs-structure] {did} ...", flush=True)
        radius = None
        if args.use_atlas_radius:
            radius = float(get_entry(did)["radius"])
        result = evaluate_one(
            spec,
            max_cells=args.max_cells,
            n_ligands=args.n_ligands,
            seed=args.seed,
            radius=radius,
        )
        all_rows.extend(result["rows"])
        meta.append(
            {
                "dataset_id": did,
                "cell_types": result["cell_types"],
                "ligands": result["ligands"],
            }
        )
        sp = next(r for r in result["rows"] if r["method"] == "structure_pooled")
        print(
            f"  type_pearson={sp['type_pearson']:.3f} cell_pearson={sp['cell_pearson']:.3f} "
            f"type_mae={sp['type_mae']:.4g} ({sp['elapsed_sec']}s)",
            flush=True,
        )

    detail = pd.DataFrame(all_rows)
    summary = summarize(detail)
    detail.to_csv(OUT / "gt_vs_structure_detail.csv", index=False)
    summary.to_csv(OUT / "gt_vs_structure_summary.csv", index=False)
    (OUT / "gt_vs_structure_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_report(summary, detail)
    print(f"[gt-vs-structure] wrote {OUT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
