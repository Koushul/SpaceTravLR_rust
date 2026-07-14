#!/usr/bin/env python3
"""Genius vs Silly™️: lasso coefficients from real vs structure-inferred received ligands.

Genius  — X = spatial Gaussian received ligands (ground truth)
Silly   — X = type-pooled S_hat inferred received ligands
Y       — same target-gene expression

Fits are pooled across cell types (required: type-pooled S_hat is constant
within a type, so within-type Lasso on Silly is degenerate).

Alpha is chosen by LassoCV on Genius and reused for Silly and abundance.

Outputs under results/neighborhood_atlas/:
  genius_vs_silly_detail.csv
  genius_vs_silly_summary.csv
  GENIUS_VS_SILLY.md
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.linear_model import Lasso, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler

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
    type_maps,
    type_mean_expr,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "neighborhood_atlas"


def pick_genes_by_score(X, genes: Sequence[str], n: int) -> List[str]:
    if sparse.issparse(X):
        means = np.asarray(X.mean(axis=0)).ravel()
        meansq = np.asarray(X.power(2).mean(axis=0)).ravel()
        var = np.maximum(meansq - means**2, 0.0)
    else:
        Xd = np.asarray(X, dtype=np.float64)
        means = Xd.mean(axis=0)
        var = Xd.var(axis=0)
    score = means * (1.0 + np.sqrt(var))
    order = np.argsort(-score)
    return [str(genes[i]) for i in order[:n]]


def dense_cols(X, cols: Sequence[int]) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X[:, list(cols)].toarray(), dtype=np.float64)
    return np.asarray(X[:, list(cols)], dtype=np.float64)


def load_harmonized(spec, max_cells: int, seed: int):
    a = ad.read_h5ad(spec.path)
    raw = a.obs[spec.obs_column].astype(str).to_numpy()
    missing = sorted(set(raw) - set(spec.mapping))
    if missing:
        raise ValueError(f"{spec.dataset_id}: unmapped {missing}")
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


def scale_modulators(X: np.ndarray) -> np.ndarray:
    """Match SpaceTravLR / CellOracle: divide by std, no mean subtraction."""
    scaler = StandardScaler(with_mean=False)
    return np.asarray(scaler.fit_transform(X), dtype=np.float64)


def coef_metrics(b_g: np.ndarray, b_s: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    g = np.asarray(b_g, dtype=np.float64).ravel()
    s = np.asarray(b_s, dtype=np.float64).ravel()
    sg = np.abs(g) > eps
    ss = np.abs(s) > eps
    if np.allclose(g, 0, atol=eps) and np.allclose(s, 0, atol=eps):
        pear, cos = 1.0, 1.0
    elif np.std(g) < 1e-12 or np.std(s) < 1e-12:
        pear = float("nan")
        denom = float(np.linalg.norm(g) * np.linalg.norm(s))
        cos = float(g @ s / denom) if denom > 0 else float("nan")
    else:
        pear = float(pearsonr(g, s)[0])
        denom = float(np.linalg.norm(g) * np.linalg.norm(s))
        cos = float(g @ s / denom) if denom > 0 else float("nan")
    inter = int(np.sum(sg & ss))
    union = int(np.sum(sg | ss))
    jacc = float(inter / union) if union > 0 else (1.0 if inter == 0 else float("nan"))
    both_nz = sg & ss
    sign_ag = float(np.mean(np.sign(g[both_nz]) == np.sign(s[both_nz]))) if np.any(both_nz) else float("nan")
    return {
        "coef_pearson": pear,
        "coef_cosine": cos,
        "coef_mae": float(np.mean(np.abs(g - s))),
        "coef_rmse": float(np.sqrt(np.mean((g - s) ** 2))),
        "support_jaccard": jacc,
        "sign_agreement": sign_ag,
        "n_nonzero_genius": int(sg.sum()),
        "n_nonzero_other": int(ss.sum()),
        "both_zero_frac": float(np.mean(~sg & ~ss)),
        "degenerate_both_zero": bool(np.allclose(g, 0, atol=eps) and np.allclose(s, 0, atol=eps)),
    }


def feature_col_pearson(Xg: np.ndarray, Xs: np.ndarray) -> float:
    ps = []
    for j in range(Xg.shape[1]):
        a, b = Xg[:, j], Xs[:, j]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        ps.append(pearsonr(a, b)[0])
    return float(np.nanmean(ps)) if ps else float("nan")


def choose_alpha(Xg_s: np.ndarray, y: np.ndarray, max_iter: int, seed: int) -> float:
    n = Xg_s.shape[0]
    cv = min(5, max(3, n // 50))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LassoCV(
            cv=cv,
            n_alphas=50,
            max_iter=max_iter,
            random_state=seed,
            n_jobs=1,
            fit_intercept=True,
        )
        model.fit(Xg_s, y)
    return float(model.alpha_)


def fit_lasso(X: np.ndarray, y: np.ndarray, alpha: float, max_iter: int, seed: int):
    Xs = scale_modulators(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Lasso(
            alpha=alpha,
            max_iter=max_iter,
            random_state=seed,
            fit_intercept=True,
            tol=1e-4,
        )
        m.fit(Xs, y)
    r2 = float(m.score(Xs, y))
    return np.asarray(m.coef_, dtype=np.float64), r2


def fit_ridge_pair(
    Xg: np.ndarray, Xs: np.ndarray, Xa: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Stable linear coefficients (RidgeCV) for Genius / Silly / abundance."""
    alphas = np.logspace(-3, 3, 25)
    out = []
    r2s = []
    for X in (Xg, Xs, Xa):
        Xscl = scale_modulators(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = RidgeCV(alphas=alphas, fit_intercept=True)
            m.fit(Xscl, y)
        out.append(np.asarray(m.coef_, dtype=np.float64))
        r2s.append(float(m.score(Xscl, y)))
    return out[0], out[1], out[2], r2s[0], r2s[1], r2s[2]


def evaluate_dataset(
    spec,
    *,
    max_cells: int,
    n_ligands: int,
    n_targets: int,
    seed: int,
    radius: Optional[float],
    fixed_alpha: Optional[float],
    max_iter: int,
    min_genius_r2: float,
    min_nonzero: int,
) -> List[Dict]:
    xy, labels, X, genes = load_harmonized(spec, max_cells, seed)
    labels_list = [str(x) for x in labels]
    names, codes = type_maps(labels_list)
    r = float(radius) if radius is not None else choose_radius(xy)

    ranked = pick_genes_by_score(X, genes, n_ligands + n_targets + 80)
    lig_names = ranked[:n_ligands]
    target_names = [g for g in ranked[n_ligands:] if g not in set(lig_names)][:n_targets]
    gidx = {g: i for i, g in enumerate(genes)}

    lig_expr = dense_cols(X, [gidx[g] for g in lig_names])
    genius_X = gaussian_received(xy, lig_expr, r)
    ref = build_structure_ref(xy, labels_list, r)
    assert ref["cell_types"] == names
    mu = type_mean_expr(lig_expr, codes, len(names))
    silly_X = infer_from_structure(ref["mean_weight_mass"], codes, mu)
    abund_X = infer_from_structure(
        abundance_baseline(ref["mean_weight_mass"], ref["ref_type_counts"]), codes, mu
    )

    # Drop ligands with no variation on Genius (pooled)
    keep = genius_X.var(axis=0) > 1e-12
    # Silly must also vary across types
    keep &= silly_X.var(axis=0) > 1e-12
    if int(keep.sum()) < 3:
        return []
    genius_X = genius_X[:, keep]
    silly_X = silly_X[:, keep]
    abund_X = abund_X[:, keep]
    lig_used = [lig_names[j] for j, k in enumerate(keep) if k]
    feat_r = feature_col_pearson(genius_X, silly_X)
    feat_r_ab = feature_col_pearson(genius_X, abund_X)

    rows: List[Dict] = []
    for ti, tname in enumerate(target_names):
        y = dense_cols(X, [gidx[tname]]).ravel()
        if float(np.std(y)) < 1e-12:
            continue

        Xg_s = scale_modulators(genius_X)
        if fixed_alpha is None:
            alpha = choose_alpha(Xg_s, y, max_iter=max_iter, seed=seed + ti)
        else:
            alpha = float(fixed_alpha)

        bg, r2g = fit_lasso(genius_X, y, alpha, max_iter, seed + ti)
        bs, r2s = fit_lasso(silly_X, y, alpha, max_iter, seed + ti)
        ba, r2a = fit_lasso(abund_X, y, alpha, max_iter, seed + ti)
        rg, rs, ra, r2rg, r2rs, r2ra = fit_ridge_pair(genius_X, silly_X, abund_X, y)

        m_s = coef_metrics(bg, bs)
        m_a = coef_metrics(bg, ba)
        rm_s = coef_metrics(rg, rs)
        rm_a = coef_metrics(rg, ra)

        # Skip fully degenerate Genius Lasso models (no signal under this alpha)
        if r2g < min_genius_r2 or m_s["n_nonzero_genius"] < min_nonzero:
            status = "skipped_weak_genius"
        else:
            status = "ok"

        rows.append(
            {
                "dataset_id": spec.dataset_id,
                "technology": spec.technology,
                "organ": spec.organ,
                "species": spec.species,
                "target_gene": tname,
                "status": status,
                "n_cells": int(xy.shape[0]),
                "n_types": len(names),
                "n_ligands_used": len(lig_used),
                "radius": r,
                "alpha": alpha,
                "r2_genius": r2g,
                "r2_silly": r2s,
                "r2_abundance": r2a,
                "ridge_r2_genius": r2rg,
                "ridge_r2_silly": r2rs,
                "ridge_r2_abundance": r2ra,
                "feature_col_pearson_silly": feat_r,
                "feature_col_pearson_abund": feat_r_ab,
                **{f"silly_{k}": v for k, v in m_s.items()},
                **{f"abund_{k}": v for k, v in m_a.items()},
                **{f"ridge_silly_{k}": v for k, v in rm_s.items()},
                **{f"ridge_abund_{k}": v for k, v in rm_a.items()},
            }
        )
    return rows


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for did, g_all in detail.groupby("dataset_id", sort=True):
        g = g_all[g_all["status"] == "ok"]
        meta = g_all.iloc[0]
        if g.empty:
            rows.append(
                {
                    "dataset_id": did,
                    "technology": meta["technology"],
                    "organ": meta["organ"],
                    "species": meta["species"],
                    "n_targets_total": int(len(g_all)),
                    "n_targets_ok": 0,
                    "silly_coef_pearson_median": float("nan"),
                    "silly_coef_cosine_median": float("nan"),
                    "abund_coef_pearson_median": float("nan"),
                    "pearson_lift_vs_abund": float("nan"),
                    "feature_col_pearson_silly": float(meta["feature_col_pearson_silly"]),
                    "ridge_silly_coef_pearson_median": float(
                        g_all["ridge_silly_coef_pearson"].median()
                    ),
                    "ridge_silly_coef_cosine_median": float(
                        g_all["ridge_silly_coef_cosine"].median()
                    ),
                    "ridge_abund_coef_pearson_median": float(
                        g_all["ridge_abund_coef_pearson"].median()
                    ),
                    "ridge_pearson_lift_vs_abund": float(
                        g_all["ridge_silly_coef_pearson"].median()
                        - g_all["ridge_abund_coef_pearson"].median()
                    ),
                    "ridge_r2_genius_median": float(g_all["ridge_r2_genius"].median()),
                    "ridge_r2_silly_median": float(g_all["ridge_r2_silly"].median()),
                }
            )
            continue
        rows.append(
            {
                "dataset_id": did,
                "technology": meta["technology"],
                "organ": meta["organ"],
                "species": meta["species"],
                "n_targets_total": int(len(g_all)),
                "n_targets_ok": int(len(g)),
                "n_cells": int(meta["n_cells"]),
                "n_ligands_used": int(meta["n_ligands_used"]),
                "median_alpha": float(g["alpha"].median()),
                "r2_genius_median": float(g["r2_genius"].median()),
                "r2_silly_median": float(g["r2_silly"].median()),
                "r2_abundance_median": float(g["r2_abundance"].median()),
                "feature_col_pearson_silly": float(g["feature_col_pearson_silly"].median()),
                "feature_col_pearson_abund": float(g["feature_col_pearson_abund"].median()),
                "silly_coef_pearson_median": float(g["silly_coef_pearson"].median()),
                "silly_coef_pearson_mean": float(g["silly_coef_pearson"].mean()),
                "silly_coef_cosine_median": float(g["silly_coef_cosine"].median()),
                "silly_coef_mae_median": float(g["silly_coef_mae"].median()),
                "silly_support_jaccard_median": float(g["silly_support_jaccard"].median()),
                "silly_sign_agreement_median": float(g["silly_sign_agreement"].median()),
                "abund_coef_pearson_median": float(g["abund_coef_pearson"].median()),
                "abund_coef_cosine_median": float(g["abund_coef_cosine"].median()),
                "abund_support_jaccard_median": float(g["abund_support_jaccard"].median()),
                "pearson_lift_vs_abund": float(
                    g["silly_coef_pearson"].median() - g["abund_coef_pearson"].median()
                ),
                "frac_degenerate_silly": float(g["silly_degenerate_both_zero"].mean()),
                # Ridge (stable) coefficient closeness — all targets, not only Lasso-ok
                "ridge_silly_coef_pearson_median": float(
                    g_all["ridge_silly_coef_pearson"].median()
                ),
                "ridge_silly_coef_cosine_median": float(
                    g_all["ridge_silly_coef_cosine"].median()
                ),
                "ridge_abund_coef_pearson_median": float(
                    g_all["ridge_abund_coef_pearson"].median()
                ),
                "ridge_pearson_lift_vs_abund": float(
                    g_all["ridge_silly_coef_pearson"].median()
                    - g_all["ridge_abund_coef_pearson"].median()
                ),
                "ridge_r2_genius_median": float(g_all["ridge_r2_genius"].median()),
                "ridge_r2_silly_median": float(g_all["ridge_r2_silly"].median()),
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame) -> None:
    lines = [
        "# Genius vs Silly — lasso coefficients from real vs inferred received ligands",
        "",
        "```text",
        "Genius X  = spatial Gaussian received ligands",
        "Silly  X  = structure-inferred received ligands (S_hat @ mu_query)",
        "Y         = target gene expression (same)",
        "Fit       = pooled-cell Lasso, column std scaling (no mean subtraction)",
        "Alpha     = LassoCV on Genius, reused for Silly and abundance",
        "```",
        "",
        "Fits are pooled across cell types because type-pooled `S_hat` is constant",
        "within a type (within-type Silly Lasso is degenerate).",
        "",
        "## Summary (median over target genes with non-trivial Genius fits)",
        "",
        "### Lasso (sparse; non-trivial Genius fits only)",
        "",
        "| dataset | tech | organ | n_ok | coef r | cosine | Jaccard | abund r | lift | feat r |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.sort_values(["technology", "organ", "dataset_id"]).iterrows():
        lines.append(
            f"| `{r['dataset_id']}` | {r['technology']} | {r['organ']} | "
            f"{int(r.get('n_targets_ok', 0))}/{int(r.get('n_targets_total', 0))} | "
            f"{r['silly_coef_pearson_median']:.3f} | {r['silly_coef_cosine_median']:.3f} | "
            f"{r.get('silly_support_jaccard_median', float('nan')):.3f} | "
            f"{r['abund_coef_pearson_median']:.3f} | {r['pearson_lift_vs_abund']:+.3f} | "
            f"{r['feature_col_pearson_silly']:.3f} |"
        )
    lines += [
        "",
        "### Ridge (stable linear coefficients; all targets)",
        "",
        "| dataset | tech | organ | ridge coef r | ridge cosine | abund ridge r | lift | ridge R2 genius | ridge R2 silly |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.sort_values(["technology", "organ", "dataset_id"]).iterrows():
        lines.append(
            f"| `{r['dataset_id']}` | {r['technology']} | {r['organ']} | "
            f"{r['ridge_silly_coef_pearson_median']:.3f} | "
            f"{r['ridge_silly_coef_cosine_median']:.3f} | "
            f"{r['ridge_abund_coef_pearson_median']:.3f} | "
            f"{r['ridge_pearson_lift_vs_abund']:+.3f} | "
            f"{r['ridge_r2_genius_median']:.3f} | {r['ridge_r2_silly_median']:.3f} |"
        )
    lines += [
        "",
        "Lasso betas are brittle when Silly X is piecewise-constant by type (low rank).",
        "Ridge answers the same question with a stable linear map.",
        "Abundance is the composition-only negative control.",
        "",
        "Files: `genius_vs_silly_summary.csv`, `genius_vs_silly_detail.csv`.",
        "",
    ]
    (OUT / "GENIUS_VS_SILLY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-cells", type=int, default=8000)
    ap.add_argument("--n-ligands", type=int, default=30)
    ap.add_argument("--n-targets", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--max-iter", type=int, default=8000)
    ap.add_argument("--min-genius-r2", type=float, default=0.02)
    ap.add_argument("--min-nonzero", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--use-atlas-radius", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    specs = {s.dataset_id: s for s in DATASETS}
    ids = [e["dataset_id"] for e in list_entries()]
    if args.only:
        ids = [i for i in ids if i in set(args.only)]

    all_rows: List[Dict] = []
    for did in ids:
        spec = specs[did]
        print(f"[genius-vs-silly] {did} ...", flush=True)
        radius = float(get_entry(did)["radius"]) if args.use_atlas_radius else None
        rows = evaluate_dataset(
            spec,
            max_cells=args.max_cells,
            n_ligands=args.n_ligands,
            n_targets=args.n_targets,
            seed=args.seed,
            radius=radius,
            fixed_alpha=args.alpha,
            max_iter=args.max_iter,
            min_genius_r2=args.min_genius_r2,
            min_nonzero=args.min_nonzero,
        )
        all_rows.extend(rows)
        d = pd.DataFrame(rows)
        ok = d[d["status"] == "ok"] if len(d) else d
        if len(ok):
            print(
                f"  ok={len(ok)}/{len(d)}  silly_r_med={ok['silly_coef_pearson'].median():.3f}  "
                f"abund_r_med={ok['abund_coef_pearson'].median():.3f}  "
                f"feat_r={ok['feature_col_pearson_silly'].median():.3f}  "
                f"r2g_med={ok['r2_genius'].median():.3f}",
                flush=True,
            )
        else:
            print(f"  no non-trivial fits ({len(d)} attempted)", flush=True)

    detail = pd.DataFrame(all_rows)
    if detail.empty:
        raise SystemExit("No fits produced")
    summary = summarize(detail)
    detail.to_csv(OUT / "genius_vs_silly_detail.csv", index=False)
    summary.to_csv(OUT / "genius_vs_silly_summary.csv", index=False)
    write_report(summary)
    (OUT / "genius_vs_silly_meta.json").write_text(
        json.dumps(
            {
                "max_cells": args.max_cells,
                "n_ligands": args.n_ligands,
                "n_targets": args.n_targets,
                "alpha": args.alpha,
                "min_genius_r2": args.min_genius_r2,
                "min_nonzero": args.min_nonzero,
                "use_atlas_radius": args.use_atlas_radius,
                "seed": args.seed,
                "note": "Pooled-cell Lasso; within-type Silly is constant under type-pooled S_hat",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"[genius-vs-silly] wrote {OUT}")
    cols = [
        "dataset_id",
        "n_targets_ok",
        "silly_coef_pearson_median",
        "abund_coef_pearson_median",
        "ridge_silly_coef_pearson_median",
        "ridge_abund_coef_pearson_median",
        "ridge_pearson_lift_vs_abund",
        "feature_col_pearson_silly",
    ]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
