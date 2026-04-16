#!/usr/bin/env python3
"""L2 norm of (perturbed - baseline) across all genes per feather; cheap global summary."""
from pathlib import Path

import argparse
import numpy as np
import pandas as pd
import scanpy as sc


def l2_delta(feather: Path, baseline: np.ndarray, gene_cols: list[str]) -> float:
    df = pd.read_feather(feather)
    if "CellID" not in df.columns:
        raise ValueError(feather)
    df = df.set_index("CellID")
    x = df[gene_cols].to_numpy(dtype=np.float64)
    d = x - baseline
    return float(np.linalg.norm(d.ravel()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--h5ad", type=str, default="/ix/djishnu/shared/djishnu_kor11/training_data_revision/snrna_human_tonsil.h5ad")
    ap.add_argument("--layer", type=str, default="imputed_count")
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    ad = sc.read_h5ad(args.h5ad, backed="r")
    if args.layer not in ad.layers:
        raise SystemExit(f"missing layer {args.layer}")
    lay = ad.layers[args.layer]
    X = lay.toarray() if hasattr(lay, "toarray") else np.asarray(lay)
    genes = list(ad.var_names.astype(str))
    cell_ids = list(ad.obs_names.astype(str))
    baseline = np.asarray(X, dtype=np.float64)

    rows = []
    for run_dir in sorted(p for p in args.runs_dir.iterdir() if p.is_dir()):
        pf = run_dir / "perturb_feathers"
        if not pf.is_dir():
            continue
        for feather in sorted(pf.glob("*.feather")):
            try:
                v = l2_delta(feather, baseline, genes)
            except Exception as e:
                v = float("nan")
                err = str(e)
            else:
                err = ""
            rows.append(
                {
                    "run": run_dir.name,
                    "feather": feather.name,
                    "l2_delta_all_genes": v,
                    "error": err,
                }
            )

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(out.groupby("run")["l2_delta_all_genes"].mean().sort_values(ascending=False).head(15))


if __name__ == "__main__":
    main()
