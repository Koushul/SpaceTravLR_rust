#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import scanpy as sc

RUNS = Path(__file__).resolve().parent / "runs"
GENES = ["IL21", "BCL6"]


def main() -> None:
    rows = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        proc = d / "snrna_human_tonsil_processed.h5ad"
        if not proc.is_file():
            continue
        a = sc.read_h5ad(proc, backed="r")
        if "mean_lasso_r2" not in a.var.columns:
            continue
        r2 = a.var["mean_lasso_r2"]
        for g in GENES:
            if g not in a.var_names:
                continue
            rows.append({"run": d.name, "gene": g, "mean_lasso_r2": float(r2[g])})
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parent / "analysis" / "training_mean_lasso_r2.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.pivot(index="run", columns="gene", values="mean_lasso_r2"))


if __name__ == "__main__":
    main()
