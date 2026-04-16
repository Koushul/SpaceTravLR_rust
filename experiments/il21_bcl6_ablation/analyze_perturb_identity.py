#!/usr/bin/env python3
"""Check whether perturb outputs are identical across runs (sanity for ablation interpretability)."""
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parent / "runs"
FEATHER = "perturb_feathers/IL21_KO_perturb_expr.feather"


def main() -> None:
    paths = sorted(RUNS.glob(f"*/{FEATHER}"))
    assert paths, "no perturb feathers"
    ref = pd.read_feather(paths[0]).set_index("CellID").sort_index()
    mat_ref = ref.to_numpy(dtype=np.float64)
    rows = []
    for p in paths:
        d = pd.read_feather(p).set_index("CellID").sort_index()
        m = d.to_numpy(dtype=np.float64)
        md = float(np.max(np.abs(m - mat_ref)))
        rows.append({"run": p.parent.parent.name, "max_abs_diff_vs_first": md})
    out = pd.DataFrame(rows).sort_values("max_abs_diff_vs_first", ascending=False)
    print(out.to_string(index=False))
    print("\nIf all zeros, every run produced identical IL21 KO perturb vectors.")


if __name__ == "__main__":
    main()
