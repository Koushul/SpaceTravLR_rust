#!/usr/bin/env python3
"""Score existing perturbation sweep dirs on spatial + CCC metrics."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _py_boot import ensure_boot

ensure_boot()

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

_spec22 = importlib.util.spec_from_file_location("s22", HERE / "22_spatial_ccc_tune.py")
_s22 = importlib.util.module_from_spec(_spec22)
_spec22.loader.exec_module(_s22)

_spec09 = importlib.util.spec_from_file_location("s09", HERE / "09_spatial_validation.py")
_s09 = importlib.util.module_from_spec(_spec09)
_spec09.loader.exec_module(_s09)

VAL_GENES = _s22.VAL_GENES


def main() -> None:
    sweep_root = ROOT / "results/iteration_sweep"
    baseline_h5ad = ROOT / "runs/baseline_pooled_seed/spacetravlr_prep/baseline_ntc_0c6fbac5e6cd947c_fullprep.h5ad"
    data_root = ROOT / "data"
    baseline = _s09.load_baseline(baseline_h5ad)

    rows = []
    for d in sorted(sweep_root.glob("beta*_np*")):
        if not (d / f"predicted_KO_{VAL_GENES[0]}.feather").exists():
            continue
        sp = _s22.score_spatial(d, baseline, data_root)
        cc = _s22.score_ccc(d, baseline, data_root)
        m = {**sp, **cc, "pred_dir": str(d)}
        parts = d.name.replace("beta", "").split("_np")
        m["beta_scale_factor"] = float(parts[0])
        m["n_propagation"] = int(parts[1])
        rows.append(m)
        print(f"{d.name}: spatial={m.get('spatial_median_r', float('nan')):+.3f}  ccc={m.get('ccc_pearson_r', float('nan')):+.3f}")

    df = pd.DataFrame(rows)
    df["composite"] = df.apply(
        lambda r: _s22.composite_score({"spatial_median_r": r.spatial_median_r, "ccc_pearson_r": r.ccc_pearson_r}),
        axis=1,
    )
    out = ROOT / "results/spatial_ccc_tune"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "existing_sweep_spatial_ccc.csv", index=False)
    best = df.loc[df.composite.idxmax()].to_dict()
    (out / "best_existing_spatial_ccc.json").write_text(json.dumps(best, indent=2, default=str))
    print("\nBEST:", json.dumps(best, indent=2, default=str))

    tuned = ROOT / "results/predictions_spatial_tuned"
    tuned.mkdir(parents=True, exist_ok=True)
    src = Path(best["pred_dir"])
    for g in VAL_GENES:
        shutil.copy2(src / f"predicted_KO_{g}.feather", tuned / f"predicted_KO_{g}.feather")
    (tuned / "tuning_meta.json").write_text(json.dumps(best, indent=2, default=str))


if __name__ == "__main__":
    main()
