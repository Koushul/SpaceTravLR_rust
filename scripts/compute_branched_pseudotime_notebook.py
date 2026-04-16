#!/usr/bin/env python3
"""
Reproduce embeds2perturb.ipynb branched pseudotime (cell 9):

  pairs = [('Naive CD4 T', 'T_follicular_helper'), ('Naive CD4 T', 'Th1'), ('Naive CD4 T', 'Th2')]
  source_cell_type = 'Naive CD4 T'
  annot = 'cell_type_2'
  tonsil.compute_branched_pseudotime(pairs, annot, source_cell_type, n_source_cells=1)

Writes CSV: obs_name,pseudotime (for spacetravlr-alignment --pseudotime-csv).

Requires: scanpy, scanpy.external (Palantir).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import scanpy as sc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--out-csv", required=True, help="Two columns: obs_name,pseudotime")
    ap.add_argument("--annot", default="cell_type_2")
    ap.add_argument("--source-cell-type", default="Naive CD4 T")
    ap.add_argument(
        "--pairs",
        nargs="*",
        default=[
            "Naive CD4 T|T_follicular_helper",
            "Naive CD4 T|Th1",
            "Naive CD4 T|Th2",
        ],
        help='Each "A|B" is one Palantir subgraph (same as notebook pairs).',
    )
    ap.add_argument("--n-source-cells", type=int, default=1)
    ap.add_argument("--palantir-knn", type=int, default=10)
    ap.add_argument("--palantir-n-components", type=int, default=5)
    args = ap.parse_args()

    try:
        import scanpy.external as sce
    except ImportError as e:
        print("Need scanpy with external (Palantir):", e, file=sys.stderr)
        sys.exit(1)

    adata = sc.read_h5ad(args.h5ad)
    if args.annot not in adata.obs:
        raise SystemExit(f"missing obs column {args.annot!r}")
    if "pseudotime" in adata.obs:
        del adata.obs["pseudotime"]

    pairs = []
    for s in args.pairs:
        parts = s.split("|", 1)
        if len(parts) != 2:
            raise SystemExit(f"bad --pairs entry {s!r}, expected A|B")
        pairs.append((parts[0].strip(), parts[1].strip()))

    source_cells = adata[adata.obs[args.annot] == args.source_cell_type].obs_names
    rng = np.random.default_rng(0)
    source_cells = rng.choice(np.asarray(source_cells), size=min(args.n_source_cells, len(source_cells)), replace=False)

    pseudo_frames = []
    for pair in pairs:
        sub = adata[adata.obs[args.annot].isin([pair[0], pair[1]])].copy()
        sce.tl.palantir(sub, n_components=args.palantir_n_components, knn=args.palantir_knn)
        for ij, cell in enumerate(source_cells):
            pltr = sce.tl.palantir_results(sub, cell)
            col = f"pseudotime_{pair[0]}_{pair[1]}_{ij}"
            _df = pltr.pseudotime.to_frame()
            _df.columns = [col]
            sub.obs = sub.obs.join(_df)
            pseudo_frames.append(sub.obs[[col]].copy())

    _df_pst = pd.concat(pseudo_frames).fillna(0).sum(axis=1).to_frame()
    _df_pst.columns = ["pseudotime"]
    _df_pst = _df_pst.groupby(_df_pst.index).mean()

    out = pd.DataFrame({"obs_name": _df_pst.index.astype(str), "pseudotime": _df_pst["pseudotime"].values})
    out.to_csv(args.out_csv, index=False)
    print("wrote", args.out_csv, "rows", len(out))


if __name__ == "__main__":
    main()
