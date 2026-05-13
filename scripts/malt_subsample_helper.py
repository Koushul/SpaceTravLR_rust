"""
Thin wrapper: subset a query h5ad to given obs_names, run MALT, clean up temp.
Called by the umap_lab Rust server for the optimized MALT path.
Does NOT modify malt_label_transfer.py — imports and calls run_malt().
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

def main() -> None:
    p = argparse.ArgumentParser(description="Subsample query then run MALT")
    p.add_argument("--query", required=True, help="Full query .h5ad path")
    p.add_argument("--reference", required=True, help="Reference .h5ad path")
    p.add_argument("--subset-names-json", required=True,
                   help="JSON file containing a list of obs_names to keep")
    p.add_argument("--outdir", required=True)
    p.add_argument("--groupby", default=None)
    p.add_argument("--no-leiden-map", action="store_true")
    args = p.parse_args()

    with open(args.subset_names_json) as f:
        keep_names = set(json.load(f))

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from malt_label_transfer import read_h5ad_compat, run_malt

    print(f"[subsample] Loading query: {args.query}")
    query = read_h5ad_compat(args.query)
    mask = query.obs_names.isin(keep_names)
    n_keep = int(mask.sum())
    print(f"[subsample] Keeping {n_keep}/{query.n_obs} cells "
          f"({n_keep / query.n_obs * 100:.1f}%)")
    subset = query[mask].copy()
    del query

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5ad")
    os.close(tmp_fd)
    try:
        subset.write_h5ad(tmp_path)
        del subset

        gb = [args.groupby] if args.groupby else None
        run_malt(
            reference_path=args.reference,
            query_path=tmp_path,
            groupby_columns=gb,
            outdir=args.outdir,
            leiden_map=not args.no_leiden_map,
        )
    finally:
        try:
            os.unlink(tmp_path)
            print(f"[subsample] Cleaned up temp h5ad")
        except OSError:
            pass


if __name__ == "__main__":
    main()
