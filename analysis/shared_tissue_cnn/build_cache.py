"""Build spatial tensor cache from h5ad without training."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_utils import build_spatial_cache


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--spatial-dim", type=int, default=16)
    p.add_argument("--radius", type=float, default=300.0)
    p.add_argument("--force-genes", default="")
    args = p.parse_args()
    force = [g.strip() for g in args.force_genes.split(",") if g.strip()]
    cache = build_spatial_cache(
        args.h5ad,
        spatial_dim=args.spatial_dim,
        radius=args.radius,
        force_genes=force or None,
    )
    cache.save(args.cache)
    print(f"wrote {args.cache} ({len(cache.obs_names)} cells)")


if __name__ == "__main__":
    main()
