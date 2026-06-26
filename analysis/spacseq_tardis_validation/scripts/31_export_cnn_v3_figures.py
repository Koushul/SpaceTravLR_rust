#!/usr/bin/env python3
"""Build CNN v3 microniches (higher niche count) and export figures."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_FIG_DIR = ROOT / "figures" / "cnn_microniche_v3_improved"
TAG = "cnn_v3"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument("--skip-tune", action="store_true", help="Use existing per_slice_leiden_v3.json")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    if not args.skip_tune:
        print("Tuning per-slice Leiden for more niches (v3)...")
        subprocess.run(
            [
                py, str(HERE / "30_tune_per_slice_microniches.py"),
                "--objective", "niches",
                "--coarse",
                "--min-r-floor", "0.15",
            ],
            check=True,
        )

    enrich_flags = [] if args.figures_only else []
    print(f"Running enrichment tag={args.tag} → {args.fig_dir}")
    subprocess.run(
        [
            py, str(HERE / "23_cnn_microniche_enrichment.py"),
            "--tag", args.tag,
            "--fig-dir", str(args.fig_dir),
            *enrich_flags,
        ],
        check=True,
    )
    subprocess.run(
        [
            py, str(HERE / "25_cnn_microniche_controls.py"),
            "--tag", args.tag,
            "--fig-dir", str(args.fig_dir),
            *([] if args.figures_only else []),
        ],
        check=True,
    )
    subprocess.run(
        [
            py, str(HERE / "29_spatial_embedding_enrichment.py"),
            "--tag", args.tag,
            "--fig-dir", str(args.fig_dir),
        ],
        check=True,
    )

    n_png = len(list(args.fig_dir.glob("*.png")))
    n_svg = len(list(args.fig_dir.glob("*.svg")))
    print(f"Done: {n_png} PNG + {n_svg} SVG in {args.fig_dir}")


if __name__ == "__main__":
    main()
