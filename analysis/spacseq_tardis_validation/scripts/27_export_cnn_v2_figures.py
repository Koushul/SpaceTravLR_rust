#!/usr/bin/env python3
"""Export all tuned CNN v2 microniche figures into a single folder."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_FIG_DIR = ROOT / "figures" / "cnn_microniche_v2_improved"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="cnn_v2")
    ap.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    ap.add_argument("--recompute", action="store_true", help="Re-run enrichment instead of using cached CSVs")
    args = ap.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    figures_only = [] if args.recompute else ["--figures-only"]

    print(f"Exporting CNN microniche figures → {args.fig_dir}")
    subprocess.run(
        [py, str(HERE / "23_cnn_microniche_enrichment.py"), "--tag", args.tag, "--fig-dir", str(args.fig_dir), *figures_only],
        check=True,
    )
    subprocess.run(
        [py, str(HERE / "25_cnn_microniche_controls.py"), "--tag", args.tag, "--fig-dir", str(args.fig_dir), *figures_only],
        check=True,
    )

    subprocess.run(
        [py, str(HERE / "29_spatial_embedding_enrichment.py"), "--tag", args.tag, "--fig-dir", str(args.fig_dir)],
        check=True,
    )

    n_png = len(list(args.fig_dir.glob("*.png")))
    n_svg = len(list(args.fig_dir.glob("*.svg")))
    print(f"Done: {n_png} PNG + {n_svg} SVG in {args.fig_dir}")


if __name__ == "__main__":
    main()
