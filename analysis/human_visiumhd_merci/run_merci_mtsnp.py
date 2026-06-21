#!/usr/bin/env python3
"""Run MERCI-mtSNP on a Space Ranger possorted_genome_bam.bam for Visium HD."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam", type=Path, required=True)
    parser.add_argument("--barcodes-dir", type=Path, required=True, help="filtered_feature_bc_matrix dir with barcodes.tsv.gz")
    parser.add_argument("--genome-fa", type=Path, required=True)
    parser.add_argument("--sample-id", default="P1CRC")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--merci-script", type=Path, default=None)
    args = parser.parse_args()

    merci_script = args.merci_script or (
        Path(__file__).resolve().parent.parent / "mc38_visiumhd" / "MERCI" / "MERCI-mtSNP.py"
    )
    if not merci_script.exists():
        sys.exit(f"MERCI-mtSNP.py not found at {merci_script}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bam = args.bam.resolve()
    bai = Path(str(bam) + ".bai")
    if not bai.exists():
        subprocess.run(["samtools", "index", str(bam)], check=True)

    cmd = [
        sys.executable,
        str(merci_script),
        "-D", "10x_scRNA-seq",
        "-i", args.sample_id,
        "-b", str(bam),
        "-f", str(args.genome_fa.resolve()),
        "-c", str(args.barcodes_dir.resolve()),
        "-o", str(args.out_dir.resolve()),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
