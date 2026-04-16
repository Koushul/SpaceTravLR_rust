#!/usr/bin/env python3
"""Write a small analysis copy with obs['pseudotime'] = UMAP dim1 (for alignment script)."""
from pathlib import Path

import numpy as np
import scanpy as sc

SRC = "/ix/djishnu/shared/djishnu_kor11/training_data_revision/snrna_human_tonsil.h5ad"
HERE = Path(__file__).resolve().parent
DST = HERE / "analysis" / "tonsil_with_pseudotime.h5ad"


def main() -> None:
    DST.parent.mkdir(parents=True, exist_ok=True)
    a = sc.read_h5ad(SRC)
    u = np.asarray(a.obsm["X_umap"], dtype=np.float64)
    a.obs["pseudotime"] = u[:, 0]
    a.write_h5ad(DST)
    print("wrote", DST)


if __name__ == "__main__":
    main()
