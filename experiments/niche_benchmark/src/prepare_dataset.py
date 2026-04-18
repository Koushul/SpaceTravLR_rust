"""Prepare the SlideTags human tonsil benchmark AnnData.

The Slide-tags human tonsil dataset (Russell et al. 2024) carries a
fine-grained `cell_type_2` annotation that maps directly onto well-known
*functional microniches* of the secondary lymphoid follicle:

  * **Germinal Center Light Zone** (centrocytes + FDC + Tfh interaction)
  * **Germinal Center Dark Zone** (proliferating centroblasts)
  * **Germinal Center Intermediate Zone**
  * **FDC** network (defines GC scaffolding)
  * **T_follicular_helper / Treg / NKT / Th1 / Th2 / T_memory / Naive CD4 T / T_CD8**
    (T-cell zone microniches)
  * **B_naive** mantle / **B_memory**
  * **plasma** cell foci
  * **mDC**, **pDC**, **myeloid**

These cell-type-2 labels are used as the *ground-truth functional microniche*
for the benchmark. We also keep the coarser `cell_type` as a sanity check.

Outputs
-------
- ``experiments/niche_benchmark/results/tonsil_prepared.h5ad``: AnnData with
  raw counts in ``.layers['counts']``, log-normalized ``.X``, ``microniche_gt``
  in ``.obs`` and 2D coords in ``.obsm['spatial']``.
"""

from __future__ import annotations

from pathlib import Path

from _common import (
    DEFAULT_DATASET,
    EXP_DIR,
    GROUND_TRUTH_KEY,
    RESULTS_DIR,
    add_normalized_layers,
    basic_qc,
    load_dataset,
)


def main() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    adata = load_dataset(DEFAULT_DATASET, ground_truth_key=GROUND_TRUTH_KEY)
    print(f"loaded {DEFAULT_DATASET}: {adata.shape}")

    adata = basic_qc(adata, min_counts=200, min_genes_per_cell=50, min_cells_per_gene=5)
    print(f"after QC: {adata.shape}")

    adata = add_normalized_layers(adata)

    out = RESULTS_DIR / "tonsil_prepared.h5ad"
    adata.write_h5ad(out)
    print(f"wrote {out}: {adata.shape}, "
          f"{adata.obs['microniche_gt'].nunique()} ground-truth microniches")
    print(adata.obs["microniche_gt"].value_counts())
    return out


if __name__ == "__main__":
    main()
