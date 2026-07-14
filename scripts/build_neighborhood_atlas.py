#!/usr/bin/env python3
"""Build a neighborhood-grammar atlas from SlideSeqV2 / VisiumHD / SlideTags datasets.

Stores under data/neighborhood_atlas/:
  ontology.json          — global label set + lineage map
  manifest.json          — entry index
  entries/<id>/meta.json — per-dataset metadata + raw→harmonized map
  entries/<id>/structure.npz — Ŝ matrices + type counts

Cell-type labels are mapped to one shared ontology before computing Ŝ.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_structure_ligands import build_structure_ref  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "neighborhood_atlas"

# ---------------------------------------------------------------------------
# Global ontology (shared across organs / technologies)
# ---------------------------------------------------------------------------

ONTOLOGY: Dict[str, Dict] = {
    # Immune
    "B": {"lineage": "B", "organ_agnostic": True},
    "B_naive": {"lineage": "B", "organ_agnostic": True},
    "B_memory": {"lineage": "B", "organ_agnostic": True},
    "B_GC": {"lineage": "B", "organ_agnostic": True},
    "Plasma": {"lineage": "Plasma", "organ_agnostic": True},
    "CD4_T": {"lineage": "T", "organ_agnostic": True},
    "CD8_T": {"lineage": "T", "organ_agnostic": True},
    "Tfh": {"lineage": "T", "organ_agnostic": True},
    "Treg": {"lineage": "T", "organ_agnostic": True},
    "T": {"lineage": "T", "organ_agnostic": True},
    "NK": {"lineage": "NK", "organ_agnostic": True},
    "DC": {"lineage": "DC", "organ_agnostic": True},
    "Myeloid": {"lineage": "Myeloid", "organ_agnostic": True},
    "Macrophage": {"lineage": "Myeloid", "organ_agnostic": True},
    "Microglia": {"lineage": "Myeloid", "organ_agnostic": True},
    "Granulocyte": {"lineage": "Granulocyte", "organ_agnostic": True},
    "FDC": {"lineage": "Stromal", "organ_agnostic": True},
    # Vascular / stromal
    "Endothelial": {"lineage": "Endothelial", "organ_agnostic": True},
    "Stromal": {"lineage": "Stromal", "organ_agnostic": True},
    # Epithelial / parenchyma
    "Epithelial": {"lineage": "Epithelial", "organ_agnostic": True},
    # Neural
    "Neuron": {"lineage": "Neuron", "organ_agnostic": True},
    "Neuron_CA": {"lineage": "Neuron", "organ_agnostic": False},
    "Neuron_Dentate": {"lineage": "Neuron", "organ_agnostic": False},
    "Neuron_Interneuron": {"lineage": "Neuron", "organ_agnostic": False},
    "Neuron_Entorhinal": {"lineage": "Neuron", "organ_agnostic": False},
    "Neuron_Neurogenesis": {"lineage": "Neuron", "organ_agnostic": False},
    "Astrocyte": {"lineage": "Glia", "organ_agnostic": True},
    "Oligodendrocyte": {"lineage": "Glia", "organ_agnostic": True},
    "OPC": {"lineage": "Glia", "organ_agnostic": True},
    "Ependymal": {"lineage": "Ependymal", "organ_agnostic": True},
    "Choroid_Plexus": {"lineage": "Ependymal", "organ_agnostic": True},
    # Tumor
    "Tumor": {"lineage": "Tumor", "organ_agnostic": True},
    "Other": {"lineage": "Other", "organ_agnostic": True},
}

# ---------------------------------------------------------------------------
# Dataset registry (technology-restricted)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    path: str
    technology: str  # SlideSeqV2 | VisiumHD | SlideTags
    organ: str
    species: str
    obs_column: str
    mapping: Dict[str, str]
    notes: str = ""


DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        dataset_id="mouse_ln_slideseqv2",
        path="/ix1/ylee/kor11/tools/SpaceTravLR/data/SlideSeqV2_mouse_lymphnode.h5ad",
        technology="SlideSeqV2",
        organ="lymph_node",
        species="mouse",
        obs_column="cell_type",
        mapping={
            "B": "B_naive",
            "Resting T": "CD4_T",
            "Tfh": "Tfh",
            "CD8+ T": "CD8_T",
            "Treg": "Treg",
            "Th2": "CD4_T",
            "DC": "DC",
        },
        notes="Stickels/Cable-style LN Slide-seqV2; follicular B mapped to B_naive.",
    ),
    DatasetSpec(
        dataset_id="human_tonsil_slidetags",
        path="/ix1/ylee/kor11/tools/SpaceTravLR/data/Slidetags_human_tonsil.h5ad",
        technology="SlideTags",
        organ="tonsil",
        species="human",
        obs_column="cell_type",
        mapping={
            "B_germinal_center": "B_GC",
            "B_naive": "B_naive",
            "B_memory": "B_memory",
            "T_CD4": "CD4_T",
            "T_follicular_helper": "Tfh",
            "plasma": "Plasma",
            "FDC": "FDC",
            "T_CD8": "CD8_T",
            "NK": "NK",
            "mDC": "DC",
            "pDC": "DC",
            "myeloid": "Myeloid",
            "T_double_neg": "CD4_T",
        },
    ),
    DatasetSpec(
        dataset_id="human_tonsil_slidetags_fine",
        path="/ix1/ylee/kor11/tools/structures/data/h5ad/SlideTags_human_tonsil.h5ad",
        technology="SlideTags",
        organ="tonsil",
        species="human",
        obs_column="cell_type_2",
        mapping={
            "B_naive": "B_naive",
            "B_memory": "B_memory",
            "GC Light Zone": "B_GC",
            "GC Dark Zone": "B_GC",
            "GC Intermediate Zone": "B_GC",
            "plasma": "Plasma",
            "Naive CD4 T": "CD4_T",
            "Th1": "CD4_T",
            "Th2": "CD4_T",
            "T memory": "CD4_T",
            "T_follicular_helper": "Tfh",
            "T_CD8": "CD8_T",
            "Treg": "Treg",
            "mDC": "DC",
            "NKT": "NK",
            "FDC": "FDC",
            "myeloid": "Myeloid",
            "pDC": "DC",
            "T_double_neg": "CD4_T",
            "NK": "NK",
        },
        notes="Same SlideTags tonsil sample with finer T/GC subtypes.",
    ),
    DatasetSpec(
        dataset_id="human_melanoma_slidetags",
        path="/ix1/ylee/kor11/tools/SpaceTravLR/data/Slidetags_human_melanoma.h5ad",
        technology="SlideTags",
        organ="tumor_melanoma",
        species="human",
        obs_column="cell_type",
        mapping={
            "CD8+ T": "CD8_T",
            "CD4+ T": "CD4_T",
            "Treg": "Treg",
            "Plasma/B": "Plasma",
            "Mono-mac": "Myeloid",
            "Tumor 1": "Tumor",
            "Tumor 2": "Tumor",
            "Other": "Other",
        },
    ),
    DatasetSpec(
        dataset_id="mouse_kidney_visiumhd",
        path="/ix1/ylee/kor11/djishnu_kor11/training_data_2025/mouse_kidney_visiumHD.h5ad",
        technology="VisiumHD",
        organ="kidney",
        species="mouse",
        obs_column="cell_type",
        mapping={
            "Epithelial": "Epithelial",
            "Mesenchymal_Stromal": "Stromal",
            "Endothelial": "Endothelial",
            "Myeloid": "Myeloid",
            "T": "T",
            "NK": "NK",
            "B": "B",
        },
    ),
    DatasetSpec(
        dataset_id="mouse_brain_slideseqv2",
        path="/ix1/ylee/kor11/djishnu_kor11/training_data_2025/slideseq_brain_processed_v2.h5ad",
        technology="SlideSeqV2",
        organ="brain",
        species="mouse",
        obs_column="cell_type",
        mapping={
            "Neuron": "Neuron",
            "Endothelial": "Endothelial",
            "Oligodendrocyte": "Oligodendrocyte",
            "Choroid_Plexus": "Choroid_Plexus",
        },
    ),
    DatasetSpec(
        dataset_id="mouse_hippocampus_slideseqv2",
        path="/ix1/ylee/kor11/tools/spatial-signatures/data/anndata/slideseqv2.h5ad",
        technology="SlideSeqV2",
        organ="hippocampus",
        species="mouse",
        obs_column="cluster",
        mapping={
            "CA1_CA2_CA3_Subiculum": "Neuron_CA",
            "DentatePyramids": "Neuron_Dentate",
            "Interneurons": "Neuron_Interneuron",
            "Subiculum_Entorhinal_cl2": "Neuron_Entorhinal",
            "Subiculum_Entorhinal_cl3": "Neuron_Entorhinal",
            "Neurogenesis": "Neuron_Neurogenesis",
            "Astrocytes": "Astrocyte",
            "Oligodendrocytes": "Oligodendrocyte",
            "Polydendrocytes": "OPC",
            "Endothelial_Stalk": "Endothelial",
            "Endothelial_Tip": "Endothelial",
            "Microglia": "Microglia",
            "Mural": "Stromal",
            "Ependymal": "Ependymal",
        },
        notes="Classic hippocampus Slide-seqV2 (RCTD-style cluster labels).",
    ),
    DatasetSpec(
        dataset_id="mouse_brain_slideseqv2_regions",
        path="/ix1/ylee/kor11/djishnu_kor11/lasso_runs/oxidized/brain_train/slideseqv2_brain.h5ad",
        technology="SlideSeqV2",
        organ="brain",
        species="mouse",
        obs_column="cell_type",
        mapping={
            "Oligodendrocyte": "Oligodendrocyte",
            "Astrocyte": "Astrocyte",
            "Microglia": "Microglia",
            "Endothelial": "Endothelial",
            "Vascular": "Endothelial",
            "Choroid Plexus": "Choroid_Plexus",
            "Neuron": "Neuron",
            "Allocortex": "Neuron",
            "CA1/2/3": "Neuron_CA",
            "Dentate gyrus": "Neuron_Dentate",
            "Neocortex": "Neuron",
            "Hippocampal region": "Neuron",
            "Thalamus LD": "Neuron",
            "Thalamus LP": "Neuron",
            "Hypothalamus": "Neuron",
            "Striatum": "Neuron",
            "Fiber tracts": "Oligodendrocyte",
        },
        notes="Regional + glial labels; cortical/thalamic/striatal regions → Neuron; fiber tracts → Oligodendrocyte.",
    ),
]


def median_nn(xy: np.ndarray, max_pts: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = xy.shape[0]
    idx = rng.choice(n, size=min(n, max_pts), replace=False) if n > max_pts else np.arange(n)
    pts = xy[idx]
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(d2.min(1))))


def choose_radius(xy: np.ndarray, k_nn: float = 12.0) -> float:
    nn = median_nn(xy)
    return float(max(nn * k_nn, nn * 5.0))


def load_xy(adata: ad.AnnData) -> np.ndarray:
    for key in ("spatial", "X_spatial", "spatial_loc"):
        if key in adata.obsm:
            xy = np.asarray(adata.obsm[key], dtype=np.float64)
            if xy.ndim == 2 and xy.shape[1] >= 2:
                return xy[:, :2]
    if {"x", "y"}.issubset(adata.obs.columns):
        return np.column_stack(
            [
                adata.obs["x"].astype(float).to_numpy(),
                adata.obs["y"].astype(float).to_numpy(),
            ]
        )
    raise KeyError("No spatial coordinates found")


def stratified_subsample(
    labels: np.ndarray, max_cells: int, seed: int = 0
) -> np.ndarray:
    n = labels.shape[0]
    if n <= max_cells:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    types, counts = np.unique(labels, return_counts=True)
    # Proportional allocation with at least 1 per type when possible
    alloc = np.maximum(1, np.floor(counts / counts.sum() * max_cells).astype(int))
    while alloc.sum() > max_cells:
        alloc[np.argmax(alloc)] -= 1
    while alloc.sum() < max_cells:
        # add to largest remaining pool
        remain = counts - alloc
        if remain.max() <= 0:
            break
        alloc[np.argmax(remain)] += 1
    sel = []
    for t, k in zip(types, alloc):
        idx = np.where(labels == t)[0]
        take = min(int(k), idx.size)
        sel.append(rng.choice(idx, size=take, replace=False))
    return np.sort(np.concatenate(sel))


def validate_mapping(spec: DatasetSpec, raw_labels: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Return (unmapped_raw_labels, invalid_ontology_targets)."""
    present = sorted(set(raw_labels))
    unmapped = [t for t in present if t not in spec.mapping]
    invalid = sorted(
        {v for v in spec.mapping.values() if v not in ONTOLOGY}
    )
    return unmapped, invalid


def project_to_lineage(cell_types: Sequence[str], matrix: np.ndarray) -> Tuple[List[str], np.ndarray]:
    """Aggregate Ŝ over lineage buckets for cross-organ comparison."""
    lineages = sorted({ONTOLOGY[t]["lineage"] for t in cell_types})
    lin_idx = {l: i for i, l in enumerate(lineages)}
    n = len(lineages)
    out = np.zeros((n, n), dtype=np.float64)
    type_to_lin = [lin_idx[ONTOLOGY[t]["lineage"]] for t in cell_types]
    for i, li in enumerate(type_to_lin):
        for j, lj in enumerate(type_to_lin):
            out[li, lj] += matrix[i, j]
    # Average when multiple subtypes map into same lineage row/col.
    # Recompute as mean of subtype rows/cols that belong to each lineage.
    out = np.zeros((n, n), dtype=np.float64)
    for li, L in enumerate(lineages):
        rows = [i for i, t in enumerate(cell_types) if ONTOLOGY[t]["lineage"] == L]
        for lj, M in enumerate(lineages):
            cols = [j for j, t in enumerate(cell_types) if ONTOLOGY[t]["lineage"] == M]
            block = matrix[np.ix_(rows, cols)]
            out[li, lj] = float(block.mean()) if block.size else 0.0
    return lineages, out


def process_dataset(
    spec: DatasetSpec,
    max_cells: int,
    radius_k: float,
    seed: int,
) -> Dict:
    t0 = time.time()
    path = Path(spec.path)
    if not path.exists():
        raise FileNotFoundError(spec.path)

    adata = ad.read_h5ad(spec.path)
    if spec.obs_column not in adata.obs.columns:
        raise KeyError(f"{spec.dataset_id}: missing obs column {spec.obs_column}")

    raw = adata.obs[spec.obs_column].astype(str).to_numpy()
    unmapped, invalid = validate_mapping(spec, raw)
    if invalid:
        raise ValueError(f"{spec.dataset_id}: mapping targets not in ontology: {invalid}")
    if unmapped:
        raise ValueError(
            f"{spec.dataset_id}: unmapped raw labels (refuse silent drop): {unmapped}"
        )

    harm = np.array([spec.mapping[x] for x in raw], dtype=object)
    for h in sorted(set(harm.tolist())):
        if h not in ONTOLOGY:
            raise ValueError(f"{spec.dataset_id}: harmonized label missing from ontology: {h}")

    xy = load_xy(adata)
    n_full = int(xy.shape[0])
    sel = stratified_subsample(harm, max_cells, seed=seed)
    xy_s = xy[sel]
    harm_s = harm[sel]
    raw_s = raw[sel]

    radius = choose_radius(xy_s, k_nn=radius_k)
    nn = median_nn(xy_s)
    ref = build_structure_ref(xy_s, [str(x) for x in harm_s], radius)

    # Drop private per-cell arrays before serialize
    cell_types = list(ref["cell_types"])
    mean_w = np.asarray(ref["mean_weight_mass"], dtype=np.float64)
    mean_s = np.asarray(ref["mean_soft_counts"], dtype=np.float64)
    mean_h = np.asarray(ref["mean_hard_counts"], dtype=np.float64)
    counts = np.asarray(ref["ref_type_counts"], dtype=np.int64)
    lineages, mean_w_lin = project_to_lineage(cell_types, mean_w)
    _, mean_s_lin = project_to_lineage(cell_types, mean_s)

    # Abundance baseline matrix for convenience
    freqs = counts.astype(np.float64)
    freqs = freqs / max(freqs.sum(), 1.0)
    abund = np.zeros_like(mean_w)
    for r in range(mean_w.shape[0]):
        abund[r] = mean_w[r].sum() * freqs

    entry_dir = OUT / "entries" / spec.dataset_id
    entry_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        entry_dir / "structure.npz",
        cell_types=np.asarray(cell_types, dtype=object),
        lineages=np.asarray(lineages, dtype=object),
        mean_weight_mass=mean_w,
        mean_soft_counts=mean_s,
        mean_hard_counts=mean_h,
        abundance_baseline=abund,
        mean_weight_mass_lineage=mean_w_lin,
        mean_soft_counts_lineage=mean_s_lin,
        ref_type_counts=counts,
        radius=np.array([radius], dtype=np.float64),
        scale_factor=np.array([1.0], dtype=np.float64),
        hard_radius=np.array([radius], dtype=np.float64),
        median_nn=np.array([nn], dtype=np.float64),
        n_ref_cells=np.array([xy_s.shape[0]], dtype=np.int64),
        n_source_cells=np.array([n_full], dtype=np.int64),
    )

    # Raw→harmonized usage counts on the (possibly subsampled) cells used for Ŝ
    usage: Dict[str, Dict[str, int]] = {}
    for r, h in zip(raw_s.tolist(), harm_s.tolist()):
        usage.setdefault(r, {})
        usage[r][h] = usage[r].get(h, 0) + 1

    meta = {
        "dataset_id": spec.dataset_id,
        "source_path": str(path.resolve()),
        "technology": spec.technology,
        "organ": spec.organ,
        "species": spec.species,
        "obs_column": spec.obs_column,
        "mapping": dict(spec.mapping),
        "mapping_usage_counts": usage,
        "notes": spec.notes,
        "n_source_cells": n_full,
        "n_ref_cells": int(xy_s.shape[0]),
        "subsampled": bool(n_full > xy_s.shape[0]),
        "max_cells": max_cells,
        "seed": seed,
        "radius": radius,
        "radius_k_nn": radius_k,
        "median_nn": nn,
        "cell_types": cell_types,
        "lineages": lineages,
        "ref_type_counts": {t: int(c) for t, c in zip(cell_types, counts)},
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (entry_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def write_ontology() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "description": (
            "Shared cell-type ontology for neighborhood-grammar atlas entries. "
            "Every harmonized label used in structure.npz must appear here."
        ),
        "labels": ONTOLOGY,
        "lineages": sorted({v["lineage"] for v in ONTOLOGY.values()}),
    }
    (OUT / "ontology.json").write_text(json.dumps(payload, indent=2) + "\n")


def write_manifest(entries: List[Dict]) -> None:
    manifest = {
        "version": 1,
        "technologies": sorted({e["technology"] for e in entries}),
        "organs": sorted({e["organ"] for e in entries}),
        "n_entries": len(entries),
        "entries": [
            {
                "dataset_id": e["dataset_id"],
                "technology": e["technology"],
                "organ": e["organ"],
                "species": e["species"],
                "n_ref_cells": e["n_ref_cells"],
                "n_source_cells": e["n_source_cells"],
                "radius": e["radius"],
                "cell_types": e["cell_types"],
                "path": f"entries/{e['dataset_id']}",
            }
            for e in entries
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_readme() -> None:
    text = """# Neighborhood Grammar Atlas

Tissue neighborhood structure matrices (`Ŝ[receiver, sender]`) built from
**SlideSeqV2**, **VisiumHD**, and **SlideTags** datasets with a shared cell-type ontology.

## Load

```python
from scripts.load_neighborhood_atlas import load_atlas, get_entry

atlas = load_atlas()                 # manifest + all entries
entry = get_entry("mouse_ln_slideseqv2")
S = entry["mean_weight_mass"]        # (T, T) type-pooled Gaussian weight mass
types = entry["cell_types"]
S_lin = entry["mean_weight_mass_lineage"]  # lineage-projected for cross-organ compare
```

## Layout

- `ontology.json` — allowed harmonized labels + lineage map
- `manifest.json` — index of all entries
- `entries/<id>/meta.json` — raw→harmonized map, radius, counts
- `entries/<id>/structure.npz` — matrices (easily `np.load`)

Radius is adaptive: `≈ 12 × median nearest-neighbor distance` on the cells used
to estimate `Ŝ`. Large tissues are stratified-subsampled (documented in meta).
"""
    (OUT / "README.md").write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-cells", type=int, default=12000)
    ap.add_argument("--radius-k", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional dataset_id subset",
    )
    args = ap.parse_args()

    write_ontology()
    write_readme()
    OUT.mkdir(parents=True, exist_ok=True)

    specs = DATASETS
    if args.only:
        want = set(args.only)
        specs = [s for s in DATASETS if s.dataset_id in want]
        missing = want - {s.dataset_id for s in specs}
        if missing:
            raise SystemExit(f"Unknown dataset ids: {sorted(missing)}")

    entries: List[Dict] = []
    for spec in specs:
        print(f"[atlas] building {spec.dataset_id} ({spec.technology} / {spec.organ}) ...", flush=True)
        meta = process_dataset(spec, args.max_cells, args.radius_k, args.seed)
        print(
            f"  n={meta['n_ref_cells']}/{meta['n_source_cells']} "
            f"types={len(meta['cell_types'])} radius={meta['radius']:.3f} "
            f"({meta['elapsed_sec']}s)",
            flush=True,
        )
        entries.append(meta)

    write_manifest(entries)
    # Consistency check: every used label ⊆ ontology
    used = sorted({t for e in entries for t in e["cell_types"]})
    missing = [t for t in used if t not in ONTOLOGY]
    if missing:
        raise SystemExit(f"Ontology gap after build: {missing}")
    print(f"[atlas] wrote {len(entries)} entries → {OUT}")
    print(f"[atlas] harmonized labels used: {used}")


if __name__ == "__main__":
    main()
