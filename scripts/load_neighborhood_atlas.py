#!/usr/bin/env python3
"""Load the neighborhood-grammar atlas from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ATLAS_DIR = ROOT / "data" / "neighborhood_atlas"


def atlas_root(path: Optional[Path] = None) -> Path:
    return Path(path) if path is not None else ATLAS_DIR


def load_ontology(path: Optional[Path] = None) -> Dict:
    return json.loads((atlas_root(path) / "ontology.json").read_text())


def load_manifest(path: Optional[Path] = None) -> Dict:
    return json.loads((atlas_root(path) / "manifest.json").read_text())


def list_entries(
    *,
    technology: Optional[str] = None,
    organ: Optional[str] = None,
    species: Optional[str] = None,
    path: Optional[Path] = None,
) -> List[Dict]:
    entries = load_manifest(path)["entries"]
    out = []
    for e in entries:
        if technology and e["technology"] != technology:
            continue
        if organ and e["organ"] != organ:
            continue
        if species and e["species"] != species:
            continue
        out.append(e)
    return out


def get_entry(dataset_id: str, path: Optional[Path] = None) -> Dict:
    root = atlas_root(path)
    meta_path = root / "entries" / dataset_id / "meta.json"
    npz_path = root / "entries" / dataset_id / "structure.npz"
    if not meta_path.exists() or not npz_path.exists():
        raise FileNotFoundError(f"Atlas entry not found: {dataset_id} under {root}")
    meta = json.loads(meta_path.read_text())
    z = np.load(npz_path, allow_pickle=True)
    entry = dict(meta)
    entry.update(
        {
            "cell_types": [str(x) for x in z["cell_types"]],
            "lineages": [str(x) for x in z["lineages"]],
            "mean_weight_mass": np.asarray(z["mean_weight_mass"], dtype=np.float64),
            "mean_soft_counts": np.asarray(z["mean_soft_counts"], dtype=np.float64),
            "mean_hard_counts": np.asarray(z["mean_hard_counts"], dtype=np.float64),
            "abundance_baseline": np.asarray(z["abundance_baseline"], dtype=np.float64),
            "mean_weight_mass_lineage": np.asarray(
                z["mean_weight_mass_lineage"], dtype=np.float64
            ),
            "mean_soft_counts_lineage": np.asarray(
                z["mean_soft_counts_lineage"], dtype=np.float64
            ),
            "ref_type_counts_vec": np.asarray(z["ref_type_counts"], dtype=np.int64),
            "radius": float(z["radius"][0]),
            "median_nn": float(z["median_nn"][0]),
            "scale_factor": float(z["scale_factor"][0]),
        }
    )
    return entry


def load_atlas(path: Optional[Path] = None) -> Dict:
    root = atlas_root(path)
    manifest = load_manifest(root)
    ontology = load_ontology(root)
    entries = {e["dataset_id"]: get_entry(e["dataset_id"], root) for e in manifest["entries"]}
    return {"root": str(root), "manifest": manifest, "ontology": ontology, "entries": entries}


def structure_matrix(
    dataset_id: str,
    *,
    kind: str = "weight",
    lineage: bool = False,
    path: Optional[Path] = None,
) -> tuple:
    """Return (labels, matrix). kind in {weight, soft, hard, abundance}."""
    e = get_entry(dataset_id, path)
    if lineage:
        key = {
            "weight": "mean_weight_mass_lineage",
            "soft": "mean_soft_counts_lineage",
        }.get(kind)
        if key is None:
            raise ValueError("lineage projection only for kind=weight|soft")
        return e["lineages"], e[key]
    key = {
        "weight": "mean_weight_mass",
        "soft": "mean_soft_counts",
        "hard": "mean_hard_counts",
        "abundance": "abundance_baseline",
    }[kind]
    return e["cell_types"], e[key]


def shared_type_cosine(a: str, b: str, path: Optional[Path] = None) -> Dict:
    """Cosine similarity of Ŝ on the intersection of harmonized types."""
    ea = get_entry(a, path)
    eb = get_entry(b, path)
    shared = sorted(set(ea["cell_types"]) & set(eb["cell_types"]))
    if len(shared) < 1:
        return {"n_shared": 0, "cosine": float("nan"), "shared_types": []}
    ia = [ea["cell_types"].index(t) for t in shared]
    ib = [eb["cell_types"].index(t) for t in shared]
    ma = ea["mean_weight_mass"][np.ix_(ia, ia)].ravel()
    mb = eb["mean_weight_mass"][np.ix_(ib, ib)].ravel()
    denom = float(np.linalg.norm(ma) * np.linalg.norm(mb))
    cos = float(ma @ mb / denom) if denom > 0 else float("nan")
    return {"n_shared": len(shared), "cosine": cos, "shared_types": shared}


def assert_label_consistency(path: Optional[Path] = None) -> None:
    """Raise if any entry uses a label outside ontology.json."""
    ontology = load_ontology(path)["labels"]
    for e in list_entries(path=path):
        entry = get_entry(e["dataset_id"], path)
        bad = [t for t in entry["cell_types"] if t not in ontology]
        if bad:
            raise AssertionError(f"{e['dataset_id']} has non-ontology labels: {bad}")
        for raw, target in entry["mapping"].items():
            if target not in ontology:
                raise AssertionError(
                    f"{e['dataset_id']} maps {raw!r} → {target!r} (not in ontology)"
                )


__all__ = [
    "ATLAS_DIR",
    "assert_label_consistency",
    "get_entry",
    "list_entries",
    "load_atlas",
    "load_manifest",
    "load_ontology",
    "shared_type_cosine",
    "structure_matrix",
]
