# Neighborhood Grammar Atlas

Type-pooled Gaussian neighborhood structure matrices (`Ŝ[receiver, sender]`) for
**SlideSeqV2**, **VisiumHD**, and **SlideTags** tissues, with one shared cell-type ontology.

## Build

```bash
python scripts/build_neighborhood_atlas.py --max-cells 12000
```

## Load

```python
from scripts.load_neighborhood_atlas import load_atlas, get_entry, shared_type_cosine

atlas = load_atlas()
entry = get_entry("mouse_ln_slideseqv2")
S = entry["mean_weight_mass"]       # (T, T)
types = entry["cell_types"]
S_lin = entry["mean_weight_mass_lineage"]  # cross-organ lineage projection
```

On-disk layout (`data/neighborhood_atlas/`):

| Path | Contents |
|------|----------|
| `ontology.json` | Allowed harmonized labels + lineage map |
| `manifest.json` | Index of all entries |
| `entries/<id>/meta.json` | Raw→harmonized map, radius, counts |
| `entries/<id>/structure.npz` | `Ŝ` matrices (`np.load`) |

Radius ≈ `12 × median NN` on the cells used for `Ŝ`. Large tissues are
stratified-subsampled (documented in each `meta.json`).

## Included entries

| ID | Tech | Organ | Species |
|----|------|-------|---------|
| `mouse_ln_slideseqv2` | SlideSeqV2 | lymph_node | mouse |
| `mouse_hippocampus_slideseqv2` | SlideSeqV2 | hippocampus | mouse |
| `mouse_brain_slideseqv2` | SlideSeqV2 | brain | mouse |
| `mouse_brain_slideseqv2_regions` | SlideSeqV2 | brain | mouse |
| `mouse_kidney_visiumhd` | VisiumHD | kidney | mouse |
| `human_tonsil_slidetags` | SlideTags | tonsil | human |
| `human_tonsil_slidetags_fine` | SlideTags | tonsil | human |
| `human_melanoma_slidetags` | SlideTags | tumor_melanoma | human |

Label consistency is enforced at build time: every raw label must map into
`ontology.json` (no silent drops).
