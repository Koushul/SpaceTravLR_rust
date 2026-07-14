# Neighborhood Grammar Atlas

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
