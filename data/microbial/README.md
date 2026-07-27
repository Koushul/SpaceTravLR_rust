# Microbial data (v0)

Editable priors for the **bacterial secretion-source** extension (see [`docs/microbial.md`](../../docs/microbial.md)).

| File | Role |
|------|------|
| `bact_host_interactions.v0.csv` | Microbial signal → host receptor pairs (+ default radii) |
| `taxon_signal_priors.v0.csv` | Taxon / Gram / phylum → which signals they can emit |

These are **not** loaded by the trainer yet (M0 design only). Receptor symbols are mouse-style (`Tlr4`); human runs will need UPPER mapping.

## Sender table (future)

Training will expect a parquet/CSV with bacterial loci, e.g.:

| column | type | meaning |
|--------|------|---------|
| `sender_id` | str | bin or colony id |
| `x`, `y` | f64 | coordinates in **µm** (same frame as host `obsm['spatial']`) |
| `Lps`, `Lta`, … | f64 | per-signal amounts \(A_{bk}\) |

Built from Stereo-seq `*_unmap.h5ad` via taxon priors.
