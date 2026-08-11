# Tonsil GC niche analysis

Offline scripts and static HTML reports for SpaceTravLR β microniches on human tonsil snRNA (germinal-center B cells and Tfh). Not part of the MkDocs site.

## Prepare benchmarks

```bash
python analysis/tonsil_niche/prepare_tonsil_niche_benchmark.py
python analysis/tonsil_niche/prepare_tonsil_niche_functional_proof.py
python analysis/tonsil_niche/prepare_microniche_pathway_analysis.py
```

Outputs land under `analysis/tonsil_niche/public/tonsil_niche_benchmark/` (method metrics, functional proof, `get-microniches` pathway HTML).

## CLI

```bash
spacetravlr get-microniches \
  --run-toml path/to/spacetravlr_run_repro.toml \
  --cell-type B_germinal_center \
  --out ./microniches
```
