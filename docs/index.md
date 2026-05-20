# <span style="font-size:2em;">SpaceTravLR</span> {: .st-brand }
The advent of spatial omics has revolutionised our understanding of tissue biology; however, these technologies remain largely descriptive and do not capture how changes in gene regulation propagate across spatial neighbourhoods.

Here we develop SpaceTravLR, a first interpretable machine-learning that generalises across tissues and species, uncovering spatial features linked to functional outcomes.

## Quickstart
Install me
```bash
curl -fsSL https://tinyurl.com/spacetravlr/scripts/install.sh | sh
```
Run me
```bash
spacetravlr --h5ad /path/to/adata.h5ad --output-dir /path/to/outputdir
```

Join me
```bash
spacetravlr --join-output-dir  /path/to/outputdir
```

Analyze me
```bash
spacetravlr collect-interactions \
  --run-toml /path/to/outputdir/spacetravlr_run_repro.toml 
```

Learn more about [how SpaceTravLR works](math.md), other [installations](install.md) details and CLI [usage](usage.md).


![](assets/concepts_A.png)

SpaceTravLR infers how single or combinatorial genetic perturbations rewire signals across the tissue neighbourhood, by propagating effects through underlying spatially resolved molecular networks, thereby modelling how perturbations can reshape both the targeted cell and its surrounding neighbourhood.

![](assets/concepts_B.png)




