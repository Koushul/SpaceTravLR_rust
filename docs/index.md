# <span style="font-size:2em;">SpaceTravLR</span> {: .st-brand }
SpaceTravLR infers how single or combinatorial genetic perturbations rewire signals across the tissue neighbourhood, by propagating effects through underlying spatially resolved molecular networks, thereby modelling how perturbations can reshape both the targeted cell and its surrounding neighbourhood.


# I want to 🧞️

<div id="st-i-want-to" class="st-i-want-to" markdown="0"></div>



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


<!-- ![](assets/concepts_A.png) -->



<!-- ![](assets/concepts_B.png) -->



## Training time estimate {#training-time-estimate}

The equation below provides a rough estimate of how long training SpaceTravLR on your dataset should take. This was empirically estimated using multiple runs across datasets on a100, l40s and rtx6k GPUs.

\[
T_{\mathrm{seconds}}
  \approx
  \frac{26.259\; G}{W \cdot s_{\mathrm{gpu}}}
  \cdot \left(
    \frac{N_{\mathrm{cnn}}}{200}
    \left(
      \frac{\ln(1 + N/N_{\mathrm{cnn}})}{\ln 2}
    \right)^{0.4}
  \right)^{0.85}
  \cdot \left(\frac{D}{16}\right)^{1.5}
  \cdot \frac{E}{100}
\]

<div id="st-runtime-estimator" class="st-runtime-estimator" markdown="0"></div>

\(s_{\mathrm{gpu}}\) is a GPU model specific coefficient representing the speedup relative to only using the CPU.

`cnn_max_cells` allows the CNN to smartly subsample the dataset as the training converges. Higher values means later epochs see fewer and fewer cells, focusing on tissue region where the residual errors are higher.