<!-- # <span style="font-size:2em;">SpaceTravLR</span> {: .st-brand } -->

SpaceTravLR is a computational framework for modeling how cells extend their transcriptome beyond their own cytoplasm. SpaceTravLR aims to push spatial analyses at single cell resolution from descriptive
correlations toward functional mechanistic insights. 
<div class="st-overview" markdown="0">
--8<-- "docs/assets/overview.svg"
</div>

<div class="st-threeui-tree" markdown="0">
  <iframe
    title="Sylva tree"
    src="threeui/sylva-tree-scene.html"
    loading="eager"
    tabindex="-1"
    allowtransparency="true"
    style="background: transparent; color-scheme: inherit"
  ></iframe>
</div>
<script>
(function () {
  var frame = document.querySelector(".st-threeui-tree iframe");
  if (!frame) return;
  function setVisible(on) {
    try {
      var win = frame.contentWindow;
      if (win) win.__sylvaVisible = !!on;
    } catch (err) {}
  }
  function syncScheme() {
    try {
      var win = frame.contentWindow;
      var doc = frame.contentDocument;
      var scheme = document.body.getAttribute("data-md-color-scheme") || "default";
      var colorScheme = scheme === "slate" ? "dark" : "light";
      frame.style.colorScheme = colorScheme;
      if (doc && doc.documentElement) {
        doc.documentElement.dataset.scheme = scheme;
        doc.documentElement.style.colorScheme = colorScheme;
        if (doc.body) doc.body.style.colorScheme = colorScheme;
      }
    } catch (err) {}
  }
  function inView() {
    var r = frame.getBoundingClientRect();
    return r.height > 0 && r.bottom > 0 && r.top < window.innerHeight;
  }
  frame.addEventListener("load", function () {
    syncScheme();
    setVisible(inView());
  });
  if (window.MutationObserver) {
    new MutationObserver(syncScheme).observe(document.body, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"]
    });
  }
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      setVisible(!!(entries[0] && entries[0].isIntersecting));
    }, { threshold: 0.08, rootMargin: "80px" });
    io.observe(frame);
  }
})();
</script>


The integration of *in-silico* perturbation
screening with spatial context allows us to infer how the combination of gene-gene
interactions rewire signaling to define distinct cellular environments.



<div class="st-betas" markdown="0">
--8<-- "docs/assets/spatial_betas.svg"
</div>

We leverage spatial
transcriptomics data to uncover functional microniches - tissue regions where local
differences in the cellular microenvironment drive divergent cell fates or states.


# I want to 🧞️

<div id="st-i-want-to" class="st-i-want-to" markdown="0"></div>


## Quickstart
#### Install me
```bash
curl -fsSL https://tinyurl.com/spacetravlr/scripts/install.sh | sh
```
#### Run me
```bash
spacetravlr --h5ad /path/to/adata.h5ad --output-dir /path/to/outputdir
```

#### Join me
```bash
spacetravlr --join-output-dir  /path/to/outputdir --plain
```
For example, this slurm job will coordinate multiple workers on the same output/adata
```
#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --job-name=SpaceTravLR
#SBATCH --mem=300G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00

spacetravlr --join-output-dir /path/to/outputdir --plain
```

#### Analyze me

Here we collect the regulatory landscape by looking at the learned gene-gene interactions across all feather files.

```bash
spacetravlr collect-interactions \
  --run-toml /path/to/outputdir/spacetravlr_run_repro.toml 
```

Here we generate functional microniches from the feather files by applying the Leiden algorithm to the learned beta coefficients directly.

```bash
spacetravlr spacetravlr get-microniches \
  --run-toml /path/to/outputdir/spacetravlr_run_repro.toml 
```

<!-- Learn more about [how SpaceTravLR works](math.md), other [installations](install.md) details and CLI [usage](usage.md). -->

#### Point an Ai agent at me

SpaceTravLR publishes an [`llms.txt`](llms.txt) index and a self-contained [`llms-full.txt`](llms-full.txt) reference following the [llmstxt.org](https://llmstxt.org/) convention. Give either URL to Claude, Cursor, ChatGPT, or any coding agent and it will know the full capability surface, every CLI flag, the config schema, and the output formats.

```
https://spacetravlr-rust.readthedocs.io/en/latest/llms-full.txt
```


## Training time estimate {#training-time-estimate}

The widget below provides a rough estimate of how long training SpaceTravLR on your dataset will take. This was empirically estimated using multiple runs across datasets on a100, l40s and rtx6k GPUs.

<div id="st-runtime-estimator" class="st-runtime-estimator" markdown="0"></div>

\(s_{\mathrm{gpu}}\) is a GPU model specific coefficient representing the speedup relative to only using the CPU.

`cnn_max_cells` allows the CNN to smartly subsample the dataset as the training converges. Higher values means later epochs see fewer and fewer cells, focusing on tissue region where the residual errors are higher.