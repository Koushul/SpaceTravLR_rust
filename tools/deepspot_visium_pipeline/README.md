# DeepSpot: Visium H&E → virtual expression `.h5ad`

This folder wraps the official [DeepSpot](https://github.com/ratschlab/DeepSpot) inference
workflow so you can go from **10x Visium-aligned H&E** (or a plain H&E for the synthetic grid
demo) to an **`AnnData` `.h5ad`** that matches the SpaceTravLR conventions described in
`deepspot_spacetravlr_summary.md`.

## What you get

- **`adata.layers["imputed_count"]`** and **`adata.X`**: DeepSpot predicted expression (normalized / log-like scale as in the paper; negative predictions clipped to 0).
- **`adata.obsm["spatial"]`**: **full-resolution pixel coordinates** as `(y_pixel, x_pixel)` (same convention as Squidpy’s `spatial` scatter examples).
- **`adata.obs["cell_type"]`**: placeholder `"spot"` unless you pass **`--cluster-cell-type`** (PCA + Leiden on predictions).

Optional visualization helpers:

- **`adata.obsm["spatial_hires"]`** + **`adata.uns["spatial"]["library_id"]["images"]["hires"]`**: downsampled RGB preview aligned to `spatial_hires` (pass **`--skip-squidpy-image`** to skip).

## Important caveat: “whole transcriptome”

Pretrained DeepSpot checkpoints ship a **fixed highly-variable gene panel** (often ~5k genes;
see `info_highly_variable_genes.csv` inside each weight bundle). That is the paper’s typical
“whole-slide transcriptomics” usage. Predicting **every gene** usually requires **training**
or **atlas projection / DeepCell** workflows from the DeepSpot publications—not this
one-command inference path.

## Setup

1. Clone DeepSpot (optional; `pip` below installs it):

   ```bash
   git clone https://github.com/ratschlab/DeepSpot
   ```

2. Create an environment (reuse DeepSpot’s `environment.yaml` **or** install requirements here):

   ```bash
   cd tools/deepspot_visium_pipeline
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Install **`pyvips`** system libs (DeepSpot README recommends conda):

   ```bash
   conda install -c conda-forge pyvips
   ```

4. Download **DeepSpot pretrained weights** from Zenodo:
   [records/15322099](https://zenodo.org/records/15322099) and unzip.

5. Download the **pathology foundation model** weights referenced by the checkpoint’s
   `top_param_overall.yaml` (`image_feature_model`, usually `uni`, `phikon`, or `hoptimus0`)
   and point `--foundation-weights` at the **local checkpoint file** on disk.

## Example: 10x Visium Space Ranger `spatial/` folder

Assume Space Ranger output contains `spatial/tissue_positions*.csv`, `spatial/scalefactors_json.json`,
and a tissue image (`tissue_fullres_image.tif` / `tissue_hires_image.png`, …).

```bash
python visium_he_to_h5ad.py \
  --mode visium \
  --spatial-dir /path/to/spaceranger/outs/spatial \
  --weights-dir /path/to/DeepSpot_pretrained_model_weights/Colon_HEST1K \
  --foundation-weights /path/to/uni/pytorch_model.bin \
  --out-h5ad ./virtual_visium_deepspot.h5ad \
  --cluster-cell-type
```

If your H&E path differs from the auto-detected image under `spatial/`, pass **`--image`**
explicitly. Coordinates must refer to the **same pixel frame** as that image.

## Example: synthetic square grid (upstream toy notebook pattern)

This mirrors `GettingStartedWithDeepSpot_3.1_inference_pretrained_models.ipynb` when you only
have a flat H&E (`jpg/png/tiff`) and want a grid of virtual spots:

```bash
python visium_he_to_h5ad.py \
  --mode grid \
  --image /path/to/slide.jpg \
  --weights-dir /path/to/DeepSpot_pretrained_model_weights/Colon_HEST1K \
  --foundation-weights /path/to/uni/pytorch_model.bin \
  --out-h5ad ./virtual_grid_deepspot.h5ad
```

Grid spacing defaults come from `spot_diameter` / `spot_distance` in `top_param_overall.yaml`.

## SpaceTravLR next step

Point SpaceTravLR at the generated `.h5ad`, using `layer = "imputed_count"` and the same
spatial units as `obsm["spatial"]` when configuring `[spatial].radius`.

```bash
spacetravlr --plain \
  --config /path/to/spaceship_config.toml \
  --h5ad ./virtual_visium_deepspot.h5ad \
  --output-dir /path/to/spacetravlr_run
```

## Example artifact (CI / plumbing)

If gated Hugging Face weights are unavailable, use **`--foundation-timm-imagenet`** with Colon
(checkpoint must use `image_feature_model: uni`) for an ImageNet ViT-L backbone with the same
architecture as UNI — **not** the pathology FM from the paper. Alternatively see
`example_run/README_RUN.md` and `demo_dummy_fm_inference.py` for a random-FM plumbing test.

**Paired Visium benchmark** (measured RNA vs DeepSpot on the same spots): see
[`example_run/README_ZEN38_PAIRED.md`](example_run/README_ZEN38_PAIRED.md) and `eval_paired_zen38.py`.

### MAGIC + SpaceTravLR seed-beta benchmark

To test whether SpaceTravLR seed betas trained from H&E-imputed expression agree with betas trained
from measured spatial transcriptomics on the same spots, run:

```bash
python compare_spacetravlr_magic_betas.py \
  --paired-h5ad example_run/zen38_paired_uni_official.h5ad \
  --out-dir example_run/spacetravlr_magic_beta_benchmark \
  --force-spatial-bins \
  --spacetravlr-cmd "cargo run --release --bin spacetravlr --"
```

The script selects 10 shared genes by measured-vs-H&E expression correlation unless `--genes` is
provided, applies clusterwise MAGIC to both source layers, trains two matched seed-mode runs with the
same 10 genes as extra modulators, and writes `beta_correlation_summary.json`,
`matched_beta_pairs.csv`, and per-gene beta correlations.

## References

- DeepSpot repository: [github.com/ratschlab/DeepSpot](https://github.com/ratschlab/DeepSpot)
- Tutorial notebooks: `example_notebook/Visium_spot_example/GettingStartedWithDeepSpot_3.1_inference_pretrained_models.ipynb`
- Integration notes: `deepspot_spacetravlr_summary.md` (repo root)
