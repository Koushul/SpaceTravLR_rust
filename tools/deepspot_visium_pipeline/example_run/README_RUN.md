# Example DeepSpot → `.h5ad` run (this workspace)

## Saved output

| File | Description |
|------|-------------|
| `deepspot_ZEN38_dummy_fm.h5ad` | AnnData `(24 spots × 5000 genes)` from DeepSpot **Colon_HEST1K** head + **random** morphology features (see below). |

**Input image:** `ZEN38_without_fud.jpg` from the upstream DeepSpot repo (`example_data/data/image/`).

## Why “dummy FM”?

Full inference needs a pathology foundation model on disk (UNI, H-optimus-0, Phikon, …). Both Zenodo DeepSpot weights and those models were prepared for this run:

- **Zenodo** `DeepSpot_pretrained_model_weights.zip` (~1.5 GiB) — downloaded and Colon_HEST1K unzipped under `example_run/DeepSpot_pretrained_model_weights/`.
- **Hugging Face** UNI / H-optimus-0 checkpoints are **gated**; this environment had **no `HF_TOKEN`**, so real FM weights could not be fetched.

The script `demo_dummy_fm_inference.py` therefore feeds **random tensors** of the correct dimensionality into the pretrained DeepSpot regression head. This produces a valid `.h5ad` for plumbing tests (SpaceTravLR config, layers, `obsm['spatial']`) but **not** for biology.

## How this was generated

```bash
cd tools/deepspot_visium_pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision timm transformers huggingface_hub scanpy anndata squidpy \
  pandas numpy scipy pyyaml tqdm pillow lightning pyvips  # pyvips optional if libvips present
pip install lightning
# DeepSpot API on PYTHONPATH (pip install from git may fail on setuptools; clone instead):
#   git clone https://github.com/ratschlab/DeepSpot /tmp/DeepSpot

PYTHONPATH=/tmp/DeepSpot python demo_dummy_fm_inference.py \
  --weights-dir example_run/DeepSpot_pretrained_model_weights/Colon_HEST1K \
  --image /path/to/ZEN38_without_fud.jpg \
  --out-h5ad example_run/deepspot_ZEN38_dummy_fm.h5ad \
  --max-spots 24 \
  --white-cutoff 1000
```

`--white-cutoff 1000` disables the “near-white tile” filter for this small corner grid (all default grid tiles were bright).

## Real inference

1. Accept the license on Hugging Face and `huggingface-cli login` (or set `HF_TOKEN`).
2. Download the FM weights referenced in `top_param_overall.yaml` for your chosen Zenodo bundle.
3. Run `visium_he_to_h5ad.py` (requires **libvips** / working `pyvips` for full parity with DeepSpot).

```bash
PYTHONPATH=/tmp/DeepSpot python visium_he_to_h5ad.py \
  --mode grid \
  --image ZEN38_without_fud.jpg \
  --weights-dir example_run/DeepSpot_pretrained_model_weights/Colon_HEST1K \
  --foundation-weights /path/to/uni/pytorch_model.bin \
  --out-h5ad virtual_colon_real_fm.h5ad \
  --skip-squidpy-image
```

Large downloads not committed to git: `DeepSpot_pretrained_model_weights.zip` (keep locally or re-download from Zenodo).
