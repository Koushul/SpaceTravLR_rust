# Example DeepSpot → `.h5ad` runs

## Full inference (real ViT tiles + Colon DeepSpot head)

After `hf auth login`, **you must still click “Agree”** on each gated model page (e.g.
[Maho­odLab/UNI](https://huggingface.co/MahmoodLab/UNI)) — login alone returns **403**
until the license is accepted.

### What we ran here (no UNI download — HF access pending)

This machine did **not** have `libvips`, and Hugging Face returned **access denied** for `MahmoodLab/UNI`.  
We therefore ran **`visium_he_to_h5ad.py`** with:

- **`--foundation-timm-imagenet`** — ImageNet **ViT-L/14** via timm (same architecture as UNI, **not** pathology-trained weights).
- **`--pillow`** — Pillow-based tiling (no pyvips).

Output:

| File | Description |
|------|-------------|
| **`deepspot_ZEN38_timm_imagenet.h5ad`** | **32 spots × 5000 genes**, real forward pass through tiles + Colon_Zenodo head; `uns['morphology_note']` warns about backbone. |

Command:

```bash
cd tools/deepspot_visium_pipeline
source .venv/bin/activate
export PYTHONPATH=/path/to/DeepSpot   # clone github.com/ratschlab/DeepSpot

python visium_he_to_h5ad.py \
  --mode grid \
  --image /path/to/ZEN38_without_fud.jpg \
  --weights-dir example_run/DeepSpot_pretrained_model_weights/Colon_HEST1K \
  --foundation-timm-imagenet \
  --pillow \
  --max-spots 32 \
  --white-cutoff 1000 \
  --out-h5ad example_run/deepspot_ZEN38_timm_imagenet.h5ad \
  --skip-squidpy-image
```

### After you approve UNI on Hugging Face

1. Download UNI:

   ```bash
   hf download MahmoodLab/UNI pytorch_model.bin --local-dir ./uni_ckpt
   ```

2. Prefer **pyvips** for large slides (`conda install -c conda-forge pyvips`). Otherwise keep **`--pillow`** for smaller exports.

3. Run **without** `--foundation-timm-imagenet`:

   ```bash
   python visium_he_to_h5ad.py \
     --mode grid \
     --image ZEN38_without_fud.jpg \
     --weights-dir DeepSpot_pretrained_model_weights/Colon_HEST1K \
     --foundation-weights ./uni_ckpt/pytorch_model.bin \
     --pillow \
     --max-spots 32 \
     --white-cutoff 1000 \
     --out-h5ad deepspot_ZEN38_uni_official.h5ad \
     --skip-squidpy-image
   ```

---

## Dummy-only run (random FM features)

See commit history / `deepspot_ZEN38_dummy_fm.h5ad` and `demo_dummy_fm_inference.py` — uses **random** morphology tensors (plumbing test only).
