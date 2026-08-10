# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="medium",
    app_title="Slide-seq + BANKSY: spatial clustering & interactive UMAP",
)


@app.cell
def _():
    import io
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    IN_BROWSER = sys.platform == "emscripten"
    DATA_REL = "slideseq_banksy"
    return DATA_REL, IN_BROWSER, Path, io, json, mo, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
# Slide-seq mouse brain + BANKSY

This notebook runs a **BANKSY** spatial clustering workflow on Squidpy's
**Slide-seq V2 mouse hippocampus** dataset:

1. Load `squidpy.datasets.slideseqv2()`
2. Build the BANKSY neighbour-augmented expression matrix (`λ`)
3. Leiden-cluster in BANKSY PCA space
4. Explore **spatial vs UMAP** with interactive `λ`, `n_neighbors`, and `min_dist`

Demo assets under `public/slideseq_banksy/` were prepared with Squidpy +
`pybanksy` so the hosted WASM notebook stays interactive without those packages
in the browser.
"""
    )
    return


@app.cell
async def _(DATA_REL, IN_BROWSER, Path, io, json, mo, np):
    async def fetch_bytes(name: str) -> bytes:
        rel = f"{DATA_REL}/{name}"
        errors = []

        if IN_BROWSER:
            from pyodide.http import pyfetch

            urls = []
            loc = mo.notebook_location()
            if loc is not None:
                loc_str = str(loc).rstrip("/")
                if loc_str.startswith(("http://", "https://")):
                    urls.append(f"{loc_str}/public/{rel}")
            urls.extend([f"./public/{rel}", f"public/{rel}", f"/public/{rel}"])
            for url in urls:
                try:
                    resp = await pyfetch(url)
                    if getattr(resp, "ok", True) is False:
                        errors.append(f"{url}: HTTP {getattr(resp, 'status', '?')}")
                        continue
                    data = await resp.bytes()
                    if data:
                        return bytes(data)
                    errors.append(f"{url}: empty body")
                except Exception as exc:
                    errors.append(f"{url}: {exc}")
            raise FileNotFoundError(
                f"Could not fetch public/{rel} in WASM ({'; '.join(errors[-5:])})"
            )

        candidates = []
        notebook_dir = mo.notebook_dir()
        if notebook_dir is not None:
            candidates.append(Path(notebook_dir) / "public" / rel)
        loc = mo.notebook_location()
        if loc is not None and not str(loc).startswith(("http://", "https://")):
            candidates.append(Path(str(loc)) / "public" / rel)
        candidates.extend(
            [
                Path("public") / rel,
                Path.cwd() / "public" / rel,
                Path.cwd() / "docs/tutorials/marimo/public" / rel,
            ]
        )
        for path in candidates:
            if path.is_file():
                return path.read_bytes()
            errors.append(f"missing {path}")
        raise FileNotFoundError(
            f"Could not load public/{rel}. Run prepare_slideseq_banksy.py first. "
            f"({'; '.join(errors[-5:])})"
        )

    def hex_to_rgb(hex_color: str):
        h = hex_color.lstrip("#")
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    meta = json.loads((await fetch_bytes("meta.json")).decode("utf-8"))
    LAMBDA_OPTS = [float(x) for x in meta["lambda_opts"]]
    N_NEIGHBORS_OPTS = list(meta["n_neighbors_opts"])
    MIN_DIST_OPTS = list(meta["min_dist_opts"])
    bundle = {
        "meta": meta,
        "spatial": np.load(io.BytesIO(await fetch_bytes("spatial.npy")), allow_pickle=False),
        "published_labels": np.load(
            io.BytesIO(await fetch_bytes("published_labels.npy")), allow_pickle=False
        ),
    }
    return (
        LAMBDA_OPTS,
        MIN_DIST_OPTS,
        N_NEIGHBORS_OPTS,
        bundle,
        fetch_bytes,
        hex_to_rgb,
        meta,
    )


@app.cell
def _(IN_BROWSER, meta, mo):
    _where = (
        "WebAssembly (precomputed assets)"
        if IN_BROWSER
        else "local (public/slideseq_banksy)"
    )
    mo.md(
        f"""
## 1. Load Slide-seq V2 (Squidpy)

```python
import squidpy as sq
import scanpy as sc

adata = sq.datasets.slideseqv2()
adata  # {meta["n_obs_full"]} beads × {meta["n_vars"]} HVGs (full)
# demo subset: {meta["n_obs"]} beads
```

Published annotation column: `{meta["cluster_key_published"]}` · example gene: `{meta["gene_example"]}`

This session reads the prepared demo bundle from `public/slideseq_banksy/` ({_where}).
"""
    )
    return


@app.cell
def _(LAMBDA_OPTS, MIN_DIST_OPTS, N_NEIGHBORS_OPTS, meta, mo):
    banksy_lambda = mo.ui.dropdown(
        options={str(v): i for i, v in enumerate(LAMBDA_OPTS)},
        value=str(meta["default_lambda"]),
        label="BANKSY λ",
    )
    color_by = mo.ui.dropdown(
        options={"BANKSY clusters": "banksy", "Published cluster": "published"},
        value="BANKSY clusters",
        label="Color by",
    )
    n_neighbors = mo.ui.dropdown(
        options={str(v): i for i, v in enumerate(N_NEIGHBORS_OPTS)},
        value="15",
        label="UMAP n_neighbors",
    )
    min_dist = mo.ui.dropdown(
        options={str(v): i for i, v in enumerate(MIN_DIST_OPTS)},
        value="0.3",
        label="UMAP min_dist",
    )
    mo.hstack(
        [banksy_lambda, color_by, n_neighbors, min_dist], justify="start", gap=1
    )
    return banksy_lambda, color_by, min_dist, n_neighbors


@app.cell
async def _(
    LAMBDA_OPTS,
    MIN_DIST_OPTS,
    N_NEIGHBORS_OPTS,
    banksy_lambda,
    bundle,
    fetch_bytes,
    io,
    min_dist,
    n_neighbors,
    np,
):
    lam = LAMBDA_OPTS[banksy_lambda.value]
    nn = N_NEIGHBORS_OPTS[n_neighbors.value]
    md = MIN_DIST_OPTS[min_dist.value]
    banksy_labels = np.load(
        io.BytesIO(await fetch_bytes(f"banksy/labels_lambda{lam}.npy")),
        allow_pickle=False,
    )
    umap_xy = np.load(
        io.BytesIO(await fetch_bytes(f"umap/lambda{lam}/nn{nn}_md{md}.npy")),
        allow_pickle=False,
    )
    view = {
        "lambda": lam,
        "n_neighbors": nn,
        "min_dist": md,
        "banksy_labels": banksy_labels,
        "umap": umap_xy,
        "spatial": bundle["spatial"],
        "published_labels": bundle["published_labels"],
    }
    return lam, view


@app.cell
def _(bundle, color_by, hex_to_rgb, lam, meta, mo, np, plt, view):
    if color_by.value == "banksy":
        labels = np.asarray([str(int(x)) for x in view["banksy_labels"]])
        palette = meta["banksy"][str(lam)]["palette"]
        legend_title = f"BANKSY λ={lam}"
    else:
        labels = np.asarray([str(x) for x in view["published_labels"]])
        palette = meta["published_palette"]
        legend_title = "published cluster"

    colors = np.array([hex_to_rgb(str(palette[str(lab)])) for lab in labels])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), layout="constrained")
    sp = view["spatial"]
    axes[0].scatter(sp[:, 0], sp[:, 1], c=colors, s=4, linewidths=0, alpha=0.9)
    axes[0].set_title(f"Spatial · {legend_title}")
    axes[0].set_aspect("equal")
    axes[0].invert_yaxis()
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    um = view["umap"]
    axes[1].scatter(um[:, 0], um[:, 1], c=colors, s=4, linewidths=0, alpha=0.9)
    axes[1].set_title(
        f"UMAP on BANKSY PCA · n_neighbors={view['n_neighbors']} · "
        f"min_dist={view['min_dist']}"
    )
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlabel("UMAP1")
    axes[1].set_ylabel("UMAP2")

    categories = sorted(set(labels), key=lambda x: (len(x), x))
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette[cat],
            markersize=6,
            label=cat,
        )
        for cat in categories
        if cat in palette
    ]
    axes[1].legend(
        handles=handles,
        title=legend_title,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=6,
        title_fontsize=7,
        frameon=False,
        ncol=1,
    )

    fig.suptitle(
        "Slide-seq V2 mouse hippocampus — BANKSY clusters vs embedding",
        fontsize=12,
    )
    mo.mpl.interactive(fig)
    return


@app.cell
def _(lam, meta, mo, view):
    mo.md(
        f"""
## 2. BANKSY + UMAP controls

| Control | Current | Role |
|---|---:|---|
| BANKSY `λ` | **{lam}** | Neighbourhood contribution in the BANKSY matrix. Higher `λ` → more spatial context; lower → closer to non-spatial expression. |
| `n_neighbors` | **{view["n_neighbors"]}** | kNN size for UMAP on BANKSY PCA. |
| `min_dist` | **{view["min_dist"]}** | UMAP packing. Smaller → tighter clumps. |
| BANKSY resolution | **{meta["banksy_resolution"]}** | Leiden resolution used when preparing clusters (fixed in this demo). |

Equivalent code (prep-time; uses [`pybanksy`](https://github.com/prabhakarlab/Banksy_py)):

```python
from banksy.initialize_banksy import initialize_banksy
from banksy.run_banksy import generate_banksy_matrix, pca_umap, run_Leiden_partition

banksy_dict = initialize_banksy(
    adata, coord_keys=("x", "y", "spatial"),
    num_neighbours=15, nbr_weight_decay="scaled_gaussian", max_m=1,
)
banksy_dict, _ = generate_banksy_matrix(adata, banksy_dict, [{lam}], max_m=1)
pca_umap(banksy_dict, pca_dims=[20])
results_df, _ = run_Leiden_partition(
    banksy_dict, resolutions=[{meta["banksy_resolution"]}], num_nn=50,
)
labels = results_df.iloc[0]["labels"].dense
```
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
## 3. Next steps

```python
import squidpy as sq

sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)
sq.gr.nhood_enrichment(adata, cluster_key="banksy")
sq.pl.nhood_enrichment(adata, cluster_key="banksy")
```

Try comparing BANKSY labels to the published `cluster` annotations, then
neighbourhood enrichment on the BANKSY domains.
"""
    )
    return


if __name__ == "__main__":
    app.run()
