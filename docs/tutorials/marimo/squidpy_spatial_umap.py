# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scanpy; sys_platform != 'emscripten'",
#     "squidpy; sys_platform != 'emscripten'",
#     "umap-learn; sys_platform != 'emscripten'",
#     "anndata; sys_platform != 'emscripten'",
# ]
#
# [tool.marimo.runtime]
# cache_cells = true
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium", app_title="Squidpy intro: spatial + interactive UMAP")


@app.cell
def _():
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    IN_BROWSER = sys.platform == "emscripten"
    N_NEIGHBORS_OPTS = [5, 15, 30, 50]
    MIN_DIST_OPTS = [0.05, 0.1, 0.3, 0.5, 0.8]
    return IN_BROWSER, MIN_DIST_OPTS, N_NEIGHBORS_OPTS, mo, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
# Squidpy intro: load spatial data & explore UMAP

This notebook walks through a **Squidpy** starter workflow:

1. Load a Visium H&E mouse brain section (`squidpy.datasets.visium_hne_adata`)
2. Inspect spatial coordinates and cluster labels
3. Recompute **UMAP** while you change `n_neighbors` and `min_dist`
4. Compare the **tissue map** next to the **UMAP embedding**

In the browser build, Squidpy/Scanpy run once at export time; UMAP results for
each parameter combo are cached so sliders stay interactive without a Python
server.
"""
    )
    return


@app.cell
def _(IN_BROWSER, mo):
    if IN_BROWSER:
        _msg = mo.md(
            "*Running in WebAssembly — using cached Visium embeddings. "
            "Parameter changes swap precomputed UMAPs instantly.*"
        )
    else:
        _msg = mo.md(
            "*Running locally — Squidpy will download/load the Visium dataset "
            "and recompute UMAP for the parameter grid.*"
        )
    _msg
    return


@app.cell
def _(IN_BROWSER, np, plt):
    def hex_to_rgb(hex_color: str):
        h = hex_color.lstrip("#")
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    def load_visium_bundle():
        """Load spatial coords, clusters, PCA, and palette from Squidpy Visium."""
        if IN_BROWSER:
            raise RuntimeError("Native Squidpy load is unavailable in the browser")

        import scanpy as sc
        import squidpy as sq

        adata = sq.datasets.visium_hne_adata()
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata)

        cluster_key = "cluster" if "cluster" in adata.obs else "leiden"
        labels = np.asarray(adata.obs[cluster_key].astype(str).to_numpy(), dtype=object)
        categories = [
            str(c) for c in adata.obs[cluster_key].astype("category").cat.categories
        ]
        color_key = f"{cluster_key}_colors"
        if color_key in adata.uns:
            palette = {
                cat: str(adata.uns[color_key][i % len(adata.uns[color_key])])
                for i, cat in enumerate(categories)
            }
        else:
            cmap = plt.get_cmap("tab20")
            palette = {
                cat: "#%02x%02x%02x"
                % tuple(int(255 * c) for c in cmap(i % 20)[:3])
                for i, cat in enumerate(categories)
            }

        return {
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "cluster_key": str(cluster_key),
            "spatial": np.asarray(adata.obsm["spatial"], dtype=np.float64).copy(),
            "pca": np.asarray(adata.obsm["X_pca"][:, :30], dtype=np.float64).copy(),
            "labels": labels,
            "categories": categories,
            "palette": palette,
            "gene_example": "Sox2"
            if "Sox2" in adata.var_names
            else str(adata.var_names[0]),
        }

    return load_visium_bundle, hex_to_rgb


@app.cell
def _(IN_BROWSER, load_visium_bundle, mo):
    @mo.persistent_cache(method="lazy")
    def visium_bundle():
        return load_visium_bundle()

    if not IN_BROWSER:
        bundle = visium_bundle()
    else:
        bundle = visium_bundle()
    return bundle, visium_bundle


@app.cell
def _(bundle, mo):
    mo.md(
        f"""
## 1. Load with Squidpy

```python
import squidpy as sq
import scanpy as sc

adata = sq.datasets.visium_hne_adata()
adata  # {bundle["n_obs"]} spots × {bundle["n_vars"]} genes
```

Cluster column: `{bundle["cluster_key"]}` · example gene: `{bundle["gene_example"]}`
"""
    )
    return


@app.cell
def _(MIN_DIST_OPTS, N_NEIGHBORS_OPTS, mo):
    n_neighbors = mo.ui.dropdown(
        options={str(v): i for i, v in enumerate(N_NEIGHBORS_OPTS)},
        value="15",
        label="n_neighbors",
    )
    min_dist = mo.ui.dropdown(
        options={str(v): i for i, v in enumerate(MIN_DIST_OPTS)},
        value="0.3",
        label="min_dist",
    )
    mo.hstack([n_neighbors, min_dist], justify="start", gap=1)
    return min_dist, n_neighbors


@app.cell
def _(IN_BROWSER, MIN_DIST_OPTS, N_NEIGHBORS_OPTS, bundle, mo, np):
    @mo.persistent_cache(method="lazy")
    def compute_umap(n_neighbors_idx: int, min_dist_idx: int):
        import scanpy as sc
        from anndata import AnnData

        n_neighbors = N_NEIGHBORS_OPTS[n_neighbors_idx]
        min_dist = MIN_DIST_OPTS[min_dist_idx]
        ad = AnnData(np.zeros((bundle["pca"].shape[0], 1)))
        ad.obsm["X_pca"] = bundle["pca"]
        sc.pp.neighbors(ad, n_neighbors=n_neighbors, use_rep="X_pca")
        sc.tl.umap(ad, min_dist=min_dist)
        return {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "umap": np.asarray(ad.obsm["X_umap"], dtype=np.float64),
        }

    if not IN_BROWSER:
        for _nn_i in range(len(N_NEIGHBORS_OPTS)):
            for _md_i in range(len(MIN_DIST_OPTS)):
                compute_umap(_nn_i, _md_i)
    return compute_umap,


@app.cell
def _(compute_umap, min_dist, n_neighbors):
    umap_result = compute_umap(n_neighbors.value, min_dist.value)
    return umap_result,


@app.cell
def _(hex_to_rgb, bundle, mo, np, plt, umap_result):
    labels = bundle["labels"]
    palette = bundle["palette"]
    colors = np.array([hex_to_rgb(palette[lab]) for lab in labels])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), layout="constrained")

    sp = bundle["spatial"]
    axes[0].scatter(sp[:, 0], sp[:, 1], c=colors, s=8, linewidths=0, alpha=0.9)
    axes[0].set_title(f"Spatial ({bundle['cluster_key']})")
    axes[0].set_aspect("equal")
    axes[0].invert_yaxis()
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    um = umap_result["umap"]
    axes[1].scatter(um[:, 0], um[:, 1], c=colors, s=8, linewidths=0, alpha=0.9)
    axes[1].set_title(
        f"UMAP · n_neighbors={umap_result['n_neighbors']} · "
        f"min_dist={umap_result['min_dist']}"
    )
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlabel("UMAP1")
    axes[1].set_ylabel("UMAP2")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette[cat],
            markersize=7,
            label=cat,
        )
        for cat in bundle["categories"]
    ]
    axes[1].legend(
        handles=handles,
        title=bundle["cluster_key"],
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        title_fontsize=8,
        frameon=False,
    )

    fig.suptitle("Visium H&E — tissue vs transcriptome neighborhood", fontsize=12)
    mo.mpl.interactive(fig)
    return


@app.cell
def _(bundle, mo, umap_result):
    _ck = bundle["cluster_key"]
    mo.md(
        f"""
## 2. What the controls do

| Parameter | Current | Effect |
|---|---:|---|
| `n_neighbors` | **{umap_result["n_neighbors"]}** | Local neighborhood size for the kNN graph. Smaller → more fragmented clusters; larger → smoother global structure. |
| `min_dist` | **{umap_result["min_dist"]}** | How tightly UMAP packs points. Smaller → denser clumps; larger → more spread out. |

Equivalent Scanpy/Squidpy code:

```python
sc.pp.neighbors(adata, n_neighbors={umap_result["n_neighbors"]}, use_rep="X_pca")
sc.tl.umap(adata, min_dist={umap_result["min_dist"]})
sq.pl.spatial_scatter(adata, color="{_ck}")
sc.pl.umap(adata, color="{_ck}")
```
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
## 3. Next steps with Squidpy

```python
import squidpy as sq

sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=6)
sq.gr.nhood_enrichment(adata, cluster_key="cluster")
sq.pl.nhood_enrichment(adata, cluster_key="cluster")

sq.gr.co_occurrence(adata, cluster_key="cluster")
sq.pl.co_occurrence(adata, cluster_key="cluster", clusters="Hippocampus")
```

Try neighborhood enrichment and co-occurrence next — they use the same spatial
graph intuition as the UMAP neighborhood size above.
"""
    )
    return


if __name__ == "__main__":
    app.run()
