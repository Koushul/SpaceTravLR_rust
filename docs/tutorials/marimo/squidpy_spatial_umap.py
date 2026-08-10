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
app = marimo.App(width="medium", app_title="Squidpy intro: spatial + interactive UMAP")


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
    DATA_REL = "visium_demo"
    return DATA_REL, IN_BROWSER, Path, io, json, mo, np, plt


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

Demo assets under `public/visium_demo/` were prepared with Squidpy/Scanpy so the
hosted WASM notebook stays interactive without those packages in the browser.
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
            urls.extend(
                [
                    f"./public/{rel}",
                    f"public/{rel}",
                    f"/public/{rel}",
                ]
            )
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
            f"Could not load public/{rel}. Run prepare_visium_demo.py first. "
            f"({'; '.join(errors[-5:])})"
        )

    def hex_to_rgb(hex_color: str):
        h = hex_color.lstrip("#")
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    meta = json.loads((await fetch_bytes("meta.json")).decode("utf-8"))
    N_NEIGHBORS_OPTS = list(meta["n_neighbors_opts"])
    MIN_DIST_OPTS = list(meta["min_dist_opts"])
    bundle = {
        "n_obs": int(meta["n_obs"]),
        "n_vars": int(meta["n_vars"]),
        "cluster_key": str(meta["cluster_key"]),
        "categories": list(meta["categories"]),
        "palette": dict(meta["palette"]),
        "gene_example": str(meta["gene_example"]),
        "spatial": np.load(io.BytesIO(await fetch_bytes("spatial.npy")), allow_pickle=False),
        "labels": np.load(io.BytesIO(await fetch_bytes("labels.npy")), allow_pickle=False),
    }
    return MIN_DIST_OPTS, N_NEIGHBORS_OPTS, bundle, fetch_bytes, hex_to_rgb


@app.cell
def _(IN_BROWSER, bundle, mo):
    _where = (
        "WebAssembly (precomputed assets)"
        if IN_BROWSER
        else "local (public/visium_demo)"
    )
    mo.md(
        f"""
## 1. Load with Squidpy

```python
import squidpy as sq
import scanpy as sc

adata = sq.datasets.visium_hne_adata()
adata  # {bundle["n_obs"]} spots × {bundle["n_vars"]} genes
```

This session is reading the prepared demo bundle from `public/visium_demo/`
({_where}).

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
async def _(MIN_DIST_OPTS, N_NEIGHBORS_OPTS, fetch_bytes, io, min_dist, n_neighbors, np):
    nn = N_NEIGHBORS_OPTS[n_neighbors.value]
    md = MIN_DIST_OPTS[min_dist.value]
    umap_result = {
        "n_neighbors": nn,
        "min_dist": md,
        "umap": np.load(
            io.BytesIO(await fetch_bytes(f"umap/nn{nn}_md{md}.npy")),
            allow_pickle=False,
        ),
    }
    return umap_result,


@app.cell
def _(bundle, hex_to_rgb, mo, np, plt, umap_result):
    labels = bundle["labels"]
    palette = bundle["palette"]
    colors = np.array([hex_to_rgb(str(palette[str(lab)])) for lab in labels])

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
