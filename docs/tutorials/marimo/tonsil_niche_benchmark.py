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
    app_title="Niche method benchmark: tonsil GC B cells",
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
    DATA_REL = "tonsil_niche_benchmark"
    return DATA_REL, IN_BROWSER, Path, io, json, mo, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
# Niche method comparison on tonsil germinal-center B cells

Unsupervised niche comparison of **SpaceTravLR β** vs **BANKSY**, **COVET**, and
**NicheCompass** on human tonsil snRNA-seq germinal-center B cells
(`B_germinal_center`, n≈1848).

GC Light / Dark / Intermediate Zone labels in `cell_type_2` were originally
derived from gene expression, so they are **not** treated as ground truth here.
Primary read-outs are:

- **Spatial consistency:** CAS, MLAMI, CLISIS, GCS
- **Niche coherence:** CNMI, NASW
- **Method agreement:** pairwise ARI between niche clusterings

Metrics follow the NicheCompass single-sample suite used in
[graceful-hollow-a6tk.here.now](https://graceful-hollow-a6tk.here.now/).
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
            f"Could not load public/{rel}. Run prepare_tonsil_niche_benchmark.py first. "
            f"({'; '.join(errors[-5:])})"
        )

    def hex_to_rgb(hex_color: str):
        h = hex_color.lstrip("#")
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    meta = json.loads((await fetch_bytes("meta.json")).decode("utf-8"))
    methods = list(meta["methods"])
    bundle = {
        "meta": meta,
        "spatial": np.load(io.BytesIO(await fetch_bytes("spatial.npy")), allow_pickle=False),
        "zone_labels": np.load(
            io.BytesIO(await fetch_bytes("zone_labels_confounded.npy")), allow_pickle=False
        ),
        "method_data": {},
    }
    for _name in methods:
        bundle["method_data"][_name] = {
            "labels": np.load(
                io.BytesIO(await fetch_bytes(f"methods/{_name}_labels.npy")),
                allow_pickle=False,
            ),
            "umap": np.load(
                io.BytesIO(await fetch_bytes(f"methods/{_name}_umap.npy")),
                allow_pickle=False,
            ),
        }
    return bundle, hex_to_rgb, meta, methods


@app.cell
def _(IN_BROWSER, meta, mo):
    _where = (
        "WebAssembly (precomputed assets)"
        if IN_BROWSER
        else "local (public/tonsil_niche_benchmark)"
    )
    mo.md(
        f"""
## 1. Dataset

Human tonsil snRNA · germinal-center B cells only  
`{meta["n_obs"]}` cells · `{meta["n_vars"]}` genes · seed={meta["seed"]}  
Session source: `{_where}`

SpaceTravLR β uses `{meta["params"]["n_beta_features"]}` spatially filtered β features
(Moran's I / η² / FDR / decorrelated). Expression methods use `raw_count` (`.X` is pre-scaled).
"""
    )
    return


@app.cell
def _(meta, methods, mo):
    display = meta.get("display_names", {})
    method_ui = mo.ui.dropdown(
        options={display.get(m, m): m for m in methods},
        value=display.get("spacetravlr_beta", "spacetravlr_beta"),
        label="Method",
    )
    color_by = mo.ui.dropdown(
        options={
            "Method niches": "method",
            "GC zones (expression-confounded)": "zone",
        },
        value="Method niches",
        label="Color by",
    )
    mo.hstack([method_ui, color_by], justify="start", gap=1)
    return color_by, method_ui


@app.cell
def _(bundle, color_by, hex_to_rgb, meta, method_ui, mo, np, plt):
    selected_method = method_ui.value
    spatial = bundle["spatial"]
    zone = np.asarray([str(x) for x in bundle["zone_labels"]])
    mdata = bundle["method_data"][selected_method]
    method_labels = np.asarray([str(x) for x in mdata["labels"]])
    um = mdata["umap"]
    display = meta.get("display_names", {})
    pretty = display.get(selected_method, selected_method)

    if color_by.value == "method":
        labels = method_labels
        cats = sorted(set(labels), key=lambda x: (len(x), x))
        raw_palette = meta["method_meta"][selected_method]["palette"]
        palette = {
            cat: raw_palette.get(str(i), raw_palette.get(cat, "#999999"))
            for i, cat in enumerate(cats)
        }
        for cat in cats:
            if cat in raw_palette:
                palette[cat] = raw_palette[cat]
        legend_title = pretty
    else:
        labels = zone
        palette = meta.get("zone_palette", {})
        legend_title = "GC zones (confounded)"

    colors = np.array(
        [hex_to_rgb(str(palette.get(str(lab), "#999999"))) for lab in labels]
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(11.0, 4.8), layout="constrained")
    _axes[0].scatter(spatial[:, 0], spatial[:, 1], c=colors, s=6, linewidths=0, alpha=0.9)
    _axes[0].set_title(f"Spatial · {legend_title}")
    _axes[0].set_aspect("equal")
    _axes[0].invert_yaxis()
    _axes[0].set_xticks([])
    _axes[0].set_yticks([])
    _axes[0].set_xlabel("x")
    _axes[0].set_ylabel("y")

    _axes[1].scatter(um[:, 0], um[:, 1], c=colors, s=6, linewidths=0, alpha=0.9)
    _axes[1].set_title(f"UMAP on {pretty} embedding")
    _axes[1].set_xticks([])
    _axes[1].set_yticks([])
    _axes[1].set_xlabel("UMAP1")
    _axes[1].set_ylabel("UMAP2")

    categories = sorted(set(labels), key=lambda x: (len(x), x))
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette.get(cat, "#999999"),
            markersize=6,
            label=str(cat),
        )
        for cat in categories
    ]
    _axes[1].legend(
        handles=handles,
        title=legend_title,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=6,
        title_fontsize=7,
        frameon=False,
        ncol=1,
    )
    _fig.suptitle("Tonsil GC B cells — niche comparison", fontsize=12)
    mo.mpl.interactive(_fig)
    return (selected_method,)


@app.cell
def _(meta, methods, mo, np, plt, selected_method):
    metric_names = [
        n
        for n in meta["metric_names"]
        if n
        not in (
            "ari_vs_zone_confounded",
            "nmi_vs_zone_confounded",
        )
    ]
    groups = {
        k: v
        for k, v in meta["metric_groups"].items()
        if k != "zone_agreement_confounded"
    }
    metrics = meta["metrics"]
    display = meta.get("display_names", {})

    rows = []
    for m in methods:
        row = {"method": m, "n_clusters": meta["method_meta"][m]["n_clusters"]}
        for name in metric_names:
            val = metrics[m].get(name)
            row[name] = float(val) if val is not None else np.nan
        rows.append(row)

    header = "| method | " + " | ".join(metric_names) + " | n_clusters |"
    sep = "|---|---:|" + "---:|" * len(metric_names)
    lines = [header, sep]
    for row in rows:
        cells = [display.get(row["method"], row["method"])]
        for name in metric_names:
            v = row[name]
            cells.append("—" if np.isnan(v) else f"{v:.3f}")
        cells.append(str(row["n_clusters"]))
        lines.append("| " + " | ".join(cells) + " |")

    def fmt(v):
        if v is None:
            return "—"
        try:
            if np.isnan(float(v)):
                return "—"
        except Exception:
            return "—"
        return f"{float(v):.3f}"

    highlight = metrics[selected_method]
    pretty = display.get(selected_method, selected_method)
    mo.md(
        f"""
## 2. Benchmark metrics (primary)

Selected **{pretty}** — CAS={fmt(highlight.get('cas'))},
MLAMI={fmt(highlight.get('mlami'))},
CNMI={fmt(highlight.get('cnmi'))},
NASW={fmt(highlight.get('nasw'))}

"""
        + "\n".join(lines)
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(11.0, 3.8), layout="constrained")
    x = np.arange(len(methods))
    xticklabels = [display.get(m, m) for m in methods]
    for ax, (group_name, names) in zip(_axes, groups.items()):
        width = 0.8 / max(1, len(names))
        for i, name in enumerate(names):
            vals = []
            for m in methods:
                v = metrics[m].get(name)
                vals.append(np.nan if v is None else float(v))
            ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width=width, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(group_name.replace("_", " "))
        ax.legend(fontsize=7, frameon=False)
        ax.set_ylabel("score")
    _fig.suptitle("Unsupervised niche metrics (higher is better)", fontsize=11)
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(meta, methods, mo, np, plt):
    display = meta.get("display_names", {})
    pw = meta.get("pairwise_ari", {})
    mat = np.array(
        [[float(pw[a][b]) for b in methods] for a in methods], dtype=np.float64
    )
    labels = [display.get(m, m) for m in methods]

    _fig, ax = plt.subplots(figsize=(5.2, 4.4), layout="constrained")
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(
                j,
                i,
                f"{mat[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if mat[i, j] < 0.55 else "black",
                fontsize=8,
            )
    _fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ARI")
    ax.set_title("Pairwise niche agreement (ARI)")
    mo.md("## 3. Method–method agreement")
    mo.mpl.interactive(_fig)

    zone_lines = [
        "| method | ARI vs GC zones (confounded) | NMI vs GC zones (confounded) |",
        "|---|---:|---:|",
    ]
    for m in methods:
        ari = meta["metrics"][m].get("ari_vs_zone_confounded")
        nmi = meta["metrics"][m].get("nmi_vs_zone_confounded")
        zone_lines.append(
            f"| {display.get(m, m)} | {float(ari):.3f} | {float(nmi):.3f} |"
        )
    mo.md(
        "GC zone agreement is shown only as a confounded reference "
        "(Light/Dark/Intermediate zones were expression-derived):\n\n"
        + "\n".join(zone_lines)
    )
    return


@app.cell
def _(meta, mo):
    p = meta["params"]
    mo.md(
        f"""
## 4. Prep parameters

| Setting | Value |
|---|---|
| Cells | {meta["n_obs"]} GC B cells (seed={meta["seed"]}) |
| Leiden resolution | {p["leiden_resolution"]} |
| BANKSY λ | {p["banksy_lambda"]} |
| NicheCompass epochs | {p["nichecompass_epochs"]} |
| SpaceTravLR β features | {p["n_beta_features"]} |

Recompute locally:

```bash
source /tmp/nichebench-venv/bin/activate
export HOME=/tmp/fakehome
python docs/tutorials/marimo/prepare_tonsil_niche_benchmark.py
```
"""
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
## 5. How to read this comparison

Higher is better for CAS / MLAMI / CLISIS / GCS / CNMI / NASW.

- **CAS / CLISIS** — preservation of neighbourhood structure between physical space and the embedding (here using LZ/DZ only inside the CAS machinery; treat cautiously).
- **MLAMI / GCS** — unsupervised spatial conservation of the latent neighbourhood graph.
- **CNMI / NASW** — niche coherence / silhouette-like structure in latent space.
- **Pairwise ARI** — whether methods recover *the same* niches (primary agreement metric when ground truth is confounded).
- **ARI vs GC zones** — expected to be low/misleading if zones were defined from expression.
"""
    )
    return


if __name__ == "__main__":
    app.run()
