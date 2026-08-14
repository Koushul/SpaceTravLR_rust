"""IL21 KO/OE parameter sweeps using the Rust transition backend."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pyarrow.feather as feather

from . import compute_transition_grid
from .plot import plot_quiver_side_by_side
from .umap_embed import ensure_umap_embedding


def _dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float64)


def _load_feather(path: Path, obs_names, var_names) -> np.ndarray:
    df = feather.read_table(path).to_pandas()
    if "CellID" in df.columns:
        df = df.set_index("CellID")
    df = df.reindex(index=obs_names, columns=var_names)
    if df.isna().any().any():
        raise ValueError(f"NA after align for {path}")
    return df.to_numpy(dtype=np.float64)


def run_il21_parameter_sweep(
    *,
    h5ad: str | Path,
    pert_dir: str | Path,
    out_dir: str | Path,
    arms: tuple[str, ...] = ("spatial", "meanfield"),
    n_neighbors_list: tuple[int, ...] = (50, 100, 150),
    grid_scale_list: tuple[float, ...] = (0.75, 1.0, 1.5),
    vector_scale_list: tuple[float, ...] = (0.85,),
    unit_directions_list: tuple[bool, ...] = (False, True),
    null_modes: tuple[str, ...] = ("raw", "clip_renorm"),
    umap_n_neighbors_list: tuple[int, ...] = (15, 30),
    umap_min_dist_list: tuple[float, ...] = (0.3, 0.5),
    prefer_rust_umap: bool = False,
) -> list[Path]:
    """Compute side-by-side KO|OE quivers across parameter combinations.

    UMAP settings are applied first (one embedding per umap_* combo), then
    transition fields for each arm × transition knobs.
    """
    h5ad = Path(h5ad)
    pert_dir = Path(pert_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = ad.read_h5ad(h5ad)
    expr0 = _dense(base.X)
    cell_types = base.obs["cell_type"].astype(str).values
    written: list[Path] = []

    for unn, umd in itertools.product(umap_n_neighbors_list, umap_min_dist_list):
        adata = base.copy()
        ensure_umap_embedding(
            adata,
            prefer_rust=prefer_rust_umap,
            n_neighbors=unn,
            min_dist=umd,
            force=True,
        )
        umap = np.asarray(adata.obsm["X_umap"], dtype=np.float64)
        expr = _dense(adata.X)

        for arm in arms:
            ko = _load_feather(
                pert_dir / arm / "IL21_KO.feather", adata.obs_names, adata.var_names
            )
            oe = _load_feather(
                pert_dir / arm / "IL21_OE.feather", adata.obs_names, adata.var_names
            )
            delta_ko = np.round(ko - expr, 3)
            delta_oe = np.round(oe - expr, 3)

            for nn, gs, vs, unit, nmode in itertools.product(
                n_neighbors_list,
                grid_scale_list,
                vector_scale_list,
                unit_directions_list,
                null_modes,
            ):
                kw: dict[str, Any] = dict(
                    n_neighbors=nn,
                    grid_scale=gs,
                    vector_scale=vs,
                    unit_directions=unit,
                    null_subtract_mode=nmode,
                    remove_null=True,
                    temperature=0.05,
                    delta_rescale=1.0,
                )
                g_ko = compute_transition_grid(expr, delta_ko, umap, **kw)
                g_oe = compute_transition_grid(expr, delta_oe, umap, **kw)
                tag = (
                    f"{arm}_umap{unn}_md{umd}_nn{nn}_gs{gs}_vs{vs}"
                    f"_unit{int(unit)}_{nmode}"
                )
                out = out_dir / f"IL21_KO_OE_quiver_{tag}.png"
                plot_quiver_side_by_side(
                    umap,
                    cell_types,
                    [
                        (f"{arm} · IL21 KO\nnn={nn} gs={gs} unit={unit} {nmode}", g_ko),
                        (f"{arm} · IL21 OE\nnn={nn} gs={gs} unit={unit} {nmode}", g_oe),
                    ],
                    out_path=str(out),
                    suptitle=f"UMAP n_neighbors={unn} min_dist={umd}",
                )
                written.append(out)
                print(f"wrote {out}")

    # compact index
    index = out_dir / "sweep_index.txt"
    index.write_text("\n".join(str(p) for p in written) + "\n")
    return written


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--h5ad",
        default="/Users/koush/Projects/SpaceTravLR_rust/data/h5ad/SlideTags_human_tonsil.h5ad",
    )
    ap.add_argument(
        "--pert-dir",
        default="/tmp/tonsil_full_seed_20260805/perturbations",
    )
    ap.add_argument(
        "--out-dir",
        default="/tmp/tonsil_full_seed_20260805/perturbations/sweep_rust",
    )
    ap.add_argument("--quick", action="store_true", help="Smaller sweep for smoke tests")
    args = ap.parse_args()
    if args.quick:
        run_il21_parameter_sweep(
            h5ad=args.h5ad,
            pert_dir=args.pert_dir,
            out_dir=args.out_dir,
            n_neighbors_list=(80, 150),
            grid_scale_list=(1.0,),
            unit_directions_list=(False,),
            null_modes=("raw",),
            umap_n_neighbors_list=(15,),
            umap_min_dist_list=(0.5,),
            prefer_rust_umap=False,
        )
    else:
        run_il21_parameter_sweep(
            h5ad=args.h5ad,
            pert_dir=args.pert_dir,
            out_dir=args.out_dir,
        )
