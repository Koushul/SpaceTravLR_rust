"""
Compare Python vs Rust perturb using saved inputs from Rust test.

Usage:
    cd /path/to/SpaceTravLR
    spacetravlr/bin/python /path/to/SpaceTravLR_rust/compare_perturb.py
"""
import sys, os, glob, time
sys.path.insert(0, os.path.expanduser("~/Projects/jishnulab/SpaceTravLR/src"))

import numpy as np
import pandas as pd
from SpaceTravLR.beta import BetaFrame
from SpaceTravLR.models.parallel_estimators import compute_radius_weights_fast

BETAS_DIR = "/tmp/betas"
INPUTS_DIR = "/tmp/perturb_compare"

# ── Load saved inputs from Rust ──────────────────────────────────────────────

gene_names = [l.strip() for l in open(f"{INPUTS_DIR}/gene_names.csv") if l.strip()]
gene_mtx = np.loadtxt(f"{INPUTS_DIR}/gene_mtx.csv", delimiter=",")
xy = np.loadtxt(f"{INPUTS_DIR}/xy.csv", delimiter=",")

rw_lig_df = pd.read_csv(f"{INPUTS_DIR}/rw_ligands.csv")
try:
    rw_tfl_df = pd.read_csv(f"{INPUTS_DIR}/rw_tfligands.csv")
except pd.errors.EmptyDataError:
    rw_tfl_df = pd.DataFrame(index=range(gene_mtx.shape[0]))

lr_radii = {}
for line in open(f"{INPUTS_DIR}/lr_radii.csv"):
    parts = line.strip().split(",")
    if len(parts) == 2:
        lr_radii[parts[0]] = float(parts[1])

config = {}
for line in open(f"{INPUTS_DIR}/config.txt"):
    k, v = line.strip().split("=", 1)
    config[k] = v

target = config["target"]
gene_expr = float(config["gene_expr"])
n_propagation = int(config["n_propagation"])

n_cells, n_genes = gene_mtx.shape
print(f"Loaded inputs: {n_cells} cells, {n_genes} genes, target={target}->{gene_expr}, {n_propagation} iters")

# ── Load BetaFrames ──────────────────────────────────────────────────────────


def load_betadata_table(path):
    df = pd.read_feather(path)
    return df.set_index(df.columns[0])


beta_paths = sorted(glob.glob(f"{BETAS_DIR}/*_betadata.feather"))
obs_names = [f"cell_{i}" for i in range(n_cells)]
n_clusters = 13
clusters = [i % n_clusters for i in range(n_cells)]

betaframes = {}
for path in beta_paths:
    gene_name = os.path.basename(path).replace("_betadata.feather", "")
    df = load_betadata_table(path)
    new_cols = {c: c if c == "beta0" else f"beta_{c}" for c in df.columns}
    df = df.rename(columns=new_cols)

    expanded = pd.DataFrame(index=obs_names, columns=df.columns, dtype=float)
    for i, cell in enumerate(obs_names):
        expanded.iloc[i] = df.iloc[clusters[i] % len(df)].values
    expanded = expanded.fillna(0.0).astype(float)
    expanded.index.name = gene_name
    bf = BetaFrame(expanded)
    bf.modulator_gene_indices = [
        gene_names.index(g.replace("beta_", ""))
        for g in bf.modulators_genes
        if g.replace("beta_", "") in gene_names
    ]
    betaframes[gene_name] = bf

print(f"Loaded {len(betaframes)} BetaFrames")

# ── Collect ligand sets ──────────────────────────────────────────────────────

ligands_set = set()
tfl_ligands_set = set()
for bf in betaframes.values():
    ligands_set.update(bf._ligands)
    tfl_ligands_set.update(bf._tfl_ligands)

all_ligands = sorted(set(ligands_set) | set(tfl_ligands_set))

gene_to_idx = {g: i for i, g in enumerate(gene_names)}

# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_weighted_ligands_np(gene_mtx_cur, ligand_names, min_expression=1e-9):
    """Compute received ligands using Gaussian kernel."""
    unique_ligs = sorted(set(l for l in ligand_names if l in gene_to_idx))
    if not unique_ligs:
        return pd.DataFrame(index=obs_names)

    # Extract and filter ligand expression
    lig_data = np.zeros((n_cells, len(unique_ligs)))
    for j, lig in enumerate(unique_ligs):
        col = gene_mtx_cur[:, gene_to_idx[lig]]
        lig_data[:, j] = np.where(col > min_expression, col, 0.0)

    lig_df = pd.DataFrame(lig_data, index=obs_names, columns=unique_ligs)

    # Group by radius and compute
    radius_groups = {}
    for lig in unique_ligs:
        r = lr_radii.get(lig, 200.0)
        radius_groups.setdefault(r, []).append(lig)

    parts = []
    for radius, ligs in radius_groups.items():
        sub = lig_df[ligs]
        result = compute_radius_weights_fast(xy, sub, radius, 1.0)
        parts.append(result)

    if parts:
        return pd.concat(parts, axis=1).reindex(columns=unique_ligs, fill_value=0)
    return pd.DataFrame(0.0, index=obs_names, columns=unique_ligs)


def scatter_max_to_full(rw_lr, rw_tfl):
    """max(rw_lr, rw_tfl) scattered to (n_cells, n_genes) dense array."""
    result = np.zeros((n_cells, n_genes))
    for col_name in rw_lr.columns:
        if col_name in gene_to_idx:
            result[:, gene_to_idx[col_name]] = rw_lr[col_name].values
    for col_name in rw_tfl.columns:
        if col_name in gene_to_idx:
            idx = gene_to_idx[col_name]
            result[:, idx] = np.maximum(result[:, idx], rw_tfl[col_name].values)
    return result


def perturb_all_cells(delta_sim, splashed_dict):
    """Apply splash derivatives × delta to compute new simulated delta."""
    result = np.zeros((n_cells, n_genes))
    for gene_name in gene_names:
        beta_out = splashed_dict.get(gene_name)
        if beta_out is not None:
            bf = betaframes[gene_name]
            mod_idx = bf.modulator_gene_indices
            gene_idx = gene_to_idx[gene_name]
            result[:, gene_idx] = np.sum(beta_out.values * delta_sim[:, mod_idx], axis=1)
    return result


# ── Run perturb ──────────────────────────────────────────────────────────────

print(f"\nRunning Python perturb: {target} -> {gene_expr}, {n_propagation} iterations")

# Setup
delta_input = np.zeros((n_cells, n_genes))
target_idx = gene_to_idx[target]
delta_input[:, target_idx] = gene_expr - gene_mtx[:, target_idx]
delta_simulated = delta_input.copy()

# ligands_0
ligand_indices = [gene_to_idx[l] for l in all_ligands if l in gene_to_idx]
ligands_0 = np.zeros((n_cells, n_genes))
for idx in ligand_indices:
    ligands_0[:, idx] = gene_mtx[:, idx]

# Initial received ligands
rw_ligands_init = pd.DataFrame(rw_lig_df.values, index=obs_names, columns=rw_lig_df.columns)
rw_tfligands_init = pd.DataFrame(rw_tfl_df.values, index=obs_names, columns=rw_tfl_df.columns)

rw_max_0 = scatter_max_to_full(rw_ligands_init, rw_tfligands_init)

rw_lr_for_splash = rw_ligands_init.copy()
gene_mtx_1 = gene_mtx.copy()
max_per_gene = gene_mtx.max(axis=0)

lr_ligands = sorted(ligands_set)
tfl_ligs = sorted(tfl_ligands_set)

t0 = time.perf_counter()

for iteration in range(n_propagation):
    print(f"  perturb iteration {iteration+1}/{n_propagation}")

    # 1. Splash all
    gex_filtered = np.where(gene_mtx_1 > 1e-9, gene_mtx_1, 0.0)
    gex_df = pd.DataFrame(gex_filtered, index=obs_names, columns=gene_names)

    splashed = {}
    for gn, bf in betaframes.items():
        splashed[gn] = bf.splash(
            rw_lr_for_splash, rw_tfligands_init, gex_df,
            scale_factor=1.0, beta_cap=None
        )

    # 2. Update gene expression
    gene_mtx_1 = gene_mtx + delta_simulated

    # 3. Recompute weighted ligands
    w_lr_new = compute_weighted_ligands_np(gene_mtx_1, lr_ligands)
    w_tfl_new = compute_weighted_ligands_np(gene_mtx_1, tfl_ligs)

    # 4. Delta in received ligands
    rw_max_1 = scatter_max_to_full(w_lr_new, w_tfl_new)
    delta_rw = rw_max_1 - rw_max_0

    # rw_lr_for_splash becomes max-combined for next iteration
    rw_lr_for_splash = pd.DataFrame(rw_max_1, index=obs_names, columns=gene_names)

    # 5. Delta in ligand expression
    ligands_1 = np.zeros((n_cells, n_genes))
    for idx in ligand_indices:
        ligands_1[:, idx] = gene_mtx_1[:, idx]
    delta_ligands = ligands_1 - ligands_0

    # 6. Replace ligand deltas with received-ligand deltas
    delta_simulated = delta_simulated + delta_rw - delta_ligands

    # 7. Perturb all cells
    delta_simulated = perturb_all_cells(delta_simulated, splashed)

    # 8. Pin target genes
    delta_simulated = np.where(delta_input != 0, delta_input, delta_simulated)

    # 9. Clip
    gem_tmp = np.clip(gene_mtx + delta_simulated, a_min=0.0, a_max=max_per_gene)
    delta_simulated = gem_tmp - gene_mtx

elapsed_py = time.perf_counter() - t0

# Final
simulated_py = gene_mtx + delta_simulated
simulated_py[:, target_idx] = gene_expr

delta_py = simulated_py - gene_mtx
nonzero_py = np.sum(np.abs(delta_py) > 1e-15)
print(f"\nPython result:")
print(f"  nonzero delta: {nonzero_py} / {delta_py.size}")
print(f"  max |delta|: {np.max(np.abs(delta_py)):.6e}")
print(f"  mean |delta|: {np.mean(np.abs(delta_py)):.6e}")
print(f"  elapsed: {elapsed_py*1000:.1f} ms")

# ── Save Python result ──────────────────────────────────────────────────────

np.savetxt(
    f"{INPUTS_DIR}/result_python.csv",
    simulated_py, delimiter=",",
    header=",".join(gene_names), comments=""
)

# ── Compare with Rust ────────────────────────────────────────────────────────

rust_result = pd.read_csv(f"{INPUTS_DIR}/result_rust.csv")
rust_vals = rust_result.values

diff = np.abs(simulated_py - rust_vals)
max_diff = np.max(diff)
mean_diff = np.mean(diff)
n_mismatch = np.sum(diff > 1e-6)

print(f"\nComparison with Rust:")
print(f"  max |diff|: {max_diff:.6e}")
print(f"  mean |diff|: {mean_diff:.6e}")
print(f"  elements with |diff| > 1e-6: {n_mismatch} / {diff.size}")

if max_diff < 1e-4:
    print("  ✓ MATCH (within 1e-4)")
elif max_diff < 1e-2:
    print(f"  ~ CLOSE (max diff {max_diff:.6e})")
else:
    print(f"  ✗ MISMATCH")

    # Find worst mismatches
    flat_idx = np.argsort(diff.ravel())[-10:][::-1]
    for fi in flat_idx:
        r, c = divmod(fi, n_genes)
        print(f"    [{r},{c}] gene={gene_names[c]}: py={simulated_py[r,c]:.10f} rs={rust_vals[r,c]:.10f} diff={diff[r,c]:.6e}")


# ── Benchmark ────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("Python perturb benchmark")
print(f"{'='*60}")

for n_cells_bench in [200, 500, 1000, 2000, 5000, 10000]:
    obs = [f"cell_{i}" for i in range(n_cells_bench)]
    clust = [i % n_clusters for i in range(n_cells_bench)]

    # Rebuild betaframes for this size
    bfs = {}
    for path in beta_paths:
        gn = os.path.basename(path).replace("_betadata.feather", "")
        df_raw = load_betadata_table(path)
        new_cols = {c: c if c == "beta0" else f"beta_{c}" for c in df_raw.columns}
        df_raw = df_raw.rename(columns=new_cols)
        expanded = pd.DataFrame(index=obs, columns=df_raw.columns, dtype=float)
        for i in range(n_cells_bench):
            expanded.iloc[i] = df_raw.iloc[clust[i] % len(df_raw)].values
        expanded = expanded.fillna(0.0).astype(float)
        expanded.index.name = gn
        bf = BetaFrame(expanded)
        bf.modulator_gene_indices = [
            gene_names.index(g.replace("beta_", ""))
            for g in bf.modulators_genes
            if g.replace("beta_", "") in gene_names
        ]
        bfs[gn] = bf

    # Mock data matching Rust
    gmtx = np.array([[1.0 + 0.01*g + 0.001*(c%7) for g in range(n_genes)] for c in range(n_cells_bench)])
    grid_w = int(np.ceil(np.sqrt(n_cells_bench)))
    xy_b = np.array([[(i % grid_w) * 10.0, (i // grid_w) * 10.0] for i in range(n_cells_bench)])

    # Compute initial wl
    lr_l = sorted(ligands_set)
    tfl_l = sorted(tfl_ligands_set)

    def compute_wl_bench(gmtx_b, lig_list):
        unique_l = sorted(set(l for l in lig_list if l in gene_to_idx))
        if not unique_l:
            return pd.DataFrame(index=obs)
        lig_d = np.zeros((n_cells_bench, len(unique_l)))
        for j, l in enumerate(unique_l):
            col = gmtx_b[:, gene_to_idx[l]]
            lig_d[:, j] = np.where(col > 1e-9, col, 0.0)
        lig_df = pd.DataFrame(lig_d, index=obs, columns=unique_l)
        r_groups = {}
        for l in unique_l:
            r = lr_radii.get(l, 200.0)
            r_groups.setdefault(r, []).append(l)
        parts = []
        for rad, ls in r_groups.items():
            parts.append(compute_radius_weights_fast(xy_b, lig_df[ls], rad, 1.0))
        if parts:
            return pd.concat(parts, axis=1).reindex(columns=unique_l, fill_value=0)
        return pd.DataFrame(0.0, index=obs, columns=unique_l)

    rw_lr_b = compute_wl_bench(gmtx, lr_l)
    rw_tfl_b = compute_wl_bench(gmtx, tfl_l)

    def scatter_max_b(rw_lr, rw_tfl):
        result = np.zeros((n_cells_bench, n_genes))
        for cn in rw_lr.columns:
            if cn in gene_to_idx:
                result[:, gene_to_idx[cn]] = rw_lr[cn].values
        for cn in rw_tfl.columns:
            if cn in gene_to_idx:
                idx = gene_to_idx[cn]
                result[:, idx] = np.maximum(result[:, idx], rw_tfl[cn].values)
        return result

    def perturb_cells_b(dsim, sp):
        result = np.zeros((n_cells_bench, n_genes))
        for gn2 in gene_names:
            bo = sp.get(gn2)
            if bo is not None:
                bf2 = bfs[gn2]
                mi = bf2.modulator_gene_indices
                gi = gene_to_idx[gn2]
                result[:, gi] = np.sum(bo.values * dsim[:, mi], axis=1)
        return result

    # Run perturb
    t0 = time.perf_counter()

    di = np.zeros((n_cells_bench, n_genes))
    di[:, gene_to_idx[target]] = gene_expr - gmtx[:, gene_to_idx[target]]
    ds = di.copy()

    lig_0 = np.zeros((n_cells_bench, n_genes))
    for idx in ligand_indices:
        lig_0[:, idx] = gmtx[:, idx]

    rm0 = scatter_max_b(rw_lr_b, rw_tfl_b)
    rw_sp = rw_lr_b.copy()
    gm1 = gmtx.copy()
    mpg = gmtx.max(axis=0)

    for it in range(n_propagation):
        gf = np.where(gm1 > 1e-9, gm1, 0.0)
        gdf = pd.DataFrame(gf, index=obs, columns=gene_names)
        sp = {}
        for gn2, bf2 in bfs.items():
            sp[gn2] = bf2.splash(rw_sp, rw_tfl_b, gdf, scale_factor=1.0, beta_cap=None)
        gm1 = gmtx + ds
        wl = compute_wl_bench(gm1, lr_l)
        wtfl = compute_wl_bench(gm1, tfl_l)
        rm1 = scatter_max_b(wl, wtfl)
        drw = rm1 - rm0
        rw_sp = pd.DataFrame(rm1, index=obs, columns=gene_names)
        lig_1 = np.zeros((n_cells_bench, n_genes))
        for idx in ligand_indices:
            lig_1[:, idx] = gm1[:, idx]
        dl = lig_1 - lig_0
        ds = ds + drw - dl
        ds = perturb_cells_b(ds, sp)
        ds = np.where(di != 0, di, ds)
        gt = np.clip(gmtx + ds, a_min=0.0, a_max=mpg)
        ds = gt - gmtx

    elapsed_bench = (time.perf_counter() - t0) * 1000
    print(f"  {n_cells_bench:>5} cells x {n_genes} genes x {n_propagation} iters: {elapsed_bench:>8.1f} ms")

print("\nDone.")
