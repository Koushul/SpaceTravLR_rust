"""Benchmark Python splash using /tmp/betas."""
import sys, os, glob, time
sys.path.insert(0, os.path.expanduser("~/Projects/jishnulab/SpaceTravLR/src"))

import numpy as np
import pandas as pd
from SpaceTravLR.beta import BetaFrame

BETAS_DIR = "/tmp/betas"

N_CELLS_LIST = [100, 1_000, 10_000, 50_000, 100_000]
N_CLUSTERS = 13
N_WARMUP = 1
N_ITERS = 5

def load_betadata_table(path):
    df = pd.read_feather(path)
    return df.set_index(df.columns[0])


beta_paths = sorted(glob.glob(f"{BETAS_DIR}/*_betadata.feather"))
print(f"Found {len(beta_paths)} betadata feather files\n")

for n_cells in N_CELLS_LIST:
    obs_names = [f"cell_{i}" for i in range(n_cells)]
    clusters = [i % N_CLUSTERS for i in range(n_cells)]

    betaframes = {}
    all_genes_set = set()
    lig_genes_set = set()

    for path in beta_paths:
        gene_name = os.path.basename(path).replace("_betadata.feather", "")
        df = load_betadata_table(path)
        new_cols = {c: c if c == "beta0" else f"beta_{c}" for c in df.columns}
        df = df.rename(columns=new_cols)

        expanded = pd.DataFrame(index=obs_names, columns=df.columns, dtype=float)
        for i, cell in enumerate(obs_names):
            cluster_id = clusters[i]
            if cluster_id < len(df):
                expanded.iloc[i] = df.iloc[cluster_id % len(df)].values
        expanded = expanded.fillna(0.0).astype(float)
        expanded.index.name = gene_name

        bf = BetaFrame(expanded)
        betaframes[gene_name] = bf

        for g in bf.tfs: all_genes_set.add(g)
        for g in bf.ligands: all_genes_set.add(g)
        for g in bf.receptors: all_genes_set.add(g)
        for g in bf.tfl_ligands: all_genes_set.add(g)
        for g in bf.tfl_regulators: all_genes_set.add(g)
        lig_genes_set.update(bf.ligands)
        lig_genes_set.update(bf.tfl_ligands)

    all_genes = sorted(all_genes_set)
    lig_genes = sorted(lig_genes_set)

    gex_df = pd.DataFrame(1.0, index=obs_names, columns=all_genes)
    rw_ligands = pd.DataFrame(1.0, index=obs_names, columns=lig_genes)
    rw_ligands_tfl = pd.DataFrame(1.0, index=obs_names, columns=lig_genes)

    gene_names = sorted(betaframes.keys())

    # Warmup
    for _ in range(N_WARMUP):
        for gn in gene_names:
            betaframes[gn].splash(rw_ligands, rw_ligands_tfl, gex_df, scale_factor=1.0, beta_cap=None)

    # Timed runs
    times = []
    for it in range(N_ITERS):
        t0 = time.perf_counter()
        for gn in gene_names:
            betaframes[gn].splash(rw_ligands, rw_ligands_tfl, gex_df, scale_factor=1.0, beta_cap=None)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"  {n_cells:>6} cells x {len(gene_names)} genes: {mean_ms:>8.1f} ms  (std {std_ms:.1f} ms)")

print("\nDone.")
