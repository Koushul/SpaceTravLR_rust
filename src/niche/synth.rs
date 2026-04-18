//! Synthetic spatial dataset with **known functional microniches**, fully
//! constructible in pure Rust (no h5ad / Python). Used by tests and the
//! `spacetravlr-niche synthetic` subcommand.
//!
//! Each microniche owns a small program of `(TF, ligand, receptor)` triples
//! that act inside it. Splash on the resulting Betabase therefore *encodes*
//! niche identity: only cells in niche `n` have non-zero β × wL[L] × gex[R]
//! contributions for niche `n`'s LR pairs. The CNN's job is to recover this
//! niche structure from the per-cell Jacobian.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::distributions::{Distribution, WeightedIndex};
use rand::{Rng, rngs::StdRng};

use crate::betadata::{BetaFrame, BetaFrameFromParts, Betabase, GeneMatrix};
use crate::ligand::calculate_weighted_ligands;
use crate::perturb::compute_splash_all;

/// Output bundle from [`make_synthetic_run`].
pub struct SyntheticNicheRun {
    pub n_cells: usize,
    pub n_niches: usize,
    pub xy: Array2<f64>,
    /// Ground-truth integer niche label per cell (0..n_niches).
    pub niche_gt: Vec<usize>,
    /// Ground-truth cell-type label per cell (independent of niche).
    pub cell_type: Vec<usize>,
    pub gene_names: Vec<String>,
    pub gene_matrix: Array2<f64>,
    pub bb: Betabase,
    pub rw_ligands: GeneMatrix,
    pub rw_tfligands: GeneMatrix,
    pub gex_gm: GeneMatrix,
    /// Mapping of `target gene → (n_cells, n_modulators)` splash matrix.
    pub splash: HashMap<String, GeneMatrix>,
    pub lr_pairs: Vec<(String, String, usize)>, // (ligand, receptor, owning niche)
    pub niche_tfs: HashMap<usize, Vec<String>>,
}

/// Build a synthetic spatial run with `n_niches` circular microniches, each
/// containing approximately `cells_per_niche` cells.
///
/// The dataset is intentionally **adversarial for non-spatial cell-typing
/// methods**:
///
/// * Every cell, regardless of niche, is one of three "cell types"
///   (`A`, `B`, `C`). Cell type drives which TFs and which receptors a cell
///   *expresses* — but **not** which ligand neighbourhood it sits in.
/// * Niche identity is encoded purely in *which ligands are firing in the
///   spatial neighbourhood* (each niche owns a ligand set; cells of all types
///   inside a niche express that niche's ligands). A cell's response to those
///   ligands depends on its receptor expression (i.e. on cell type).
/// * The CNN sees `splash = β × wL × gex_R`, which combines ligand
///   neighbourhood (niche-driven) and receptor expression (cell-type driven)
///   — so the per-cell Jacobian *carries niche identity* even when raw
///   expression is dominated by cell type.
pub fn make_synthetic_run(cells_per_niche: usize, n_niches: usize, seed: u64) -> SyntheticNicheRun {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_cells = cells_per_niche * n_niches;
    let n_cell_types = 3usize;
    let n_tf_per_type = 4usize;
    let n_tf_shared = 4usize;
    let n_lr_per_niche = 4usize;
    let n_targets_per_niche = 4usize;

    // Spatial layout: arrange niche centres on a grid, each niche is a disk.
    let cols = (n_niches as f64).sqrt().ceil() as usize;
    let rows = (n_niches as f64 / cols as f64).ceil() as usize;
    let field = 1000.0f64;
    let centres: Vec<[f64; 2]> = (0..n_niches)
        .map(|n| {
            let r = n / cols;
            let c = n % cols;
            [
                (c as f64 + 0.5) * field / cols as f64,
                (r as f64 + 0.5) * field / rows as f64,
            ]
        })
        .collect();
    let radius = 0.18 * field;

    let mut xy = Array2::<f64>::zeros((n_cells, 2));
    let mut niche_gt = vec![0usize; n_cells];
    for n in 0..n_niches {
        for k in 0..cells_per_niche {
            let i = n * cells_per_niche + k;
            let ang: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
            let r: f64 = radius * rng.r#gen::<f64>().sqrt();
            xy[[i, 0]] = centres[n][0] + r * ang.cos();
            xy[[i, 1]] = centres[n][1] + r * ang.sin();
            niche_gt[i] = n;
        }
    }
    // Shuffle so the dataloader doesn't see niche-ordered batches.
    let mut perm: Vec<usize> = (0..n_cells).collect();
    use rand::seq::SliceRandom;
    perm.shuffle(&mut rng);
    let xy = {
        let mut new_xy = Array2::<f64>::zeros((n_cells, 2));
        for (new_i, &old_i) in perm.iter().enumerate() {
            new_xy[[new_i, 0]] = xy[[old_i, 0]];
            new_xy[[new_i, 1]] = xy[[old_i, 1]];
        }
        new_xy
    };
    let niche_gt: Vec<usize> = perm.iter().map(|&i| niche_gt[i]).collect();

    // Cell-type label per cell (TFs and receptors are driven by cell type, not
    // niche).
    let cell_type: Vec<usize> = (0..n_cells).map(|_| rng.gen_range(0..n_cell_types)).collect();

    // Gene panel: shared TFs (everyone), per-cell-type TFs (cell-type marker),
    // and a "niche_tfs" map that's actually empty (niche identity comes from
    // the ligand neighbourhood, not from cell-internal TFs).
    let shared_tfs: Vec<String> = (0..n_tf_shared).map(|i| format!("TFs{:02}", i)).collect();
    let mut type_tfs: HashMap<usize, Vec<String>> = HashMap::new();
    let mut all_tfs: Vec<String> = shared_tfs.clone();
    for t in 0..n_cell_types {
        let v: Vec<String> = (0..n_tf_per_type).map(|i| format!("TFct{}_{}", t, i)).collect();
        all_tfs.extend(v.iter().cloned());
        type_tfs.insert(t, v);
    }
    // Keep a niche_tfs map populated with one phantom TF per niche so the
    // public field reflects the program; their β contributions are tiny.
    let mut niche_tfs: HashMap<usize, Vec<String>> = HashMap::new();
    for n in 0..n_niches {
        let v: Vec<String> = vec![format!("TFn{}_marker", n)];
        all_tfs.extend(v.iter().cloned());
        niche_tfs.insert(n, v);
    }
    let mut lr_pairs: Vec<(String, String, usize)> = Vec::new();
    let mut all_ligands: Vec<String> = Vec::new();
    let mut all_receptors: Vec<String> = Vec::new();
    for n in 0..n_niches {
        for k in 0..n_lr_per_niche {
            let l = format!("L{}_{}", n, k);
            let r = format!("R{}_{}", n, k);
            all_ligands.push(l.clone());
            all_receptors.push(r.clone());
            lr_pairs.push((l, r, n));
        }
    }
    let mut target_names: Vec<(String, usize)> = Vec::new();
    for n in 0..n_niches {
        for k in 0..n_targets_per_niche {
            target_names.push((format!("T{}_{}", n, k), n));
        }
    }

    let mut gene_set: HashSet<String> = HashSet::new();
    let mut gene_names: Vec<String> = Vec::new();
    let push = |g: &str, set: &mut HashSet<String>, names: &mut Vec<String>| {
        if set.insert(g.to_string()) {
            names.push(g.to_string());
        }
    };
    for g in &all_tfs {
        push(g, &mut gene_set, &mut gene_names);
    }
    for g in &all_ligands {
        push(g, &mut gene_set, &mut gene_names);
    }
    for g in &all_receptors {
        push(g, &mut gene_set, &mut gene_names);
    }
    for (g, _) in &target_names {
        push(g, &mut gene_set, &mut gene_names);
    }
    // Some "extra" genes for noise / realism.
    for k in 0..40 {
        push(&format!("X{:03}", k), &mut gene_set, &mut gene_names);
    }
    let n_genes = gene_names.len();
    let gene2idx: HashMap<String, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.clone(), i))
        .collect();

    // Expression matrix.
    //
    // - all genes: small noise floor (dominates raw-expression k-means)
    // - shared TFs: high in every cell
    // - cell-type TFs: high only in cells of that type (drives raw clustering)
    // - niche-marker TF: faint signal so niches are not literally readable
    //   from one gene
    // - ligands of niche n: expressed by cells **inside** niche n only
    //   (so the spatial neighbourhood of every cell in niche n receives them)
    // - receptors: expression depends on cell type, not niche
    let mut gene_matrix = Array2::<f64>::zeros((n_cells, n_genes));
    let noise = 0.05f64;
    for j in 0..n_genes {
        for i in 0..n_cells {
            gene_matrix[[i, j]] = noise * rng.r#gen::<f64>();
        }
    }
    for tf in &shared_tfs {
        let j = gene2idx[tf];
        for i in 0..n_cells {
            gene_matrix[[i, j]] = 1.0 + 0.5 * rng.r#gen::<f64>();
        }
    }
    for ct in 0..n_cell_types {
        for tf in &type_tfs[&ct] {
            let j = gene2idx[tf];
            for i in 0..n_cells {
                if cell_type[i] == ct {
                    // Strong cell-type marker — drives raw expression k-means.
                    gene_matrix[[i, j]] = 3.0 + 1.5 * rng.r#gen::<f64>();
                } else {
                    gene_matrix[[i, j]] = noise * rng.r#gen::<f64>();
                }
            }
        }
    }
    // Faint niche-marker (so a perfect cell-typing baseline cannot get >50% ARI vs niche)
    for n in 0..n_niches {
        for tf in &niche_tfs[&n] {
            let j = gene2idx[tf];
            for i in 0..n_cells {
                if niche_gt[i] == n {
                    gene_matrix[[i, j]] = 0.6 + 0.4 * rng.r#gen::<f64>();
                } else {
                    gene_matrix[[i, j]] = noise * rng.r#gen::<f64>();
                }
            }
        }
    }
    // Ligands fire inside their niche
    for (l, _r, n_owner) in lr_pairs.iter() {
        let lj = gene2idx[l];
        for i in 0..n_cells {
            if niche_gt[i] == *n_owner {
                gene_matrix[[i, lj]] = 1.5 + 0.8 * rng.r#gen::<f64>();
            } else {
                gene_matrix[[i, lj]] = noise * rng.r#gen::<f64>();
            }
        }
    }
    // Receptors expressed by cells of certain types (random per receptor).
    let receptor_to_type: HashMap<String, usize> = lr_pairs
        .iter()
        .map(|(_, r, _)| (r.clone(), rng.gen_range(0..n_cell_types)))
        .collect();
    for (_, r, _) in &lr_pairs {
        let rj = gene2idx[r];
        let owner_type = receptor_to_type[r];
        for i in 0..n_cells {
            if cell_type[i] == owner_type {
                gene_matrix[[i, rj]] = 1.0 + 0.5 * rng.r#gen::<f64>();
            } else {
                gene_matrix[[i, rj]] = noise * rng.r#gen::<f64>();
            }
        }
    }

    // Build a Betabase. Targets owned by niche `n` are driven mostly by
    // niche `n`'s LR pairs (the spatially-encoded signal) and weakly by the
    // shared TF set. The cell-type TF effects on each target are intentionally
    // small so that splash carries niche identity, not cell-type identity.
    let mut bb_data: HashMap<String, BetaFrame> = HashMap::new();
    let mut bb_lig: HashSet<String> = HashSet::new();
    let mut bb_rec: HashSet<String> = HashSet::new();
    let mut bb_tfs: HashSet<String> = HashSet::new();

    let row_labels = vec!["0".to_string()]; // single cluster for splash math
    for (tg, owner) in &target_names {
        let mut tfs: Vec<String> = shared_tfs.clone();
        tfs.extend(niche_tfs[owner].iter().cloned());
        let mut tf_betas = Array2::<f32>::zeros((1, tfs.len()));
        for (i, t) in tfs.iter().enumerate() {
            let b = if niche_tfs[owner].contains(t) {
                rng.gen_range(0.4f32..0.8)
            } else {
                rng.gen_range(0.05f32..0.20)
            };
            tf_betas[[0, i]] = b;
            bb_tfs.insert(t.clone());
        }
        let mut lig: Vec<String> = Vec::new();
        let mut rec: Vec<String> = Vec::new();
        for (l, r, n) in &lr_pairs {
            if n == owner {
                lig.push(l.clone());
                rec.push(r.clone());
                bb_lig.insert(l.clone());
                bb_rec.insert(r.clone());
            }
        }
        let mut lr_betas = Array2::<f32>::zeros((1, lig.len()));
        for i in 0..lig.len() {
            lr_betas[[0, i]] = rng.gen_range(0.8f32..1.5);
        }

        let parts = BetaFrameFromParts {
            gene_name: tg.clone(),
            row_labels: row_labels.clone(),
            intercepts: Array1::from_vec(vec![0.1f32]),
            tf_betas,
            tfs: tfs.clone(),
            lr_betas,
            ligands: lig.clone(),
            receptors: rec.clone(),
            tfl_betas: Array2::<f32>::zeros((1, 0)),
            tfl_ligands: Vec::new(),
            tfl_regulators: Vec::new(),
        };
        let mut bf: BetaFrame = parts.into();

        // Expand to all cells (single cluster → all map to row 0)
        let obs: Vec<String> = (0..n_cells).map(|i| format!("cell_{:05}", i)).collect();
        let cluster_keys: Vec<String> = vec!["0".to_string(); n_cells];
        let mapping = Arc::new(BetaFrame::compute_cell_mapping(
            &row_labels,
            &obs,
            &cluster_keys,
        ));
        bf.expand_to_cells(Arc::new(obs), mapping);
        bf.modulator_gene_indices = Some(
            bf.modulator_genes
                .iter()
                .map(|g| {
                    let plain = g.strip_prefix("beta_").unwrap_or(g);
                    *gene2idx.get(plain).expect("modulator in panel")
                })
                .collect(),
        );
        bb_data.insert(tg.clone(), bf);
    }

    let bb = Betabase {
        data: bb_data,
        ligands_set: bb_lig.clone(),
        receptors_set: bb_rec,
        tfl_ligands_set: HashSet::new(),
        tfs_set: bb_tfs,
    };

    // Initial received-ligand matrices (one column per ligand in our graph)
    let lig_names: Vec<String> = bb_lig.iter().cloned().collect();
    let mut lig_data = Array2::<f64>::zeros((n_cells, lig_names.len()));
    for (j, lname) in lig_names.iter().enumerate() {
        let gi = gene2idx[lname];
        for i in 0..n_cells {
            lig_data[[i, j]] = gene_matrix[[i, gi]];
        }
    }
    let rw_data = calculate_weighted_ligands(&xy, &lig_data, 60.0, 1.0);
    let rw_ligands = GeneMatrix::new(rw_data.mapv(|v| v as f32), lig_names);
    let rw_tfligands = GeneMatrix::new(Array2::<f32>::zeros((n_cells, 0)), Vec::new());

    let gex_gm = GeneMatrix::new(gene_matrix.mapv(|v| v as f32), gene_names.clone());

    // Splash everything (so callers don't repeat themselves).
    let splash = compute_splash_all(&bb, &rw_ligands, &rw_tfligands, &gex_gm, 1.0, None);

    SyntheticNicheRun {
        n_cells,
        n_niches,
        xy,
        niche_gt,
        cell_type,
        gene_names,
        gene_matrix,
        bb,
        rw_ligands,
        rw_tfligands,
        gex_gm,
        splash,
        lr_pairs,
        niche_tfs,
    }
}

/// Pick a random sample of `m` cells weighted by per-niche counts. Useful for
/// quick smoke tests.
pub fn random_subsample(run: &SyntheticNicheRun, m: usize, seed: u64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = run.n_cells;
    if m >= n {
        return (0..n).collect();
    }
    let weights: Vec<u64> = vec![1; n];
    let dist = WeightedIndex::new(&weights).unwrap();
    let mut chosen = HashSet::new();
    while chosen.len() < m {
        chosen.insert(dist.sample(&mut rng));
    }
    chosen.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synth_run_has_expected_shapes() {
        let run = make_synthetic_run(40, 3, 0);
        assert_eq!(run.n_cells, 120);
        assert_eq!(run.n_niches, 3);
        assert_eq!(run.xy.shape(), &[120, 2]);
        assert_eq!(run.gene_matrix.shape(), &[120, run.gene_names.len()]);
        assert!(run.splash.len() >= 9);
        for (_, gm) in &run.splash {
            assert_eq!(gm.n_rows(), 120);
        }
    }
}
