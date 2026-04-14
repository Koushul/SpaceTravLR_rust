use std::collections::{HashMap, HashSet};

use ndarray::{Array2, Axis};
use rayon::prelude::*;

use crate::betadata::BetaFrame;
use crate::ligand_jacobian::{
    received_ligand_jacobian_one_ligand, received_ligand_jacobian_row_for_cell,
};
use crate::perturb_mode::PerturbRuntime;

#[derive(Clone, Debug)]
pub struct CellCommNetworkParams {
    pub beta_scale_factor: f32,
    pub min_expression: f64,
    pub edge_threshold_abs: f64,
    pub include_self_loops: bool,
    /// When false, use `[spatial].contact_distance` from the run when finite and positive.
    pub ignore_contact_distance: bool,
}

impl Default for CellCommNetworkParams {
    fn default() -> Self {
        Self {
            beta_scale_factor: 1.0,
            min_expression: 1e-9,
            edge_threshold_abs: 0.0,
            include_self_loops: false,
            ignore_contact_distance: false,
        }
    }
}

/// Directed edges sender `j` → receiver `i`: Σ_targets Σ_LR,TFL |β · R · ∂WL/∂L_j| at baseline expression,
/// with R = receptor (LR) or TF (TFL) expression at the receiver cell; same linearization as
/// [`BetaFrame::splash`] for those ligand channels (chain rule on received ligand).
pub fn aggregate_lr_tfl_communication_edges(
    rt: &PerturbRuntime,
    params: &CellCommNetworkParams,
) -> (Array2<f64>, HashMap<String, usize>) {
    let n = rt.gene_mtx.nrows();
    let gene_to_idx: HashMap<&str, usize> = rt
        .gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();

    let mut lig_to_radius_bits: HashMap<String, u64> = HashMap::new();
    for lig in rt.bb.ligands_set.iter().chain(rt.bb.tfl_ligands_set.iter()) {
        let r = rt.lr_radii.get(lig).copied().unwrap_or(rt.cfg.spatial.radius);
        lig_to_radius_bits.insert(lig.clone(), r.to_bits());
    }

    let mut radius_groups: HashMap<u64, Vec<String>> = HashMap::new();
    for lig in lig_to_radius_bits.keys() {
        let bits = lig_to_radius_bits[lig];
        radius_groups.entry(bits).or_default().push(lig.clone());
    }

    let wl_scale = rt.cfg.spatial.weighted_ligand_scale_factor;
    let contact = if params.ignore_contact_distance {
        None
    } else {
        let c = rt.cfg.spatial.contact_distance;
        (c.is_finite() && c > 0.0).then_some(c)
    };

    let mut pair_contrib = Array2::<f64>::zeros((n, n));
    let mut seen_lig: HashSet<String> = HashSet::new();

    for (rbits, ligs) in &radius_groups {
        let radius = f64::from_bits(*rbits);
        let jac = received_ligand_jacobian_one_ligand(&rt.xy, radius, wl_scale, contact);
        for lig in ligs {
            if !seen_lig.insert(lig.clone()) {
                continue;
            }
            for bf in rt.bb.data.values() {
                accumulate_one_frame(
                    bf,
                    &rt.gene_mtx,
                    &jac,
                    lig.as_str(),
                    &gene_to_idx,
                    params,
                    &mut pair_contrib,
                );
            }
        }
    }

    let mut w = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            if !params.include_self_loops && i == j {
                continue;
            }
            let c = pair_contrib[[i, j]];
            if c > params.edge_threshold_abs {
                w[[i, j]] = c;
            }
        }
    }

    let mut obs_to_idx: HashMap<String, usize> = HashMap::with_capacity(n);
    for (i, name) in rt.obs_names.iter().enumerate() {
        obs_to_idx.insert(name.clone(), i);
    }
    (w, obs_to_idx)
}

/// Top `top_k` senders `j` by weight for fixed receiver `receiver_idx` (directed edges j → receiver).
pub fn communication_edges_from_receiver(
    rt: &PerturbRuntime,
    params: &CellCommNetworkParams,
    receiver_idx: usize,
    top_k: usize,
) -> Vec<(usize, f64)> {
    let n = rt.gene_mtx.nrows();
    if receiver_idx >= n || top_k == 0 {
        return Vec::new();
    }
    let mut contrib = vec![0.0_f64; n];
    accumulate_receiver_contrib_all(rt, params, receiver_idx, &mut contrib);
    top_k_pairs(
        &contrib,
        top_k,
        params.edge_threshold_abs,
        receiver_idx,
        params.include_self_loops,
    )
}

/// Top `top_k` receivers `i` for fixed sender `sender_idx` (directed edges sender → i).
pub fn communication_edges_from_sender(
    rt: &PerturbRuntime,
    params: &CellCommNetworkParams,
    sender_idx: usize,
    top_k: usize,
) -> Vec<(usize, f64)> {
    let n = rt.gene_mtx.nrows();
    if sender_idx >= n || top_k == 0 {
        return Vec::new();
    }
    let mut contrib = vec![0.0_f64; n];
    accumulate_sender_contrib_all(rt, params, sender_idx, &mut contrib);
    top_k_pairs(&contrib, top_k, params.edge_threshold_abs, sender_idx, params.include_self_loops)
}

fn top_k_pairs(
    contrib: &[f64],
    top_k: usize,
    threshold: f64,
    skip_idx: usize,
    include_self: bool,
) -> Vec<(usize, f64)> {
    let mut pairs: Vec<(usize, f64)> = contrib
        .iter()
        .enumerate()
        .filter(|(i, v)| {
            **v > threshold && (include_self || *i != skip_idx)
        })
        .map(|(i, v)| (i, *v))
        .collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(top_k);
    pairs
}

fn wl_contact(rt: &PerturbRuntime, params: &CellCommNetworkParams) -> Option<f64> {
    if params.ignore_contact_distance {
        None
    } else {
        let c = rt.cfg.spatial.contact_distance;
        (c.is_finite() && c > 0.0).then_some(c)
    }
}

fn accumulate_receiver_contrib_all(
    rt: &PerturbRuntime,
    params: &CellCommNetworkParams,
    receiver_idx: usize,
    contrib: &mut [f64],
) {
    let gene_to_idx: HashMap<&str, usize> = rt
        .gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();

    let mut lig_to_radius_bits: HashMap<String, u64> = HashMap::new();
    for lig in rt.bb.ligands_set.iter().chain(rt.bb.tfl_ligands_set.iter()) {
        let r = rt.lr_radii.get(lig).copied().unwrap_or(rt.cfg.spatial.radius);
        lig_to_radius_bits.insert(lig.clone(), r.to_bits());
    }

    let mut radius_groups: HashMap<u64, Vec<String>> = HashMap::new();
    for lig in lig_to_radius_bits.keys() {
        let bits = lig_to_radius_bits[lig];
        radius_groups.entry(bits).or_default().push(lig.clone());
    }

    let wl_scale = rt.cfg.spatial.weighted_ligand_scale_factor;
    let contact = wl_contact(rt, params);
    let mut seen_lig: HashSet<String> = HashSet::new();

    for (rbits, ligs) in &radius_groups {
        let radius = f64::from_bits(*rbits);
        let jac_row =
            received_ligand_jacobian_row_for_cell(&rt.xy, receiver_idx, radius, wl_scale, contact);
        for lig in ligs {
            if !seen_lig.insert(lig.clone()) {
                continue;
            }
            for bf in rt.bb.data.values() {
                accumulate_one_frame_receiver_row(
                    bf,
                    &rt.gene_mtx,
                    jac_row.as_slice(),
                    receiver_idx,
                    lig.as_str(),
                    &gene_to_idx,
                    params,
                    contrib,
                );
            }
        }
    }
}

fn accumulate_sender_contrib_all(
    rt: &PerturbRuntime,
    params: &CellCommNetworkParams,
    sender_idx: usize,
    contrib: &mut [f64],
) {
    let gene_to_idx: HashMap<&str, usize> = rt
        .gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();

    let mut lig_to_radius_bits: HashMap<String, u64> = HashMap::new();
    for lig in rt.bb.ligands_set.iter().chain(rt.bb.tfl_ligands_set.iter()) {
        let r = rt.lr_radii.get(lig).copied().unwrap_or(rt.cfg.spatial.radius);
        lig_to_radius_bits.insert(lig.clone(), r.to_bits());
    }

    let mut radius_groups: HashMap<u64, Vec<String>> = HashMap::new();
    for lig in lig_to_radius_bits.keys() {
        let bits = lig_to_radius_bits[lig];
        radius_groups.entry(bits).or_default().push(lig.clone());
    }

    let wl_scale = rt.cfg.spatial.weighted_ligand_scale_factor;
    let contact = wl_contact(rt, params);
    let mut seen_lig: HashSet<String> = HashSet::new();

    for (rbits, ligs) in &radius_groups {
        let radius = f64::from_bits(*rbits);
        let jac_col =
            received_ligand_jacobian_row_for_cell(&rt.xy, sender_idx, radius, wl_scale, contact);
        for lig in ligs {
            if !seen_lig.insert(lig.clone()) {
                continue;
            }
            for bf in rt.bb.data.values() {
                accumulate_one_frame_sender_col(
                    bf,
                    &rt.gene_mtx,
                    jac_col.as_slice(),
                    lig.as_str(),
                    &gene_to_idx,
                    params,
                    contrib,
                );
            }
        }
    }
}

fn accumulate_one_frame_receiver_row(
    bf: &BetaFrame,
    gene_mtx: &Array2<f64>,
    jac_row: &[f64],
    receiver_idx: usize,
    lig_name: &str,
    gene_to_idx: &HashMap<&str, usize>,
    params: &CellCommNetworkParams,
    contrib: &mut [f64],
) {
    let n = contrib.len();
    debug_assert_eq!(jac_row.len(), n);
    let map = bf.cell_to_beta_row.as_slice();
    let scale = params.beta_scale_factor as f64;
    let min_e = params.min_expression;
    let br = map[receiver_idx];

    for k in 0..bf.ligands.len() {
        if bf.ligands[k] != lig_name {
            continue;
        }
        let Some(&rec_gi) = gene_to_idx.get(bf.receptors[k].as_str()) else {
            continue;
        };
        let beta = f64::from(bf.lr_betas[[br, k]]);
        if beta == 0.0 {
            continue;
        }
        let rec = gene_mtx[[receiver_idx, rec_gi]];
        if rec <= min_e {
            continue;
        }
        let s = beta * rec * scale;
        for j in 0..n {
            contrib[j] += (s * jac_row[j]).abs();
        }
    }

    for k in 0..bf.tfl_ligands.len() {
        if bf.tfl_ligands[k] != lig_name {
            continue;
        }
        let Some(&reg_gi) = gene_to_idx.get(bf.tfl_regulators[k].as_str()) else {
            continue;
        };
        let beta = f64::from(bf.tfl_betas[[br, k]]);
        if beta == 0.0 {
            continue;
        }
        let reg = gene_mtx[[receiver_idx, reg_gi]];
        if reg <= min_e {
            continue;
        }
        let s = beta * reg * scale;
        for j in 0..n {
            contrib[j] += (s * jac_row[j]).abs();
        }
    }
}

fn accumulate_one_frame_sender_col(
    bf: &BetaFrame,
    gene_mtx: &Array2<f64>,
    jac_col: &[f64],
    lig_name: &str,
    gene_to_idx: &HashMap<&str, usize>,
    params: &CellCommNetworkParams,
    contrib: &mut [f64],
) {
    let n = contrib.len();
    debug_assert_eq!(jac_col.len(), n);
    let map = bf.cell_to_beta_row.as_slice();
    let scale = params.beta_scale_factor as f64;
    let min_e = params.min_expression;

    for k in 0..bf.ligands.len() {
        if bf.ligands[k] != lig_name {
            continue;
        }
        let Some(&rec_gi) = gene_to_idx.get(bf.receptors[k].as_str()) else {
            continue;
        };
        for i in 0..n {
            let br = unsafe { *map.get_unchecked(i) };
            let beta = f64::from(unsafe { *bf.lr_betas.uget((br, k)) });
            if beta == 0.0 {
                continue;
            }
            let rec = gene_mtx[[i, rec_gi]];
            if rec <= min_e {
                continue;
            }
            let s = beta * rec * scale;
            contrib[i] += (s * jac_col[i]).abs();
        }
    }

    for k in 0..bf.tfl_ligands.len() {
        if bf.tfl_ligands[k] != lig_name {
            continue;
        }
        let Some(&reg_gi) = gene_to_idx.get(bf.tfl_regulators[k].as_str()) else {
            continue;
        };
        for i in 0..n {
            let br = unsafe { *map.get_unchecked(i) };
            let beta = f64::from(unsafe { *bf.tfl_betas.uget((br, k)) });
            if beta == 0.0 {
                continue;
            }
            let reg = gene_mtx[[i, reg_gi]];
            if reg <= min_e {
                continue;
            }
            let s = beta * reg * scale;
            contrib[i] += (s * jac_col[i]).abs();
        }
    }
}

fn accumulate_one_frame(
    bf: &BetaFrame,
    gene_mtx: &Array2<f64>,
    jac: &Array2<f64>,
    lig_name: &str,
    gene_to_idx: &HashMap<&str, usize>,
    params: &CellCommNetworkParams,
    pair_contrib: &mut Array2<f64>,
) {
    let n = jac.nrows();
    let map = bf.cell_to_beta_row.as_slice();
    let scale = params.beta_scale_factor as f64;
    let min_e = params.min_expression;

    let n_lr = bf.ligands.len();
    for k in 0..n_lr {
        if bf.ligands[k] != lig_name {
            continue;
        }
        let Some(&rec_gi) = gene_to_idx.get(bf.receptors[k].as_str()) else {
            continue;
        };
        pair_contrib
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                let br = unsafe { *map.get_unchecked(i) };
                let beta = f64::from(unsafe { *bf.lr_betas.uget((br, k)) });
                if beta == 0.0 {
                    return;
                }
                let rec = gene_mtx[[i, rec_gi]];
                if rec <= min_e {
                    return;
                }
                let s = beta * rec * scale;
                for j in 0..n {
                    row[j] += (s * jac[[i, j]]).abs();
                }
            });
    }

    let n_tfl = bf.tfl_ligands.len();
    for k in 0..n_tfl {
        if bf.tfl_ligands[k] != lig_name {
            continue;
        }
        let Some(&reg_gi) = gene_to_idx.get(bf.tfl_regulators[k].as_str()) else {
            continue;
        };
        pair_contrib
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                let br = unsafe { *map.get_unchecked(i) };
                let beta = f64::from(unsafe { *bf.tfl_betas.uget((br, k)) });
                if beta == 0.0 {
                    return;
                }
                let reg = gene_mtx[[i, reg_gi]];
                if reg <= min_e {
                    return;
                }
                let s = beta * reg * scale;
                for j in 0..n {
                    row[j] += (s * jac[[i, j]]).abs();
                }
            });
    }
}
