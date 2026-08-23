//! Ligand-field LR terms for SpaceTravLR.
//!
//! Optional CellChat-style communication probabilities (Jin et al., Nat Commun 2021)
//! select LR pairs from an interaction DB. Received ligand is then aggregated as
//! either a **mean field** (global mean) or a **spatial field** (Gaussian neighborhood)
//! and multiplied by local receptor for the Lasso design matrix.

use crate::config::expand_user_path;
use crate::ligand::{
    WeightedLigandReduce, calculate_weighted_ligands, calculate_weighted_ligands_grid,
    calculate_weighted_ligands_with_cutoff_reduce,
};
use crate::network::SPACETRAVLR_DATA_DIR_ENV;
use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2, Array3, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// How received ligand is aggregated for LR columns.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LigandFieldMode {
    /// Mean field: \(X_{c,k}=\bar{L}^{(k)}\,R_c^{(k)}\) (global mean ligand).
    Meanfield,
    /// Spatial field: \(X_{c,k}=\widetilde{L}_c^{(k)}\,R_c^{(k)}\) (Gaussian neighborhood).
    #[default]
    #[serde(alias = "spatial_product")]
    Spatial,
}

/// How LR pairs are chosen before building design columns.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum PairSelectionMode {
    /// Rank/filter by CellChat-style communication probability \(P\) (default).
    #[default]
    Prob,
    /// Keep all DB pairs present in the AnnData (both L and R expressed); rank by
    /// expression product of group trimeans when applying `max_interactions`.
    Expressed,
}

/// Matched normalization of the received-ligand field before \(L\cdot R\).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReceivedLigandNorm {
    /// Legacy: spatial uses `(1/N)Σ w L`; meanfield uses global mean.
    #[default]
    GlobalN,
    /// Spatial uses kernel-mass `Σ w L / Σ w`; meanfield uses global mean (flat kernel-mass).
    /// Makes meanfield vs spatial a pure geometry contrast.
    KernelMass,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LigandFieldConfig {
    /// Optional path to `cellchat_{species}.csv` (`ligand,receptor,pathway,signaling`).
    pub db_path: Option<String>,
    /// Received-ligand aggregator: `spatial` (default, Gaussian) | `meanfield`.
    /// `spatial` requires 2D coordinates in `obsm` (`spatial` / `X_spatial` / `spatial_loc`).
    #[serde(alias = "lr_mode")]
    pub mode: LigandFieldMode,
    /// Multiplier on Gaussian kernel weights in spatial received-ligand aggregation.
    #[serde(default = "default_one_f64")]
    pub weighted_ligand_scale_factor: f64,
    /// Grid spacing as a fraction of `[spatial].radius` for approximate spatial aggregation.
    /// Omit for exact pairwise (slower on large N). Ignored when `received_ligand_norm = kernel_mass`.
    pub ligand_grid_factor: Option<f64>,
    /// Half-saturation \(K_h\) in the Hill / mass-action term (CellChat default 0.5).
    pub kh: f64,
    /// Hill coefficient \(n\) (CellChat default 1).
    pub hill_coef: f64,
    /// Drop groups with fewer than this many cells (CellChat `min.cells`).
    pub min_cells: usize,
    /// Weight probabilities by sender/receiver population fractions.
    pub population_size_weight: bool,
    /// Label-permutation nulls for p-values; `0` skips the test.
    pub n_perm: usize,
    /// Keep interactions with permutation p ≤ this (ignored when `n_perm == 0`).
    pub p_threshold: f64,
    /// Drop interactions whose max \(P\) (any sender→receiver) is below this.
    /// Ignored when `pair_selection = expressed`.
    pub min_prob: f64,
    /// When true, replace GRN `edge_type=lr` pairs with probability-selected interactions.
    pub replace_lr_pairs: bool,
    /// Cap retained interactions after filtering (by max \(P\) or expression product).
    pub max_interactions: Option<usize>,
    /// Restrict to these signaling classes (e.g. `Secreted Signaling`). Empty = all.
    #[serde(default)]
    pub signaling_types: Vec<String>,
    /// RNG seed for permutations.
    pub random_seed: u64,
    /// `prob` (default) ranks by CellChat \(P\); `expressed` keeps all present LR pairs.
    pub pair_selection: PairSelectionMode,
    /// Matched received-\(L\) normalization for fair meanfield ↔ spatial A/B.
    pub received_ligand_norm: ReceivedLigandNorm,
    /// Write `ligand_field_L_diagnostics.csv` next to commun_prob when an output dir is set.
    #[serde(default = "default_true")]
    pub write_ligand_diagnostics: bool,
}

fn default_true() -> bool {
    true
}

fn default_one_f64() -> f64 {
    1.0
}

impl Default for LigandFieldConfig {
    fn default() -> Self {
        Self {
            db_path: None,
            mode: LigandFieldMode::Spatial,
            weighted_ligand_scale_factor: 1.0,
            ligand_grid_factor: None,
            kh: 0.5,
            hill_coef: 1.0,
            min_cells: 10,
            population_size_weight: false,
            n_perm: 0,
            p_threshold: 0.05,
            min_prob: 0.0,
            replace_lr_pairs: true,
            max_interactions: Some(200),
            signaling_types: Vec::new(),
            random_seed: 42,
            pair_selection: PairSelectionMode::Prob,
            received_ligand_norm: ReceivedLigandNorm::GlobalN,
            write_ligand_diagnostics: true,
        }
    }
}

/// One CellChatDB interaction after expansion to independent single-gene units.
///
/// CellChatDB stores multi-subunit complexes (e.g. `Tgfbr1_Tgfbr2`). For SpaceTravLR
/// compatibility we expand each complex row into the cartesian product of ligand ×
/// receptor subunits, each becoming a standard `Lig$Rec` modulator column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellChatInteraction {
    pub ligand_subunits: Vec<String>,
    pub receptor_subunits: Vec<String>,
    pub pathway: String,
    pub signaling: String,
    /// SpaceTravLR column name: single-gene `Lig$Rec`.
    pub pair_name: String,
}

impl CellChatInteraction {
    /// Parse a raw CellChatDB row (complexes may still be multi-subunit here).
    pub fn from_row(ligand: &str, receptor: &str, pathway: &str, signaling: &str) -> Self {
        let ligand_subunits = split_complex(ligand);
        let receptor_subunits = split_complex(receptor);
        let pair_name = format!(
            "{}${}",
            ligand_subunits.join("_"),
            receptor_subunits.join("_")
        );
        Self {
            ligand_subunits,
            receptor_subunits,
            pathway: pathway.to_string(),
            signaling: signaling.to_string(),
            pair_name,
        }
    }

    /// Single-gene ligand (after [`expand_complexes_to_independent_units`]).
    pub fn ligand(&self) -> &str {
        self.ligand_subunits
            .first()
            .map(String::as_str)
            .unwrap_or("")
    }

    /// Single-gene receptor (after expansion).
    pub fn receptor(&self) -> &str {
        self.receptor_subunits
            .first()
            .map(String::as_str)
            .unwrap_or("")
    }

    fn singleton(ligand: String, receptor: String, pathway: &str, signaling: &str) -> Self {
        let pair_name = format!("{ligand}${receptor}");
        Self {
            ligand_subunits: vec![ligand],
            receptor_subunits: vec![receptor],
            pathway: pathway.to_string(),
            signaling: signaling.to_string(),
            pair_name,
        }
    }
}

fn split_complex(s: &str) -> Vec<String> {
    s.split('_')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| p.to_string())
        .collect()
}

/// Expand multi-subunit CellChat complexes into independent `Lig$Rec` units
/// (cartesian product of ligand × receptor subunits), deduplicated by pair name.
///
/// Example: `Tgfb1` × `Tgfbr1_Tgfbr2` → `Tgfb1$Tgfbr1`, `Tgfb1$Tgfbr2`.
pub fn expand_complexes_to_independent_units(
    interactions: &[CellChatInteraction],
) -> Vec<CellChatInteraction> {
    let mut out = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for inter in interactions {
        if inter.ligand_subunits.is_empty() || inter.receptor_subunits.is_empty() {
            continue;
        }
        for lig in &inter.ligand_subunits {
            for rec in &inter.receptor_subunits {
                let unit = CellChatInteraction::singleton(
                    lig.clone(),
                    rec.clone(),
                    &inter.pathway,
                    &inter.signaling,
                );
                if seen.insert(unit.pair_name.clone()) {
                    out.push(unit);
                }
            }
        }
    }
    out
}

/// Tukey's trimean \((Q_1 + 2 Q_2 + Q_3)/4\).
pub fn tri_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = percentile_sorted(&v, 0.25);
    let q2 = percentile_sorted(&v, 0.50);
    let q3 = percentile_sorted(&v, 0.75);
    (q1 + 2.0 * q2 + q3) / 4.0
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = idx - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

/// Geometric mean of non-negative values; zeros propagate (CellChat complex rule).
pub fn geometric_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    if values.iter().any(|&x| x <= 0.0 || !x.is_finite()) {
        return 0.0;
    }
    let n = values.len() as f64;
    let log_sum: f64 = values.iter().map(|x| x.ln()).sum();
    (log_sum / n).exp()
}

/// Hill / mass-action communication probability base term.
pub fn hill_commun_prob(ligand: f64, receptor: f64, kh: f64, hill_coef: f64) -> f64 {
    let lr = ligand * receptor;
    if !lr.is_finite() || lr <= 0.0 {
        return 0.0;
    }
    let kh = kh.max(1e-12);
    let n = hill_coef.max(1e-12);
    let num = lr.powf(n);
    let den = kh.powf(n) + num;
    if den <= 0.0 || !den.is_finite() {
        return 0.0;
    }
    num / den
}

/// Load CellChatDB CSV (`ligand,receptor,pathway,signaling`).
pub fn load_cellchat_db(path: &Path) -> Result<Vec<CellChatInteraction>> {
    let f = File::open(path).with_context(|| format!("open CellChatDB {}", path.display()))?;
    let reader = BufReader::new(f);
    let mut lines = reader.lines();
    let header = lines
        .next()
        .transpose()?
        .ok_or_else(|| anyhow::anyhow!("empty CellChatDB {}", path.display()))?;
    let cols: Vec<String> = header
        .split(',')
        .map(|s| s.trim().to_ascii_lowercase())
        .collect();
    let idx = |name: &str| -> Result<usize> {
        cols.iter()
            .position(|c| c == name)
            .ok_or_else(|| anyhow::anyhow!("CellChatDB missing column {name:?} in {}", path.display()))
    };
    let i_lig = idx("ligand")?;
    let i_rec = idx("receptor")?;
    let i_path = idx("pathway")?;
    let i_sig = idx("signaling")?;

    let mut out = Vec::new();
    for (lineno, line) in lines.enumerate() {
        let line = line.with_context(|| format!("read line {} of {}", lineno + 2, path.display()))?;
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = t.split(',').map(str::trim).collect();
        if parts.len() <= i_sig.max(i_path).max(i_rec).max(i_lig) {
            continue;
        }
        let inter = CellChatInteraction::from_row(
            parts[i_lig],
            parts[i_rec],
            parts[i_path],
            parts[i_sig],
        );
        if inter.ligand_subunits.is_empty() || inter.receptor_subunits.is_empty() {
            continue;
        }
        out.push(inter);
    }
    Ok(out) // keep multi-subunit complexes intact for CellChat-style P
}

/// Resolve `cellchat_{species}.csv` from config path, env, or `data/` search.
pub fn resolve_cellchat_db_path(
    species: &str,
    config_db_path: Option<&str>,
    config_file_parent: Option<&Path>,
) -> Result<PathBuf> {
    let mut tried = Vec::new();
    if let Some(raw) = config_db_path.map(str::trim).filter(|s| !s.is_empty()) {
        let exp = expand_user_path(raw);
        let pb = PathBuf::from(&exp);
        let cand = if pb.is_absolute() {
            pb
        } else if let Some(parent) = config_file_parent {
            parent.join(pb)
        } else {
            pb
        };
        tried.push(cand.display().to_string());
        if cand.is_file() {
            return Ok(cand);
        }
    }

    let filename = format!("cellchat_{species}.csv");
    if let Ok(dir) = std::env::var(SPACETRAVLR_DATA_DIR_ENV) {
        let cand = PathBuf::from(expand_user_path(dir.trim())).join(&filename);
        tried.push(cand.display().to_string());
        if cand.is_file() {
            return Ok(cand);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            for rel in ["data", "../data"] {
                let cand = parent.join(rel).join(&filename);
                tried.push(cand.display().to_string());
                if cand.is_file() {
                    return Ok(cand);
                }
            }
        }
    }

    let mut dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    for _ in 0..8 {
        let cand = dir.join("data").join(&filename);
        tried.push(cand.display().to_string());
        if cand.is_file() {
            return Ok(cand);
        }
        if !dir.pop() {
            break;
        }
    }

    bail!(
        "Could not find CellChatDB {filename:?}. Set [ligand_field].db_path or {SPACETRAVLR_DATA_DIR_ENV}. Tried:\n  {}",
        tried.join("\n  ")
    )
}

#[derive(Debug, Clone)]
pub struct CellChatProbResult {
    pub group_names: Vec<String>,
    pub interactions: Vec<CellChatInteraction>,
    /// Shape `(n_interactions, n_groups, n_groups)` — `[k][sender][receiver]`.
    pub prob: Array3<f64>,
    /// Same shape; `None` when permutations were skipped.
    pub pvalues: Option<Array3<f64>>,
    pub group_counts: Vec<usize>,
}

impl CellChatProbResult {
    pub fn n_groups(&self) -> usize {
        self.group_names.len()
    }

    pub fn max_prob_for_interaction(&self, k: usize) -> f64 {
        self.prob
            .index_axis(Axis(0), k)
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
    }

    /// Incoming strength to receiver group `j` for interaction `k`: \(\sum_i P_{i\to j}^k\).
    pub fn incoming_strength(&self, k: usize, receiver: usize) -> f64 {
        let n = self.n_groups();
        (0..n).map(|i| self.prob[[k, i, receiver]]).sum()
    }
}

/// Filter DB interactions to genes present in `var_names` and optional signaling classes.
pub fn filter_interactions_for_adata(
    db: &[CellChatInteraction],
    var_names: &HashSet<String>,
    signaling_types: &[String],
) -> Vec<CellChatInteraction> {
    let sig_filter: HashSet<String> = signaling_types
        .iter()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    db.iter()
        .filter(|inter| {
            if !sig_filter.is_empty()
                && !sig_filter.contains(&inter.signaling.to_ascii_lowercase())
            {
                return false;
            }
            inter
                .ligand_subunits
                .iter()
                .chain(inter.receptor_subunits.iter())
                .all(|g| var_names.contains(g))
        })
        .cloned()
        .collect()
}

fn group_trimean_expression(
    expr: &Array2<f64>,
    group_ids: &[usize],
    n_groups: usize,
    min_cells: usize,
) -> (Array2<f64>, Vec<usize>) {
    let n_genes = expr.ncols();
    let mut counts = vec![0usize; n_groups];
    for &g in group_ids {
        if g < n_groups {
            counts[g] += 1;
        }
    }
    let mut out = Array2::<f64>::zeros((n_groups, n_genes));
    for g in 0..n_groups {
        if counts[g] < min_cells {
            continue;
        }
        for gene in 0..n_genes {
            let mut vals = Vec::with_capacity(counts[g]);
            for (i, &gi) in group_ids.iter().enumerate() {
                if gi == g {
                    vals.push(expr[[i, gene]]);
                }
            }
            out[[g, gene]] = tri_mean(&vals);
        }
    }
    (out, counts)
}

fn complex_level(
    group_expr: &Array2<f64>,
    group: usize,
    subunits: &[String],
    gene_to_col: &HashMap<&str, usize>,
) -> f64 {
    let mut vals = Vec::with_capacity(subunits.len());
    for s in subunits {
        let Some(&col) = gene_to_col.get(s.as_str()) else {
            return 0.0;
        };
        vals.push(group_expr[[group, col]]);
    }
    geometric_mean(&vals)
}

fn fill_prob_tensor(
    group_expr: &Array2<f64>,
    counts: &[usize],
    interactions: &[CellChatInteraction],
    gene_to_col: &HashMap<&str, usize>,
    kh: f64,
    hill_coef: f64,
    population_size_weight: bool,
    min_cells: usize,
) -> Array3<f64> {
    let n_g = counts.len();
    let n_k = interactions.len();
    let n_cells: usize = counts.iter().sum();
    let mut prob = Array3::<f64>::zeros((n_k, n_g, n_g));
    if n_cells == 0 {
        return prob;
    }
    for (k, inter) in interactions.iter().enumerate() {
        for i in 0..n_g {
            if counts[i] < min_cells {
                continue;
            }
            let lig = complex_level(group_expr, i, &inter.ligand_subunits, gene_to_col);
            if lig <= 0.0 {
                continue;
            }
            for j in 0..n_g {
                if counts[j] < min_cells {
                    continue;
                }
                let rec = complex_level(group_expr, j, &inter.receptor_subunits, gene_to_col);
                let mut p = hill_commun_prob(lig, rec, kh, hill_coef);
                if population_size_weight {
                    let wi = counts[i] as f64 / n_cells as f64;
                    let wj = counts[j] as f64 / n_cells as f64;
                    p *= wi * wj;
                }
                prob[[k, i, j]] = p;
            }
        }
    }
    prob
}

/// Compute CellChat communication probabilities (and optional permutation p-values).
///
/// Follows Jin et al.: scale expression by its global max (`data/max(data)`), then
/// group trimeans, geometric-mean complexes, and Hill \(P=(LR)^n/(K_h^n+(LR)^n)\).
/// Cofactor / agonist–antagonist terms from full CellChatDB are not applied.
pub fn compute_commun_prob(
    expr: &Array2<f64>,
    gene_names: &[String],
    group_ids: &[usize],
    group_names: &[String],
    interactions: &[CellChatInteraction],
    cfg: &LigandFieldConfig,
) -> Result<CellChatProbResult> {
    if expr.nrows() != group_ids.len() {
        bail!(
            "CellChat expr rows ({}) != group_ids ({})",
            expr.nrows(),
            group_ids.len()
        );
    }
    if expr.ncols() != gene_names.len() {
        bail!(
            "CellChat expr cols ({}) != gene_names ({})",
            expr.ncols(),
            gene_names.len()
        );
    }
    let n_groups = group_names.len();
    let gene_to_col: HashMap<&str, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();

    // CellChat: data.use <- data/max(data) so Kh=0.5 is on a [0,1]-scaled matrix.
    let max_v = expr.iter().copied().fold(0.0_f64, f64::max);
    let expr_scaled = if max_v > 0.0 && max_v.is_finite() {
        expr * (1.0 / max_v)
    } else {
        expr.clone()
    };

    let (group_expr, counts) =
        group_trimean_expression(&expr_scaled, group_ids, n_groups, cfg.min_cells);
    let prob = fill_prob_tensor(
        &group_expr,
        &counts,
        interactions,
        &gene_to_col,
        cfg.kh,
        cfg.hill_coef,
        cfg.population_size_weight,
        cfg.min_cells,
    );

    let pvalues = if cfg.n_perm > 0 && !interactions.is_empty() {
        let mut rng = StdRng::seed_from_u64(cfg.random_seed);
        let mut exceed = Array3::<f64>::zeros(prob.raw_dim());
        let mut shuffled = group_ids.to_vec();
        for _ in 0..cfg.n_perm {
            for i in (1..shuffled.len()).rev() {
                let j = rng.gen_range(0..=i);
                shuffled.swap(i, j);
            }
            let (ge, ct) =
                group_trimean_expression(&expr_scaled, &shuffled, n_groups, cfg.min_cells);
            let null = fill_prob_tensor(
                &ge,
                &ct,
                interactions,
                &gene_to_col,
                cfg.kh,
                cfg.hill_coef,
                cfg.population_size_weight,
                cfg.min_cells,
            );
            ndarray::Zip::from(&mut exceed)
                .and(&null)
                .and(&prob)
                .for_each(|e, &n, &o| {
                    if n >= o {
                        *e += 1.0;
                    }
                });
        }
        let n_perm = cfg.n_perm as f64;
        Some(exceed.mapv(|e| (e + 1.0) / (n_perm + 1.0)))
    } else {
        None
    };

    Ok(CellChatProbResult {
        group_names: group_names.to_vec(),
        interactions: interactions.to_vec(),
        prob,
        pvalues,
        group_counts: counts,
    })
}

/// Keep significant / strong interactions; optionally cap by max \(P\) or expression product.
pub fn select_interactions(
    result: &CellChatProbResult,
    cfg: &LigandFieldConfig,
) -> Vec<usize> {
    select_interactions_with_expr_scores(result, cfg, None)
}

/// Like [`select_interactions`], with optional per-interaction expression-product scores
/// (used when `pair_selection = expressed`).
pub fn select_interactions_with_expr_scores(
    result: &CellChatProbResult,
    cfg: &LigandFieldConfig,
    expr_product_scores: Option<&[f64]>,
) -> Vec<usize> {
    let mut keep: Vec<(usize, f64)> = Vec::new();
    match cfg.pair_selection {
        PairSelectionMode::Prob => {
            for k in 0..result.interactions.len() {
                let max_p = result.max_prob_for_interaction(k);
                if max_p < cfg.min_prob {
                    continue;
                }
                if let Some(ref pvals) = result.pvalues {
                    let min_pval = pvals
                        .index_axis(Axis(0), k)
                        .iter()
                        .copied()
                        .fold(1.0_f64, f64::min);
                    if min_pval > cfg.p_threshold {
                        continue;
                    }
                }
                keep.push((k, max_p));
            }
        }
        PairSelectionMode::Expressed => {
            for k in 0..result.interactions.len() {
                let score = expr_product_scores
                    .and_then(|s| s.get(k).copied())
                    .unwrap_or_else(|| result.max_prob_for_interaction(k));
                keep.push((k, score));
            }
        }
    }
    keep.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(cap) = cfg.max_interactions {
        keep.truncate(cap);
    }
    keep.into_iter().map(|(k, _)| k).collect()
}

/// Per-interaction score = (mean ligand complex expr) × (mean receptor complex expr).
pub fn interaction_mean_expr_product_scores(
    expr: &Array2<f64>,
    gene_names: &[String],
    interactions: &[CellChatInteraction],
) -> Vec<f64> {
    let mut gene_to_idx: HashMap<String, usize> = HashMap::new();
    for (i, g) in gene_names.iter().enumerate() {
        gene_to_idx.insert(g.clone(), i);
    }
    interactions
        .iter()
        .map(|inter| {
            let lig = receptor_complex_expr(expr, &gene_to_idx, &inter.ligand_subunits)
                .map(|v| v.mean().unwrap_or(0.0))
                .unwrap_or(0.0);
            let rec = receptor_complex_expr(expr, &gene_to_idx, &inter.receptor_subunits)
                .map(|v| v.mean().unwrap_or(0.0))
                .unwrap_or(0.0);
            lig * rec
        })
        .collect()
}

/// Shared plan consumed by per-gene Lasso workers for ligand-field LR terms.
#[derive(Debug, Clone)]
pub struct LigandFieldPlan {
    pub mode: LigandFieldMode,
    pub kh: f64,
    pub hill_coef: f64,
    pub replace_lr_pairs: bool,
    /// Min cells per group when computing mean-field ligand trimeans.
    pub min_cells: usize,
    pub received_ligand_norm: ReceivedLigandNorm,
    /// Selected interactions (already filtered).
    pub interactions: Vec<CellChatInteraction>,
    /// Group label per cell (aligned to training obs rows / xy).
    pub cell_group: Vec<usize>,
    pub group_names: Vec<String>,
    /// `(n_interactions, n_groups, n_groups)` for selected interactions only.
    pub prob: Array3<f64>,
    pub pair_names: Vec<String>,
    /// Precomputed received-\(L\) fields keyed by ligand gene (shared across workers).
    pub received_ligand_cache: Option<HashMap<String, Arc<Array1<f64>>>>,
}

impl LigandFieldPlan {
    /// Build a Lasso plan from selected **complex-level** interactions.
    ///
    /// Each selected CellChat interaction is expanded into independent `Lig$Rec`
    /// units for SpaceTravLR columns; the parent \(P_{s\to t}\) slice is copied
    /// to every child (so multi-subunit complexes keep a single communication
    /// probability, while local receptor subunits may still differ in \(X\)).
    pub fn from_selected(result: CellChatProbResult, selected: &[usize], cfg: &LigandFieldConfig) -> Self {
        let n_g = result.n_groups();
        let mut interactions = Vec::new();
        let mut pair_names = Vec::new();
        let mut parent_of: Vec<usize> = Vec::new();
        for &old_k in selected {
            let units =
                expand_complexes_to_independent_units(std::slice::from_ref(&result.interactions[old_k]));
            for unit in units {
                pair_names.push(unit.pair_name.clone());
                interactions.push(unit);
                parent_of.push(old_k);
            }
        }
        let mut prob = Array3::<f64>::zeros((interactions.len(), n_g, n_g));
        for (new_k, &old_k) in parent_of.iter().enumerate() {
            for i in 0..n_g {
                for j in 0..n_g {
                    prob[[new_k, i, j]] = result.prob[[old_k, i, j]];
                }
            }
        }
        Self {
            mode: cfg.mode,
            kh: cfg.kh,
            hill_coef: cfg.hill_coef,
            replace_lr_pairs: cfg.replace_lr_pairs,
            min_cells: cfg.min_cells,
            received_ligand_norm: cfg.received_ligand_norm,
            interactions,
            cell_group: Vec::new(),
            group_names: result.group_names,
            prob,
            pair_names,
            received_ligand_cache: None,
        }
    }

    pub fn with_cell_groups(mut self, cell_group: Vec<usize>) -> Self {
        self.cell_group = cell_group;
        self
    }

    pub fn lr_pairs_as_extra(&self) -> Vec<(String, String)> {
        self.interactions
            .iter()
            .map(|inter| (inter.ligand().to_string(), inter.receptor().to_string()))
            .collect()
    }

    /// Slice the plan to `keep_idx` while preserving the shared ligand cache.
    pub fn filtered_to_indices(&self, keep_idx: &[usize]) -> Self {
        let n_g = self.group_names.len();
        let mut interactions = Vec::with_capacity(keep_idx.len());
        let mut pair_names = Vec::with_capacity(keep_idx.len());
        let mut prob = Array3::<f64>::zeros((keep_idx.len(), n_g, n_g));
        for (new_k, &old_k) in keep_idx.iter().enumerate() {
            interactions.push(self.interactions[old_k].clone());
            pair_names.push(self.pair_names[old_k].clone());
            for i in 0..n_g {
                for j in 0..n_g {
                    prob[[new_k, i, j]] = self.prob[[old_k, i, j]];
                }
            }
        }
        let mut cache = None;
        if let Some(ref src) = self.received_ligand_cache {
            let mut m = HashMap::new();
            for inter in &interactions {
                let lig = inter.ligand();
                if let Some(v) = src.get(lig) {
                    m.insert(lig.to_string(), Arc::clone(v));
                }
            }
            cache = Some(m);
        }
        Self {
            mode: self.mode,
            kh: self.kh,
            hill_coef: self.hill_coef,
            replace_lr_pairs: self.replace_lr_pairs,
            min_cells: self.min_cells,
            received_ligand_norm: self.received_ligand_norm,
            interactions,
            cell_group: self.cell_group.clone(),
            group_names: self.group_names.clone(),
            prob,
            pair_names,
            received_ligand_cache: cache,
        }
    }
}

fn geom_mean_columns(mat: &Array2<f64>, cols: &[usize]) -> Array1<f64> {
    let n = mat.nrows();
    let mut out = Array1::<f64>::zeros(n);
    if cols.is_empty() {
        return out;
    }
    for i in 0..n {
        let vals: Vec<f64> = cols.iter().map(|&c| mat[[i, c]]).collect();
        out[i] = geometric_mean(&vals);
    }
    out
}

fn receptor_complex_expr(
    expr: &Array2<f64>,
    gene_to_idx: &HashMap<String, usize>,
    subunits: &[String],
) -> Option<Array1<f64>> {
    let cols: Vec<usize> = subunits
        .iter()
        .map(|g| gene_to_idx.get(g).copied())
        .collect::<Option<Vec<_>>>()?;
    Some(geom_mean_columns(expr, &cols))
}

fn received_ligand_field(
    xy: &Array2<f64>,
    ligand_expr: &Array1<f64>,
    radius: f64,
    scale_factor: f64,
    grid_factor: Option<f64>,
    norm: ReceivedLigandNorm,
) -> Array1<f64> {
    let n = xy.nrows();
    let mut lig = Array2::<f64>::zeros((n, 1));
    lig.column_mut(0).assign(ligand_expr);
    let reduce = match norm {
        ReceivedLigandNorm::GlobalN => WeightedLigandReduce::GlobalN,
        ReceivedLigandNorm::KernelMass => WeightedLigandReduce::KernelMass,
    };
    let recv = match (norm, grid_factor.filter(|g| g.is_finite() && *g > 0.0)) {
        // Kernel-mass: exact pairwise with a soft spatial cutoff (3·radius) for tractability.
        (ReceivedLigandNorm::KernelMass, _) => {
            let cutoff = if radius.is_finite() && radius > 0.0 {
                Some(3.0 * radius)
            } else {
                None
            };
            calculate_weighted_ligands_with_cutoff_reduce(
                xy, &lig, radius, scale_factor, cutoff, reduce,
            )
        }
        (ReceivedLigandNorm::GlobalN, Some(gf)) => {
            calculate_weighted_ligands_grid(xy, &lig, radius, scale_factor, gf)
        }
        (ReceivedLigandNorm::GlobalN, None) => {
            calculate_weighted_ligands(xy, &lig, radius, scale_factor)
        }
    };
    recv.column(0).to_owned()
}

/// Precompute unique-ligand received fields into `plan.received_ligand_cache`.
pub fn precompute_received_ligand_cache(
    plan: &mut LigandFieldPlan,
    xy: &Array2<f64>,
    expr: &Array2<f64>,
    gene_to_idx: &HashMap<String, usize>,
    radius: f64,
    scale_factor: f64,
    grid_factor: Option<f64>,
) -> Result<()> {
    let n = xy.nrows();
    let mut cache: HashMap<String, Arc<Array1<f64>>> = HashMap::new();
    let mut ligands: Vec<String> = plan
        .interactions
        .iter()
        .map(|i| i.ligand().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    ligands.sort();
    ligands.dedup();

    let grid_factor = grid_factor.or_else(|| {
        if n > 5_000 {
            Some(0.5)
        } else {
            None
        }
    });

    for lig_name in ligands {
        let idx = gene_to_idx
            .get(&lig_name)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("CellChat gene {lig_name} missing from expr map"))?;
        let lig_expr = expr.column(idx).to_owned();
        let field = match plan.mode {
            LigandFieldMode::Spatial => received_ligand_field(
                xy,
                &lig_expr,
                radius,
                scale_factor,
                grid_factor,
                plan.received_ligand_norm,
            ),
            LigandFieldMode::Meanfield => {
                let l_mf = lig_expr.mean().unwrap_or(0.0);
                Array1::from_elem(n, l_mf)
            }
        };
        cache.insert(lig_name, Arc::new(field));
    }
    plan.received_ligand_cache = Some(cache);
    Ok(())
}

/// Write per-ligand meanfield vs spatial field diagnostics.
///
/// Spatial field uses a **grid approximation** (or legacy global_n exact) for speed —
/// diagnostics are not the training path. Training still uses `received_ligand_norm`.
pub fn write_ligand_field_diagnostics_csv(
    path: &Path,
    xy: &Array2<f64>,
    expr: &Array2<f64>,
    gene_to_idx: &HashMap<String, usize>,
    ligand_genes: &[String],
    radius: f64,
    scale_factor: f64,
    grid_factor: Option<f64>,
    _norm: ReceivedLigandNorm,
) -> Result<()> {
    use std::io::Write;
    let mut f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    writeln!(
        f,
        "ligand,L_bar,L_spatial_mean,L_spatial_std,mae_sp_minus_mf,mae_rel,rel_l2_sp_vs_mf,cv_spatial,frac_cells_sp_gt_mf"
    )?;
    let n = xy.nrows();
    // Prefer grid for diagnostics regardless of training norm (exact kernel_mass is O(N²·L)).
    let grid_factor = Some(grid_factor.filter(|g| g.is_finite() && *g > 0.0).unwrap_or(0.5));
    for lig in ligand_genes {
        let Some(&idx) = gene_to_idx.get(lig) else {
            continue;
        };
        let lig_expr = expr.column(idx).to_owned();
        let l_bar = lig_expr.mean().unwrap_or(0.0);
        let sp = received_ligand_field(
            xy,
            &lig_expr,
            radius,
            scale_factor,
            grid_factor,
            ReceivedLigandNorm::GlobalN,
        );
        let sp_mean = sp.mean().unwrap_or(0.0);
        let sp_std = {
            let var = sp.iter().map(|x| (x - sp_mean).powi(2)).sum::<f64>() / n.max(1) as f64;
            var.sqrt()
        };
        let mae = sp
            .iter()
            .map(|&x| (x - l_bar).abs())
            .sum::<f64>()
            / n.max(1) as f64;
        let mae_rel = if l_bar.abs() > 1e-12 {
            mae / l_bar.abs()
        } else {
            f64::NAN
        };
        let l2_sp = sp.iter().map(|x| x * x).sum::<f64>().sqrt();
        let l2_diff = sp
            .iter()
            .map(|&x| (x - l_bar).powi(2))
            .sum::<f64>()
            .sqrt();
        let rel_l2 = if l2_sp > 1e-12 {
            l2_diff / l2_sp
        } else {
            f64::NAN
        };
        let cv = if sp_mean.abs() > 1e-12 {
            sp_std / sp_mean.abs()
        } else {
            f64::NAN
        };
        let frac_gt = sp.iter().filter(|&&x| x > l_bar).count() as f64 / n.max(1) as f64;
        writeln!(
            f,
            "{lig},{l_bar},{sp_mean},{sp_std},{mae},{mae_rel},{rel_l2},{cv},{frac_gt}"
        )?;
    }
    Ok(())
}

/// Build per-cell LR design columns for the plan's selected interactions.
///
/// `expr` must contain all ligand/receptor subunit genes (columns indexed by `gene_to_idx`).
/// Returns `(n_cells × n_interactions)`.
///
/// When `grid_factor` is `Some(g)` with `g > 0`, received-ligand fields use the grid
/// approximation (same as SpaceTravLR training for large N). Unique ligand genes are
/// cached so shared ligands across interactions are not recomputed. Prefer filling
/// [`LigandFieldPlan::received_ligand_cache`] once via [`precompute_received_ligand_cache`].
pub fn build_hybrid_lr_matrix(
    plan: &LigandFieldPlan,
    xy: &Array2<f64>,
    expr: &Array2<f64>,
    gene_to_idx: &HashMap<String, usize>,
    radius: f64,
    scale_factor: f64,
) -> Result<Array2<f64>> {
    build_hybrid_lr_matrix_with_grid(plan, xy, expr, gene_to_idx, radius, scale_factor, None)
}

/// Like [`build_hybrid_lr_matrix`], with optional ligand grid factor.
pub fn build_hybrid_lr_matrix_with_grid(
    plan: &LigandFieldPlan,
    xy: &Array2<f64>,
    expr: &Array2<f64>,
    gene_to_idx: &HashMap<String, usize>,
    radius: f64,
    scale_factor: f64,
    grid_factor: Option<f64>,
) -> Result<Array2<f64>> {
    let n = xy.nrows();
    let n_k = plan.interactions.len();
    if plan.cell_group.len() != n {
        bail!(
            "CellChat plan cell_group len {} != n_cells {}",
            plan.cell_group.len(),
            n
        );
    }
    let mut out = Array2::<f64>::zeros((n, n_k));
    if n_k == 0 {
        return Ok(out);
    }

    let grid_factor = grid_factor.or_else(|| {
        if n > 5_000 {
            Some(0.5)
        } else {
            None
        }
    });

    let mut subunit_expr_cache: HashMap<String, Array1<f64>> = HashMap::new();
    let get_gene = |name: &str,
                        cache: &mut HashMap<String, Array1<f64>>|
     -> Result<Array1<f64>> {
        if let Some(v) = cache.get(name) {
            return Ok(v.clone());
        }
        let idx = gene_to_idx
            .get(name)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("CellChat gene {name} missing from expr map"))?;
        let col = expr.column(idx).to_owned();
        cache.insert(name.to_string(), col.clone());
        Ok(col)
    };

    let mut field_cache: HashMap<String, Array1<f64>> = HashMap::new();
    if let Some(ref shared) = plan.received_ligand_cache {
        for (k, v) in shared {
            field_cache.insert(k.clone(), v.as_ref().clone());
        }
    }

    for (k, inter) in plan.interactions.iter().enumerate() {
        let lig_name = inter.ligand().to_string();
        if lig_name.is_empty() {
            continue;
        }
        let lig_expr = get_gene(&lig_name, &mut subunit_expr_cache)?;
        let rec = receptor_complex_expr(expr, gene_to_idx, &inter.receptor_subunits)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "CellChat receptor complex {:?} missing subunits in expr",
                    inter.receptor_subunits
                )
            })?;

        let recv = if let Some(cached) = field_cache.get(&lig_name) {
            cached.clone()
        } else {
            let computed = match plan.mode {
                LigandFieldMode::Spatial => received_ligand_field(
                    xy,
                    &lig_expr,
                    radius,
                    scale_factor,
                    grid_factor,
                    plan.received_ligand_norm,
                ),
                LigandFieldMode::Meanfield => {
                    let l_mf = if !lig_expr.is_empty() {
                        lig_expr.mean().unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    Array1::from_elem(n, l_mf)
                }
            };
            field_cache.insert(lig_name.clone(), computed.clone());
            computed
        };
        for i in 0..n {
            out[[i, k]] = recv[i] * rec[i];
        }
    }
    Ok(out)
}

/// Map cluster integer ids to dense 0..G-1 group indices + names.
pub fn encode_groups_from_labels(labels: &[String]) -> (Vec<usize>, Vec<String>) {
    let mut names: Vec<String> = Vec::new();
    let mut map: HashMap<String, usize> = HashMap::new();
    let mut ids = Vec::with_capacity(labels.len());
    for lab in labels {
        let e = map.entry(lab.clone()).or_insert_with(|| {
            let id = names.len();
            names.push(lab.clone());
            id
        });
        ids.push(*e);
    }
    (ids, names)
}

/// Write a long-format CSV of \(P_{i\to j}^k\) (and optional p-values) for inspection.
pub fn write_prob_csv(path: &Path, result: &CellChatProbResult, selected: Option<&[usize]>) -> Result<()> {
    use std::io::Write;
    let mut f = File::create(path).with_context(|| format!("create {}", path.display()))?;
    writeln!(
        f,
        "interaction,ligand,receptor,pathway,signaling,sender,receiver,prob,pvalue"
    )?;
    let ks: Vec<usize> = match selected {
        Some(s) => s.to_vec(),
        None => (0..result.interactions.len()).collect(),
    };
    let n_g = result.n_groups();
    for k in ks {
        let inter = &result.interactions[k];
        for i in 0..n_g {
            for j in 0..n_g {
                let p = result.prob[[k, i, j]];
                if p <= 0.0 {
                    continue;
                }
                let pv = result
                    .pvalues
                    .as_ref()
                    .map(|a| a[[k, i, j]])
                    .map(|x| format!("{x}"))
                    .unwrap_or_default();
                writeln!(
                    f,
                    "{},{},{},{},{},{},{},{},{}",
                    inter.pair_name,
                    inter.ligand(),
                    inter.receptor(),
                    inter.pathway,
                    inter.signaling,
                    result.group_names[i],
                    result.group_names[j],
                    p,
                    pv
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn tri_mean_known_values() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        // Q1=2, Q2=3, Q3=4 → (2+6+4)/4 = 3
        assert_abs_diff_eq!(tri_mean(&v), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn geometric_mean_zero_propagates() {
        assert_eq!(geometric_mean(&[1.0, 0.0, 2.0]), 0.0);
        assert_abs_diff_eq!(geometric_mean(&[4.0, 1.0]), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn hill_saturates() {
        let p0 = hill_commun_prob(0.0, 1.0, 0.5, 1.0);
        assert_eq!(p0, 0.0);
        let p_half = hill_commun_prob(1.0, 0.5, 0.5, 1.0); // LR=0.5 = Kh
        assert_abs_diff_eq!(p_half, 0.5, epsilon = 1e-10);
        let p_hi = hill_commun_prob(10.0, 10.0, 0.5, 1.0);
        assert!(p_hi > 0.99);
    }

    #[test]
    fn commun_prob_sender_receiver_asymmetric() {
        // 2 groups, 2 genes (L, R). Group0 expresses L, group1 expresses R.
        let gene_names = vec!["L".into(), "R".into()];
        let expr = array![
            [2.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [0.0, 2.0],
            [0.0, 2.0],
        ];
        let group_ids = vec![0, 0, 0, 1, 1, 1];
        let group_names = vec!["A".into(), "B".into()];
        let inter = CellChatInteraction::from_row("L", "R", "Test", "Secreted Signaling");
        let cfg = LigandFieldConfig {
            min_cells: 2,
            n_perm: 0,
            ..Default::default()
        };
        let res = compute_commun_prob(
            &expr,
            &gene_names,
            &group_ids,
            &group_names,
            &[inter],
            &cfg,
        )
        .unwrap();
        // A→B should be strong; B→A ~0; A→A and B→B ~0
        assert!(res.prob[[0, 0, 1]] > 0.5);
        assert!(res.prob[[0, 1, 0]] < 1e-9);
        assert!(res.prob[[0, 0, 0]] < 1e-9);
        assert!(res.prob[[0, 1, 1]] < 1e-9);
    }

    #[test]
    fn hybrid_spatial_uses_neighborhood_ligand() {
        // Nearby senders (x=0) vs receivers (x=1): spatial field > 0 on receivers.
        let xy = array![[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]];
        let expr = array![
            [1.0, 0.0], // A
            [1.0, 0.0], // A
            [0.0, 1.0], // B
            [0.0, 1.0], // B
        ];
        let mut gene_to_idx = HashMap::new();
        gene_to_idx.insert("L".into(), 0);
        gene_to_idx.insert("R".into(), 1);
        let inter = CellChatInteraction::from_row("L", "R", "Test", "Secreted Signaling");
        let plan = LigandFieldPlan {
            mode: LigandFieldMode::Spatial,
            kh: 0.5,
            hill_coef: 1.0,
            replace_lr_pairs: true,
            min_cells: 1,
            received_ligand_norm: ReceivedLigandNorm::GlobalN,
            interactions: vec![inter],
            cell_group: vec![0, 0, 1, 1],
            group_names: vec!["A".into(), "B".into()],
            prob: Array3::<f64>::zeros((1, 2, 2)),
            pair_names: vec!["L$R".into()],
            received_ligand_cache: None,
        };
        let x = build_hybrid_lr_matrix(&plan, &xy, &expr, &gene_to_idx, 1.0, 1.0).unwrap();
        assert!(x[[2, 0]] > 0.0);
        assert!(x[[3, 0]] > 0.0);
        assert_eq!(x[[0, 0]], 0.0);
        assert_eq!(x[[1, 0]], 0.0);
    }

    #[test]
    fn hybrid_meanfield_uses_global_ligand_not_space() {
        // Distant senders (x=0) vs receivers (x=100): spatial ~0, meanfield uses global mean L.
        let xy = array![[0.0, 0.0], [0.0, 0.0], [100.0, 0.0], [100.0, 0.0]];
        let expr = array![
            [2.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ];
        let mut gene_to_idx = HashMap::new();
        gene_to_idx.insert("L".into(), 0);
        gene_to_idx.insert("R".into(), 1);
        let inter = CellChatInteraction::from_row("L", "R", "Test", "Secreted Signaling");
        let plan_mf = LigandFieldPlan {
            mode: LigandFieldMode::Meanfield,
            kh: 0.5,
            hill_coef: 1.0,
            replace_lr_pairs: true,
            min_cells: 1,
            received_ligand_norm: ReceivedLigandNorm::GlobalN,
            interactions: vec![inter.clone()],
            cell_group: vec![0, 0, 1, 1],
            group_names: vec!["A".into(), "B".into()],
            prob: Array3::<f64>::zeros((1, 2, 2)),
            pair_names: vec!["L$R".into()],
            received_ligand_cache: None,
        };
        let x_mf =
            build_hybrid_lr_matrix(&plan_mf, &xy, &expr, &gene_to_idx, 1.0, 1.0).unwrap();
        assert!((x_mf[[2, 0]] - 1.0).abs() < 1e-9);
        assert!((x_mf[[3, 0]] - 1.0).abs() < 1e-9);
        assert_eq!(x_mf[[0, 0]], 0.0);

        let plan_sp = LigandFieldPlan {
            mode: LigandFieldMode::Spatial,
            ..plan_mf
        };
        let x_sp =
            build_hybrid_lr_matrix(&plan_sp, &xy, &expr, &gene_to_idx, 1.0, 1.0).unwrap();
        assert!(x_sp[[2, 0]] < 1e-6);
        assert!(x_sp[[3, 0]] < 1e-6);
    }

    #[test]
    fn from_selected_expands_complex_and_copies_prob() {
        let inter = CellChatInteraction::from_row(
            "Tgfb1",
            "Tgfbr1_Tgfbr2",
            "TGFb",
            "Secreted Signaling",
        );
        let mut prob = Array3::<f64>::zeros((1, 2, 2));
        prob[[0, 0, 1]] = 0.7;
        let result = CellChatProbResult {
            group_names: vec!["A".into(), "B".into()],
            interactions: vec![inter],
            prob,
            pvalues: None,
            group_counts: vec![3, 3],
        };
        let cfg = LigandFieldConfig {
            mode: LigandFieldMode::Spatial,
            ..Default::default()
        };
        let plan = LigandFieldPlan::from_selected(result, &[0], &cfg);
        assert_eq!(plan.interactions.len(), 2);
        let names: HashSet<_> = plan.pair_names.iter().map(|s| s.as_str()).collect();
        assert!(names.contains("Tgfb1$Tgfbr1"));
        assert!(names.contains("Tgfb1$Tgfbr2"));
        assert!((plan.prob[[0, 0, 1]] - 0.7).abs() < 1e-12);
        assert!((plan.prob[[1, 0, 1]] - 0.7).abs() < 1e-12);
    }

    #[test]
    fn max_normalization_changes_hill_scale() {
        // Without max-norm, L=R=2 → LR=4 → Hill(Kh=0.5) ≈ 0.89
        // With max-norm (max=2), L=R=1 → LR=1 → Hill ≈ 0.67
        let gene_names = vec!["L".into(), "R".into()];
        let expr = array![[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]];
        let group_ids = vec![0, 0, 0];
        let group_names = vec!["A".into()];
        let inter = CellChatInteraction::from_row("L", "R", "Test", "Secreted Signaling");
        let cfg = LigandFieldConfig {
            min_cells: 1,
            n_perm: 0,
            ..Default::default()
        };
        let res = compute_commun_prob(
            &expr,
            &gene_names,
            &group_ids,
            &group_names,
            &[inter],
            &cfg,
        )
        .unwrap();
        let p = res.prob[[0, 0, 0]];
        let expected = hill_commun_prob(1.0, 1.0, 0.5, 1.0);
        assert!((p - expected).abs() < 1e-9, "p={p} expected={expected}");
    }

    #[test]
    fn expand_complex_to_independent_lig_rec_units() {
        let raw = CellChatInteraction::from_row(
            "Tgfb1",
            "Tgfbr1_Tgfbr2",
            "TGFb",
            "Secreted Signaling",
        );
        assert_eq!(raw.receptor_subunits.len(), 2);
        let units = expand_complexes_to_independent_units(&[raw]);
        assert_eq!(units.len(), 2);
        let names: HashSet<_> = units.iter().map(|u| u.pair_name.as_str()).collect();
        assert!(names.contains("Tgfb1$Tgfbr1"));
        assert!(names.contains("Tgfb1$Tgfbr2"));
        for u in &units {
            assert_eq!(u.ligand_subunits.len(), 1);
            assert_eq!(u.receptor_subunits.len(), 1);
        }
    }

    #[test]
    fn expand_dedupes_overlapping_complex_rows() {
        let a = CellChatInteraction::from_row("A_B", "R1_R2", "P", "Secreted Signaling");
        let b = CellChatInteraction::from_row("A", "R1", "P", "Secreted Signaling");
        let units = expand_complexes_to_independent_units(&[a, b]);
        // A×R1 appears once despite coming from both the complex cartesian product and the singleton.
        assert_eq!(
            units.iter().filter(|u| u.pair_name == "A$R1").count(),
            1
        );
        assert!(units.iter().any(|u| u.pair_name == "A$R2"));
        assert!(units.iter().any(|u| u.pair_name == "B$R1"));
        assert!(units.iter().any(|u| u.pair_name == "B$R2"));
    }

    #[test]
    fn load_mouse_db_keeps_complexes_until_expand() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/cellchat_mouse.csv");
        if !root.is_file() {
            return;
        }
        let db = load_cellchat_db(&root).unwrap();
        assert!(db.len() > 1000);
        assert!(
            db.iter()
                .any(|i| i.receptor_subunits.len() > 1 || i.ligand_subunits.len() > 1),
            "expected multi-subunit complexes in raw DB"
        );
        let units = expand_complexes_to_independent_units(&db);
        for inter in &units {
            assert_eq!(inter.ligand_subunits.len(), 1, "{}", inter.pair_name);
            assert_eq!(inter.receptor_subunits.len(), 1, "{}", inter.pair_name);
            assert!(!inter.pair_name.contains('_'));
        }
        assert!(units.iter().any(|i| i.pair_name == "Tgfb1$Tgfbr1"));
        assert!(units.iter().any(|i| i.pair_name == "Tgfb1$Tgfbr2"));
    }

    #[test]
    fn expressed_selection_ignores_min_prob_and_ranks_by_score() {
        let inter_a = CellChatInteraction::from_row("La", "Ra", "P", "Secreted Signaling");
        let inter_b = CellChatInteraction::from_row("Lb", "Rb", "P", "Secreted Signaling");
        let mut prob = Array3::<f64>::zeros((2, 1, 1));
        prob[[0, 0, 0]] = 0.01; // below a typical min_prob
        prob[[1, 0, 0]] = 0.9;
        let result = CellChatProbResult {
            group_names: vec!["A".into()],
            interactions: vec![inter_a, inter_b],
            prob,
            pvalues: None,
            group_counts: vec![10],
        };
        let cfg_prob = LigandFieldConfig {
            pair_selection: PairSelectionMode::Prob,
            min_prob: 0.05,
            max_interactions: Some(10),
            ..Default::default()
        };
        let keep_prob = select_interactions(&result, &cfg_prob);
        assert_eq!(keep_prob, vec![1]);

        let cfg_expr = LigandFieldConfig {
            pair_selection: PairSelectionMode::Expressed,
            min_prob: 0.05,
            max_interactions: Some(1),
            ..Default::default()
        };
        // Higher expression product for interaction 0 → selected despite low P.
        let scores = vec![10.0, 1.0];
        let keep_expr = select_interactions_with_expr_scores(&result, &cfg_expr, Some(&scores));
        assert_eq!(keep_expr, vec![0]);
    }

    #[test]
    fn kernel_mass_spatial_matches_meanfield_when_ligand_uniform() {
        // Uniform L → kernel-mass neighborhood mean equals global mean.
        let xy = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]];
        let expr = array![
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ];
        let mut gene_to_idx = HashMap::new();
        gene_to_idx.insert("L".into(), 0);
        gene_to_idx.insert("R".into(), 1);
        let inter = CellChatInteraction::from_row("L", "R", "Test", "Secreted Signaling");
        let plan_sp = LigandFieldPlan {
            mode: LigandFieldMode::Spatial,
            kh: 0.5,
            hill_coef: 1.0,
            replace_lr_pairs: true,
            min_cells: 1,
            received_ligand_norm: ReceivedLigandNorm::KernelMass,
            interactions: vec![inter.clone()],
            cell_group: vec![0, 0, 1, 1],
            group_names: vec!["A".into(), "B".into()],
            prob: Array3::<f64>::zeros((1, 2, 2)),
            pair_names: vec!["L$R".into()],
            received_ligand_cache: None,
        };
        let plan_mf = LigandFieldPlan {
            mode: LigandFieldMode::Meanfield,
            received_ligand_norm: ReceivedLigandNorm::KernelMass,
            ..plan_sp.clone()
        };
        let x_sp = build_hybrid_lr_matrix(&plan_sp, &xy, &expr, &gene_to_idx, 2.0, 1.0).unwrap();
        let x_mf = build_hybrid_lr_matrix(&plan_mf, &xy, &expr, &gene_to_idx, 2.0, 1.0).unwrap();
        for i in 0..4 {
            assert_abs_diff_eq!(x_sp[[i, 0]], x_mf[[i, 0]], epsilon = 1e-9);
        }
    }

    #[test]
    fn precompute_cache_reused_by_hybrid_build() {
        let xy = array![[0.0, 0.0], [1.0, 0.0]];
        let expr = array![[2.0, 1.0], [0.0, 1.0]];
        let mut gene_to_idx = HashMap::new();
        gene_to_idx.insert("L".into(), 0);
        gene_to_idx.insert("R".into(), 1);
        let inter = CellChatInteraction::from_row("L", "R", "Test", "Secreted Signaling");
        let mut plan = LigandFieldPlan {
            mode: LigandFieldMode::Meanfield,
            kh: 0.5,
            hill_coef: 1.0,
            replace_lr_pairs: true,
            min_cells: 1,
            received_ligand_norm: ReceivedLigandNorm::GlobalN,
            interactions: vec![inter],
            cell_group: vec![0, 1],
            group_names: vec!["A".into(), "B".into()],
            prob: Array3::<f64>::zeros((1, 2, 2)),
            pair_names: vec!["L$R".into()],
            received_ligand_cache: None,
        };
        precompute_received_ligand_cache(
            &mut plan,
            &xy,
            &expr,
            &gene_to_idx,
            1.0,
            1.0,
            None,
        )
        .unwrap();
        assert!(plan.received_ligand_cache.as_ref().unwrap().contains_key("L"));
        let x = build_hybrid_lr_matrix(&plan, &xy, &expr, &gene_to_idx, 1.0, 1.0).unwrap();
        // mean(L)=1 → X = 1 * R
        assert_abs_diff_eq!(x[[0, 0]], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[[1, 0]], 1.0, epsilon = 1e-12);
    }
}
