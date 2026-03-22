//! Sparse-group lasso regularised least-squares linear regression.
//!
//! This is a faithful port of the `GroupLasso` class in `_group_lasso.py`.
//!
//! # Mathematical formulation
//!
//! Minimise over **w**:
//!
//! ```text
//! (1/2n) · ‖Xw − y‖² + Σ_g reg_g · ‖w_g‖₂ + l1_reg · ‖w‖₁
//! ```
//!
//! where the groups are disjoint subsets of the feature indices determined by
//! the `groups` vector passed at construction.  Features with a negative or
//! [`GroupIndex::Unregularised`] group label are excluded from both penalties.
//!
//! # Quick start
//!
//! ```no_run
//! use space_trav_lr_rust::lasso::group_lasso::{GroupLasso, GroupLassoParams, ScaleReg};
//! use ndarray::Array2;
//!
//! let X: Array2<f64> = Array2::zeros((100, 10));
//! let y: Array2<f64> = Array2::zeros((100, 1));
//!
//! let mut model = GroupLasso::new(GroupLassoParams {
//!     groups: vec![0, 0, 1, 1, 2, 2, -1, -1, -1, -1],
//!     group_reg: 0.05,
//!     l1_reg: 0.05,
//!     ..Default::default()
//! });
//!
//! model.fit(&X, &y, None).unwrap();
//! let preds = model.predict(&X).unwrap();
//! ```

use std::collections::{HashMap, HashSet};

use ndarray::{s, Array1, Array2, Axis};
use rayon::prelude::*;

use crate::lasso::fista::{minimise as fista_minimise, FistaProblem, IterInfo};
use crate::lasso::prox::l1_l2_prox;
use crate::lasso::singular_values::find_largest_singular_value;
use crate::lasso::subsampling::SubsamplingScheme;

// ── Configuration ─────────────────────────────────────────────────────────────

/// Controls how per-group regularisation is scaled by group size.
#[derive(Debug, Clone, PartialEq)]
pub enum ScaleReg {
    /// Multiply by √(group size)  — default in the original paper.
    GroupSize,
    /// No scaling.
    None,
    /// Divide by √(group size).
    InverseGroupSize,
}

impl Default for ScaleReg {
    fn default() -> Self {
        ScaleReg::GroupSize
    }
}

/// All hyper-parameters for `GroupLasso`.
///
/// Mirrors the `__init__` arguments of the Python class.
#[derive(Debug, Clone)]
pub struct GroupLassoParams {
    /// Group label for every feature column.
    /// Negative values mean "not regularised".
    pub groups: Vec<i64>,

    /// Base regularisation coefficient for the group-ℓ₂ penalty.
    /// Can be overridden per-group by passing a `group_reg_vec` to [`GroupLasso::new_with_regs`].
    pub group_reg: f64,

    /// Regularisation coefficient for the element-wise ℓ₁ penalty.
    pub l1_reg: f64,

    /// Maximum number of FISTA iterations.
    pub n_iter: usize,

    /// Relative convergence tolerance.
    pub tol: f64,

    /// Group-size scaling mode.
    pub scale_reg: ScaleReg,

    /// Subsampling scheme for stochastic gradients.
    pub subsampling: SubsamplingScheme,

    /// Whether to fit an intercept term.
    pub fit_intercept: bool,

    /// Use the Frobenius norm (instead of power iteration) to estimate the
    /// Lipschitz constant.  Faster but may over-estimate badly.
    pub frobenius_lipschitz: bool,

    /// Random seed for the internal RNG.
    pub seed: u64,

    /// Reuse coefficients from a previous `fit` call.
    pub warm_start: bool,
}

impl Default for GroupLassoParams {
    fn default() -> Self {
        Self {
            groups: Vec::new(),
            group_reg: 0.05,
            l1_reg: 0.05,
            n_iter: 100,
            tol: 1e-5,
            scale_reg: ScaleReg::GroupSize,
            subsampling: SubsamplingScheme::None,
            fit_intercept: true,
            frobenius_lipschitz: false,
            seed: 0,
            warm_start: false,
        }
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum GroupLassoError {
    NotFitted,
    ShapeMismatch(String),
    InvalidParam(String),
    ConvergenceWarning,
}

impl std::fmt::Display for GroupLassoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFitted => write!(f, "Model has not been fitted yet"),
            Self::ShapeMismatch(s) => write!(f, "Shape mismatch: {}", s),
            Self::InvalidParam(s) => write!(f, "Invalid parameter: {}", s),
            Self::ConvergenceWarning => write!(
                f,
                "FISTA did not converge; try increasing n_iter or decreasing tol"
            ),
        }
    }
}

impl std::error::Error for GroupLassoError {}

// ── Fitted state ──────────────────────────────────────────────────────────────

/// Coefficients learned by `fit`.
#[derive(Debug, Clone)]
pub struct FittedCoefficients {
    /// Shape: `(num_features, num_targets)`
    pub coef: Array2<f64>,
    /// Shape: `(1, num_targets)` — zero when `fit_intercept` is false
    pub intercept: Array2<f64>,
}

// ── Main struct ───────────────────────────────────────────────────────────────

/// Sparse-group lasso regularised least-squares linear regression.
#[derive(Debug, Clone)]
pub struct GroupLasso {
    pub params: GroupLassoParams,

    // Per-group regularisation strengths (computed from `params.group_reg` +
    // optional per-group overrides + `scale_reg` scaling).
    group_reg_vector: Option<Vec<f64>>,

    // Boolean membership masks — one per unique non-negative group id.
    // `groups_[k][i] == true` means feature i belongs to group k.
    groups_masks: Option<Vec<Vec<bool>>>,

    // Unique non-negative group ids in sorted order.
    group_ids: Option<Vec<i64>>,

    // Raw group-id for each feature (same as params.groups but validated).
    feature_group_ids: Option<Vec<i64>>,

    /// Learned coefficients; `None` until `fit` is called.
    pub fitted: Option<FittedCoefficients>,

    // Column means used for centering (needed to correct the intercept).
    x_means: Option<Array2<f64>>,

    // Last Lipschitz estimate (may grow due to backtracking; reused in warm start).
    lipschitz: Option<f64>,
}

impl GroupLasso {
    /// Create a new model with uniform group regularisation.
    pub fn new(params: GroupLassoParams) -> Self {
        Self {
            params,
            group_reg_vector: None,
            groups_masks: None,
            group_ids: None,
            feature_group_ids: None,
            fitted: None,
            x_means: None,
            lipschitz: None,
        }
    }

    /// Create a new model with per-group regularisation coefficients.
    ///
    /// `group_reg_vec` must have one entry per unique non-negative group id,
    /// in the order they appear when sorted numerically.
    pub fn new_with_regs(params: GroupLassoParams, group_reg_vec: Vec<f64>) -> Self {
        let mut m = Self::new(params);
        m.group_reg_vector = Some(group_reg_vec);
        m
    }

    // ── Regularisation helpers ────────────────────────────────────────────────

    fn get_reg_strength(&self, group_size: usize, base_reg: f64) -> f64 {
        let sz = group_size as f64;
        match self.params.scale_reg {
            ScaleReg::GroupSize => base_reg * sz.sqrt(),
            ScaleReg::None => base_reg,
            ScaleReg::InverseGroupSize => base_reg / sz.sqrt(),
        }
    }

    fn build_reg_vector(&self, masks: &[Vec<bool>]) -> Vec<f64> {
        if let Some(v) = &self.group_reg_vector {
            return v.clone();
        }
        masks
            .iter()
            .map(|m| {
                let size = m.iter().filter(|&&b| b).count();
                self.get_reg_strength(size, self.params.group_reg)
            })
            .collect()
    }

    // ── Dataset preparation ───────────────────────────────────────────────────

    /// Validate inputs, optionally centre X, prepend an intercept column.
    ///
    /// Returns `(X_aug, X_means, y)` where `X_aug` has the intercept column
    /// prepended.
    fn prepare_dataset(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let (n, p) = (x.nrows(), x.ncols());

        // Centre X for numerical stability
        let x_means = if self.params.fit_intercept {
            x.mean_axis(Axis(0))
                .unwrap()
                .insert_axis(Axis(0)) // shape: (1, p)
        } else {
            Array2::zeros((1, p))
        };

        let x_centred: Array2<f64> = if self.params.fit_intercept {
            Array2::from_shape_fn((n, p), |(i, j)| x[[i, j]] - x_means[[0, j]])
        } else {
            x.clone()
        };

        // Prepend a column of ones for the intercept
        let mut x_aug = Array2::zeros((n, p + 1));
        x_aug.column_mut(0).fill(1.0);
        x_aug.slice_mut(s![.., 1..]).assign(&x_centred);

        (x_aug, x_means, y.clone())
    }

    // ── Group structure ───────────────────────────────────────────────────────

    fn build_groups(&mut self, num_features: usize) -> Result<(), GroupLassoError> {
        let raw = if self.params.groups.is_empty() {
            // Default: each feature gets its own group
            (0..num_features as i64).collect::<Vec<_>>()
        } else {
            if self.params.groups.len() != num_features {
                return Err(GroupLassoError::ShapeMismatch(format!(
                    "groups has length {} but X has {} features",
                    self.params.groups.len(),
                    num_features
                )));
            }
            self.params.groups.clone()
        };

        // Unique non-negative group ids (regularised groups only)
        let mut unique: Vec<i64> = raw.iter().filter(|&&g| g >= 0).cloned().collect();
        unique.sort_unstable();
        unique.dedup();

        // Boolean masks
        let masks: Vec<Vec<bool>> = unique
            .iter()
            .map(|&uid| raw.iter().map(|&g| g == uid).collect())
            .collect();

        self.feature_group_ids = Some(raw);
        self.group_ids = Some(unique);
        self.groups_masks = Some(masks);
        Ok(())
    }

    // ── Lipschitz estimation ──────────────────────────────────────────────────

    fn estimate_lipschitz(&self, x_aug: &Array2<f64>) -> f64 {
        let n = x_aug.nrows() as f64;
        if self.params.frobenius_lipschitz {
            let frob: f64 = x_aug.iter().map(|v| v * v).sum::<f64>().sqrt();
            return frob * frob / n;
        }
        let s_max = find_largest_singular_value(
            x_aug,
            self.params.seed,
            &self.params.subsampling,
            None,
            None,
        );
        1.5 * s_max * s_max / n
    }

    // ── Regularisation penalty ────────────────────────────────────────────────

    /// Compute the full regularisation penalty for a coefficient matrix `coef`
    /// (intercept row **excluded**).
    pub fn regulariser(&self, coef: &Array2<f64>) -> f64 {
        let masks = self.groups_masks.as_ref().unwrap();
        let regs = self.build_reg_vector(masks);

        let mut penalty = 0.0_f64;
        for (mask, &reg) in masks.iter().zip(regs.iter()) {
            // ℓ₂ norm over the group for each target, sum across targets
            let num_targets = coef.ncols();
            for col in 0..num_targets {
                let norm: f64 = mask
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &m)| if m { Some(coef[[i, col]].powi(2)) } else { None })
                    .sum::<f64>()
                    .sqrt();
                penalty += reg * norm;
            }
        }

        // ℓ₁ term
        let l1: f64 = coef.iter().map(|v| v.abs()).sum();
        penalty += self.params.l1_reg * l1;
        penalty
    }

    // ── MSE loss and gradient ─────────────────────────────────────────────────

    fn mse_loss(x_aug: &Array2<f64>, y: &Array2<f64>, w: &Array2<f64>) -> f64 {
        let resid = x_aug.dot(w) - y; // shape: (n, targets)
        let n = x_aug.nrows() as f64;
        0.5 * resid.iter().map(|v| v * v).sum::<f64>() / n
    }

    fn mse_grad(x_aug: &Array2<f64>, y: &Array2<f64>, w: &Array2<f64>) -> Array2<f64> {
        let n = x_aug.nrows() as f64;
        let resid = x_aug.dot(w) - y;
        x_aug.t().dot(&resid) / n
    }

    // ── Coefficient packing / unpacking ──────────────────────────────────────

    /// Pack intercept (row 0) and coefficients (rows 1..) into one matrix.
    fn join(intercept: &Array2<f64>, coef: &Array2<f64>) -> Array2<f64> {
        ndarray::concatenate(Axis(0), &[intercept.view(), coef.view()]).unwrap()
    }

    /// Unpack: returns `(intercept, coef)`.
    fn split(w: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        let intercept = w.slice(s![0..1, ..]).to_owned();
        let coef = w.slice(s![1.., ..]).to_owned();
        (intercept, coef)
    }

    // ── fit ───────────────────────────────────────────────────────────────────

    /// Fit the model to data.
    ///
    /// # Arguments
    /// * `x`         – feature matrix `(n, p)`
    /// * `y`         – target matrix `(n, t)` or vector `(n,)` (1-D will be reshaped)
    /// * `lipschitz` – optional precomputed Lipschitz bound; estimated if `None`
    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        lipschitz: Option<f64>,
    ) -> Result<bool, GroupLassoError> {
        let (n, p) = (x.nrows(), x.ncols());

        if n != y.nrows() {
            return Err(GroupLassoError::ShapeMismatch(
                "X and y have different numbers of rows".into(),
            ));
        }

        let y2d = y.clone();

        // Build group structure
        self.build_groups(p)?;

        let masks = self.groups_masks.as_ref().unwrap();
        let regs = self.build_reg_vector(masks);

        // Prepare augmented matrix
        let (x_aug, x_means, y_prep) = self.prepare_dataset(x, &y2d);

        // Lipschitz constant
        let l0 = lipschitz
            .or(self.lipschitz)
            .unwrap_or_else(|| self.estimate_lipschitz(&x_aug));

        let num_targets = y_prep.ncols();

        // Initialise weights
        let (init_intercept, init_coef) = if self.params.warm_start {
            if let Some(ref f) = self.fitted {
                (f.intercept.clone(), f.coef.clone())
            } else {
                (Array2::zeros((1, num_targets)), Array2::zeros((p, num_targets)))
            }
        } else {
            (Array2::zeros((1, num_targets)), Array2::zeros((p, num_targets)))
        };

        let w0 = Self::join(&init_intercept, &init_coef);

        // Capture data for the closure (we need owned copies)
        let x_aug_owned = x_aug.clone();
        let y_owned = y_prep.clone();
        let masks_owned = masks.clone();
        let regs_owned = regs.clone();
        let l1_reg = self.params.l1_reg;
        let fit_intercept = self.params.fit_intercept;

        // Build the FISTA-compatible problem struct
        let problem = GroupLassoProblem {
            x_aug: x_aug_owned,
            y: y_owned,
            masks: masks_owned,
            group_regs: regs_owned,
            l1_reg,
            fit_intercept,
        };

        let result = fista_minimise(
            &problem,
            w0,
            l0,
            self.params.n_iter,
            self.params.tol,
            None::<fn(&IterInfo)>,
        );

        self.lipschitz = Some(result.lipschitz);
        let (mut intercept, coef) = Self::split(&result.coef);

        // Correct the intercept for the centred X:
        // b_true = b_centred − x_means · coef
        let correction = x_means.dot(&coef); // shape: (1, targets)
        intercept = intercept - correction;

        self.fitted = Some(FittedCoefficients { coef, intercept });
        self.x_means = Some(x_means);

        if result.converged {
            Ok(true)
        } else {
            Err(GroupLassoError::ConvergenceWarning)
        }
    }

    // ── predict ───────────────────────────────────────────────────────────────

    /// Predict targets for new data.
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>, GroupLassoError> {
        let fitted = self.fitted.as_ref().ok_or(GroupLassoError::NotFitted)?;
        let w = Self::join(&fitted.intercept, &fitted.coef);

        let n = x.nrows();
        let p = fitted.coef.nrows();

        if x.ncols() != p {
            return Err(GroupLassoError::ShapeMismatch(format!(
                "X has {} features but model was fitted with {}",
                x.ncols(),
                p
            )));
        }

        // Prepend intercept column of ones
        let mut x_aug = Array2::zeros((n, p + 1));
        x_aug.column_mut(0).fill(1.0);
        x_aug.slice_mut(s![.., 1..]).assign(x);

        Ok(x_aug.dot(&w))
    }

    // ── Sparsity helpers ──────────────────────────────────────────────────────

    /// Boolean mask: `true` for features with non-negligible coefficients.
    pub fn sparsity_mask(&self) -> Result<Array1<bool>, GroupLassoError> {
        let fitted = self.fitted.as_ref().ok_or(GroupLassoError::NotFitted)?;
        let mean_abs: f64 = fitted.coef.iter().map(|v| v.abs()).sum::<f64>()
            / fitted.coef.len() as f64;
        let threshold = 1e-10 * mean_abs;
        let coef_mean_across_targets: Array1<f64> = fitted.coef.mean_axis(Axis(1)).unwrap();
        Ok(coef_mean_across_targets.mapv(|v| v.abs() > threshold))
    }

    /// Set of unique non-negative group ids that have at least one non-zero feature.
    pub fn chosen_groups(&self) -> Result<std::collections::HashSet<i64>, GroupLassoError> {
        let mask = self.sparsity_mask()?;
        let feature_ids = self
            .feature_group_ids
            .as_ref()
            .ok_or(GroupLassoError::NotFitted)?;
        let chosen = mask
            .iter()
            .zip(feature_ids.iter())
            .filter_map(|(&m, &g)| if m && g >= 0 { Some(g) } else { None })
            .collect();
        Ok(chosen)
    }

    /// Remove columns with zero coefficients from `x`.
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>, GroupLassoError> {
        let mask = self.sparsity_mask()?;
        let cols: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();
        let result = x.select(Axis(1), &cols);
        Ok(result)
    }
}

// ── Internal problem struct passed to FISTA ───────────────────────────────────

/// Wraps all data needed to compute the MSE loss and gradient.
///
/// This is kept separate from `GroupLasso` so it can hold owned data without
/// borrowing conflicts inside the closure.
struct GroupLassoProblem {
    x_aug: Array2<f64>,
    y: Array2<f64>,
    masks: Vec<Vec<bool>>,
    group_regs: Vec<f64>,
    l1_reg: f64,
    fit_intercept: bool,
}

impl FistaProblem for GroupLassoProblem {
    fn smooth_loss(&self, w: &Array2<f64>) -> f64 {
        GroupLasso::mse_loss(&self.x_aug, &self.y, w)
    }

    fn smooth_grad(&self, w: &Array2<f64>) -> Array2<f64> {
        let mut g = GroupLasso::mse_grad(&self.x_aug, &self.y, w);
        // Intercept is not regularised — but still receives gradient
        // unless fit_intercept is false (zero it in that case)
        if !self.fit_intercept {
            g.row_mut(0).fill(0.0);
        }
        g
    }

    fn prox(&self, w: &Array2<f64>, lipschitz: f64) -> Array2<f64> {
        // Row 0 is the intercept — never regularised
        let (intercept, coef) = GroupLasso::split(w);
        let scaled_l1 = self.l1_reg / lipschitz;
        let scaled_regs: Vec<f64> = self.group_regs.iter().map(|r| r / lipschitz).collect();
        let new_coef = l1_l2_prox(&coef, scaled_l1, &scaled_regs, &self.masks);
        GroupLasso::join(&intercept, &new_coef)
    }
}

// ── Clustered Group Lasso ─────────────────────────────────────────────────────

/// Fits an independent `GroupLasso` model for each cluster in the data.
#[derive(Debug, Clone)]
pub struct ClusteredGroupLasso {
    pub params: GroupLassoParams,
    /// Fitted models per cluster ID.
    pub models: HashMap<i64, GroupLasso>,
}

impl ClusteredGroupLasso {
    pub fn new(params: GroupLassoParams) -> Self {
        Self {
            params,
            models: HashMap::new(),
        }
    }

    /// Fit the model per cluster.
    ///
    /// * `x`        – feature matrix `(n, p)`
    /// * `y`        – target matrix `(n, t)`
    /// * `clusters` – cluster ID for each row `(n,)`
    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        clusters: &Array1<i64>,
    ) -> Result<bool, GroupLassoError> {
        if x.nrows() != y.nrows() || x.nrows() != clusters.len() {
            return Err(GroupLassoError::ShapeMismatch(
                "X, y, and clusters must have the same number of rows".into(),
            ));
        }

        let unique_ids: Vec<i64> = clusters.iter().cloned().collect::<HashSet<_>>().into_iter().collect();

        // Prepare data for each cluster
        let mut cluster_data = Vec::new();
        for &id in &unique_ids {
            let indices: Vec<usize> = clusters
                .iter()
                .enumerate()
                .filter(|(_, c)| **c == id)
                .map(|(i, _)| i)
                .collect();
            
            let x_cluster = x.select(Axis(0), &indices);
            let y_cluster = y.select(Axis(0), &indices);
            cluster_data.push((id, x_cluster, y_cluster));
        }

        // Fit models in parallel
        let results: Vec<(i64, GroupLasso, Result<bool, GroupLassoError>)> = cluster_data
            .into_par_iter()
            .map(|(id, x_c, y_c)| {
                let mut model = GroupLasso::new(self.params.clone());
                let res = model.fit(&x_c, &y_c, None);
                (id, model, res)
            })
            .collect();

        let mut all_converged = true;
        for (id, model, result) in results {
            match result {
                Ok(_) => {
                    self.models.insert(id, model);
                }
                Err(e) => {
                    if let GroupLassoError::ConvergenceWarning = e {
                        all_converged = false;
                        self.models.insert(id, model);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        if all_converged {
            Ok(true)
        } else {
            Err(GroupLassoError::ConvergenceWarning)
        }
    }

    /// Predict using the model corresponding to each row's cluster.
    pub fn predict(
        &self,
        x: &Array2<f64>,
        clusters: &Array1<i64>,
    ) -> Result<Array2<f64>, GroupLassoError> {
        if x.nrows() != clusters.len() {
            return Err(GroupLassoError::ShapeMismatch(
                "X and clusters must have the same number of rows".into(),
            ));
        }

        let unique_ids: Vec<i64> = clusters.iter().cloned().collect::<HashSet<_>>().into_iter().collect();
        let num_targets = self.models.values().next()
            .and_then(|m| m.fitted.as_ref())
            .map(|f| f.coef.ncols())
            .unwrap_or(1);

        // Group indices by cluster
        let mut cluster_indices = Vec::new();
        for &id in &unique_ids {
            let indices: Vec<usize> = clusters
                .iter()
                .enumerate()
                .filter(|(_, c)| **c == id)
                .map(|(i, _)| i)
                .collect();
            cluster_indices.push((id, indices));
        }

        // Predict in parallel per cluster
        let results: Vec<Result<(Vec<usize>, Array2<f64>), GroupLassoError>> = cluster_indices
            .into_par_iter()
            .map(|(id, indices)| {
                let model = self.models.get(&id).ok_or(GroupLassoError::NotFitted)?;
                let x_cluster = x.select(Axis(0), &indices);
                let p = model.predict(&x_cluster)?;
                Ok((indices, p))
            })
            .collect();

        let mut preds = Array2::zeros((x.nrows(), num_targets));
        for res in results {
            let (indices, p) = res?;
            for (local_idx, &global_idx) in indices.iter().enumerate() {
                preds.row_mut(global_idx).assign(&p.row(local_idx));
            }
        }

        Ok(preds)
    }

    /// Boolean mask for each cluster: `true` for features with non-negligible coefficients.
    pub fn coefficients(&self) -> HashMap<i64, (Array2<f64>, Array2<f64>)> {
        let mut result = HashMap::new();
        for (&id, model) in &self.models {
            if let Some(fitted) = &model.fitted {
                result.insert(id, (fitted.coef.clone(), fitted.intercept.clone()));
            }
        }
        result
    }

    pub fn sparsity_mask(&self) -> Result<HashMap<i64, Array1<bool>>, GroupLassoError> {
        let mut masks = HashMap::new();
        for (&id, model) in &self.models {
            masks.insert(id, model.sparsity_mask()?);
        }
        Ok(masks)
    }

    /// Set of unique non-negative group ids that have at least one non-zero feature, per cluster.
    pub fn chosen_groups(&self) -> Result<HashMap<i64, HashSet<i64>>, GroupLassoError> {
        let mut groups = HashMap::new();
        for (&id, model) in &self.models {
            groups.insert(id, model.chosen_groups()?);
        }
        Ok(groups)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn simple_xy() -> (Array2<f64>, Array2<f64>) {
        // y = 2·x₀ + 3·x₁ (no noise, two features in one group)
        let n = 50_usize;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 { i as f64 / n as f64 } else { (n - i) as f64 / n as f64 }
        });
        let y = Array2::from_shape_fn((n, 1), |(i, _)| 2.0 * x[[i, 0]] + 3.0 * x[[i, 1]]);
        (x, y)
    }

    #[test]
    fn fit_predict_shape() {
        let (x, y) = simple_xy();
        let mut model = GroupLasso::new(GroupLassoParams {
            groups: vec![0, 0],
            group_reg: 0.001,
            l1_reg: 0.001,
            n_iter: 200,
            ..Default::default()
        });
        let _ = model.fit(&x, &y, None);
        let pred = model.predict(&x).unwrap();
        assert_eq!(pred.shape(), &[50, 1]);
    }

    #[test]
    fn unregularised_group_recovers_coefficients() {
        // With negligible regularisation, MSE solution should match OLS closely.
        let (x, y) = simple_xy();
        let mut model = GroupLasso::new(GroupLassoParams {
            groups: vec![0, 1],
            group_reg: 1e-6,
            l1_reg: 1e-6,
            n_iter: 500,
            tol: 1e-8,
            fit_intercept: false,
            ..Default::default()
        });
        let _ = model.fit(&x, &y, None);
        let coef = &model.fitted.as_ref().unwrap().coef;
        // True coefficients are [2, 3]
        assert_abs_diff_eq!(coef[[0, 0]], 2.0, epsilon = 0.2);
        assert_abs_diff_eq!(coef[[1, 0]], 3.0, epsilon = 0.2);
    }

    #[test]
    fn high_group_reg_drives_coef_to_zero() {
        let (x, y) = simple_xy();
        let mut model = GroupLasso::new(GroupLassoParams {
            groups: vec![0, 0], // both features in the same group
            group_reg: 100.0,   // extreme regularisation
            l1_reg: 0.0,
            n_iter: 300,
            ..Default::default()
        });
        let _ = model.fit(&x, &y, None);
        let coef = &model.fitted.as_ref().unwrap().coef;
        for &v in coef.iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn clustered_fit_predict() {
        let (x, y) = simple_xy();
        // Create 2 clusters: first 25 rows and last 25 rows
        let n = x.nrows();
        let mut clusters = Array1::zeros(n);
        for i in 25..n {
            clusters[i] = 1;
        }

        let mut model = ClusteredGroupLasso::new(GroupLassoParams {
            groups: vec![0, 0],
            group_reg: 0.001,
            l1_reg: 0.001,
            n_iter: 200,
            ..Default::default()
        });

        model.fit(&x, &y, &clusters).unwrap();
        assert_eq!(model.models.len(), 2);

        let pred = model.predict(&x, &clusters).unwrap();
        assert_eq!(pred.shape(), &[n, 1]);

        // Verify that predictions are reasonably close to y
        for i in 0..n {
            assert_abs_diff_eq!(pred[[i, 0]], y[[i, 0]], epsilon = 0.5);
        }
    }
}
