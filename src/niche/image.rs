//! Build a per-cell **(n_targets × n_modulators)** image from a
//! [`compute_splash_all`](crate::perturb::compute_splash_all) result.
//!
//! Splash returns one `(n_cells × n_modulators_g)` matrix **per** target gene
//! `g`, with each target seeing only its own modulator alphabet (the ones it
//! was trained on). Different targets have different (and overlapping)
//! modulator sets. The CNN consumes a fixed-shape image per cell, so we:
//!
//! 1. Take the alphabetically-sorted **union** of all target gene names.
//! 2. Take the alphabetically-sorted **union** of all modulator gene names
//!    (after stripping the `beta_` prefix used by [`crate::betadata::BetaFrame`]).
//! 3. Allocate `images: Vec<f32>` of length `n_cells * n_targets * n_modulators`.
//!    For each `(target row, modulator column)` we copy the splash value if the
//!    target's per-target modulator alphabet contains that modulator,
//!    otherwise leave 0.
//!
//! Optional row/column standardization is exposed via [`StandardizeMode`] so
//! the CNN sees comparable scales across targets and modulators.

use std::collections::HashMap;

use ndarray::Array1;

use crate::betadata::GeneMatrix;

/// What axis to standardize the splash image along. Standardization is done
/// per axis across the dataset (i.e. across cells), then divided by `std + eps`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StandardizeMode {
    /// Leave splash values as-is.
    None,
    /// Subtract per-`(target, modulator)` mean and divide by per-`(target,
    /// modulator)` std across cells. Most useful when target gene scales differ
    /// dramatically.
    PerEntry,
    /// Subtract per-modulator mean and divide by per-modulator std across all
    /// cells × targets. Keeps relative target order intact.
    PerModulator,
}

impl Default for StandardizeMode {
    fn default() -> Self {
        StandardizeMode::PerEntry
    }
}

/// Output of [`build_niche_image_stack`].
pub struct NicheImageStack {
    /// Flat row-major buffer of shape `(n_cells, n_targets, n_modulators)`.
    pub images: Vec<f32>,
    /// Number of cells (image batch size).
    pub n_cells: usize,
    /// Image height = number of unified target genes.
    pub n_targets: usize,
    /// Image width = number of unified modulator genes.
    pub n_modulators: usize,
    /// Sorted target gene names — row labels of the image.
    pub target_names: Vec<String>,
    /// Sorted modulator gene names — column labels of the image.
    pub modulator_names: Vec<String>,
}

impl NicheImageStack {
    /// Borrow a view over a single cell's image as a row-major
    /// `(n_targets, n_modulators)` slice.
    pub fn cell(&self, cell: usize) -> &[f32] {
        let stride = self.n_targets * self.n_modulators;
        let off = cell * stride;
        &self.images[off..off + stride]
    }

    /// Per-cell L1 norm of the image; used as a sanity check.
    pub fn cell_l1_norms(&self) -> Vec<f32> {
        (0..self.n_cells)
            .map(|c| self.cell(c).iter().map(|v| v.abs()).sum())
            .collect()
    }
}

fn strip_beta(name: &str) -> &str {
    name.strip_prefix("beta_").unwrap_or(name)
}

/// Build the per-cell `(n_targets, n_modulators)` splash image stack from the
/// raw output of [`crate::perturb::compute_splash_all`].
///
/// `splash_per_target` maps **target gene name → (n_cells × n_target_modulators) GeneMatrix**;
/// the column names of each `GeneMatrix` are the per-target modulator alphabet
/// (often prefixed with `beta_`).
pub fn build_niche_image_stack(
    splash_per_target: &HashMap<String, GeneMatrix>,
    n_cells: usize,
    standardize: StandardizeMode,
) -> NicheImageStack {
    assert!(!splash_per_target.is_empty(), "splash_per_target is empty");

    let mut target_names: Vec<String> = splash_per_target.keys().cloned().collect();
    target_names.sort();

    let mut modulator_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for gm in splash_per_target.values() {
        assert_eq!(
            gm.n_rows(),
            n_cells,
            "splash matrix has {} rows but expected {}",
            gm.n_rows(),
            n_cells
        );
        for col in &gm.col_names {
            modulator_set.insert(strip_beta(col).to_string());
        }
    }
    let modulator_names: Vec<String> = modulator_set.into_iter().collect();
    let n_targets = target_names.len();
    let n_modulators = modulator_names.len();
    assert!(
        n_modulators > 0,
        "no modulator genes across splash matrices"
    );

    let modulator_idx: HashMap<&str, usize> = modulator_names
        .iter()
        .enumerate()
        .map(|(i, m)| (m.as_str(), i))
        .collect();

    let stride = n_targets * n_modulators;
    let mut images = vec![0.0f32; n_cells * stride];

    for (ti, tname) in target_names.iter().enumerate() {
        let gm = splash_per_target.get(tname).expect("target present");
        let mut col_to_unified: Vec<i32> = Vec::with_capacity(gm.n_cols());
        for col in &gm.col_names {
            let key = strip_beta(col);
            col_to_unified
                .push(*modulator_idx.get(key).expect("modulator in unified set") as i32);
        }
        let data = &gm.data;
        for c in 0..n_cells {
            let row = data.row(c);
            let cell_off = c * stride + ti * n_modulators;
            for (k, v) in row.iter().enumerate() {
                let mi = col_to_unified[k] as usize;
                images[cell_off + mi] = *v;
            }
        }
    }

    standardize_images(&mut images, n_cells, n_targets, n_modulators, standardize);

    NicheImageStack {
        images,
        n_cells,
        n_targets,
        n_modulators,
        target_names,
        modulator_names,
    }
}

fn standardize_images(
    images: &mut [f32],
    n_cells: usize,
    n_targets: usize,
    n_modulators: usize,
    mode: StandardizeMode,
) {
    match mode {
        StandardizeMode::None => {}
        StandardizeMode::PerEntry => {
            let stride = n_targets * n_modulators;
            let mut mean = Array1::<f64>::zeros(stride);
            let mut sq = Array1::<f64>::zeros(stride);
            for c in 0..n_cells {
                let off = c * stride;
                for k in 0..stride {
                    let v = images[off + k] as f64;
                    mean[k] += v;
                    sq[k] += v * v;
                }
            }
            let n = n_cells as f64;
            for k in 0..stride {
                mean[k] /= n;
                let var = (sq[k] / n - mean[k] * mean[k]).max(0.0);
                sq[k] = var.sqrt() + 1e-6;
            }
            for c in 0..n_cells {
                let off = c * stride;
                for k in 0..stride {
                    let z = ((images[off + k] as f64) - mean[k]) / sq[k];
                    images[off + k] = z as f32;
                }
            }
        }
        StandardizeMode::PerModulator => {
            let mut mean = vec![0.0f64; n_modulators];
            let mut sq = vec![0.0f64; n_modulators];
            let denom = (n_cells * n_targets) as f64;
            for c in 0..n_cells {
                for t in 0..n_targets {
                    let row = c * n_targets * n_modulators + t * n_modulators;
                    for m in 0..n_modulators {
                        let v = images[row + m] as f64;
                        mean[m] += v;
                        sq[m] += v * v;
                    }
                }
            }
            for m in 0..n_modulators {
                mean[m] /= denom;
                let var = (sq[m] / denom - mean[m] * mean[m]).max(0.0);
                sq[m] = var.sqrt() + 1e-6;
            }
            for c in 0..n_cells {
                for t in 0..n_targets {
                    let row = c * n_targets * n_modulators + t * n_modulators;
                    for m in 0..n_modulators {
                        let z = ((images[row + m] as f64) - mean[m]) / sq[m];
                        images[row + m] = z as f32;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::betadata::GeneMatrix;
    use ndarray::Array2;

    fn mk(name: &str, cols: &[&str], data: Array2<f32>) -> (String, GeneMatrix) {
        (
            name.to_string(),
            GeneMatrix::new(data, cols.iter().map(|s| s.to_string()).collect()),
        )
    }

    #[test]
    fn build_image_stack_aligns_columns() {
        let mut splash = HashMap::new();
        splash.insert(
            mk("D", &["beta_A", "beta_B"], Array2::from_shape_vec((2, 2), vec![1., 2., 3., 4.]).unwrap()).0
                .clone(),
            mk("D", &["beta_A", "beta_B"], Array2::from_shape_vec((2, 2), vec![1., 2., 3., 4.]).unwrap()).1,
        );
        splash.insert(
            "E".to_string(),
            GeneMatrix::new(
                Array2::from_shape_vec((2, 2), vec![10., 20., 30., 40.]).unwrap(),
                vec!["beta_B".to_string(), "beta_C".to_string()],
            ),
        );
        let stack = build_niche_image_stack(&splash, 2, StandardizeMode::None);
        assert_eq!(stack.n_targets, 2);
        assert_eq!(stack.n_modulators, 3);
        assert_eq!(stack.target_names, vec!["D", "E"]);
        assert_eq!(stack.modulator_names, vec!["A", "B", "C"]);

        let cell0 = stack.cell(0);
        let cell1 = stack.cell(1);
        // D row: [A=1, B=2, C=0]; E row: [A=0, B=10, C=20]
        assert_eq!(cell0, &[1., 2., 0., 0., 10., 20.]);
        assert_eq!(cell1, &[3., 4., 0., 0., 30., 40.]);
    }

    #[test]
    fn standardize_per_entry_zero_mean_unit_var() {
        let mut splash = HashMap::new();
        splash.insert(
            "D".to_string(),
            GeneMatrix::new(
                Array2::from_shape_vec((4, 1), vec![-2., -1., 1., 2.]).unwrap(),
                vec!["beta_A".to_string()],
            ),
        );
        let stack = build_niche_image_stack(&splash, 4, StandardizeMode::PerEntry);
        let mean: f64 = stack.images.iter().map(|&v| v as f64).sum::<f64>() / stack.images.len() as f64;
        assert!(mean.abs() < 1e-6, "mean = {}", mean);
        let var: f64 = stack.images.iter().map(|&v| (v as f64).powi(2)).sum::<f64>()
            / stack.images.len() as f64;
        assert!((var - 1.0).abs() < 1e-3, "var = {}", var);
    }
}
