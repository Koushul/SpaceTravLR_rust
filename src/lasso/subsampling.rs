//! Row-subsampling utilities for stochastic gradient computations.
//!
//! This is a faithful port of `_subsampling.py`.  The three subsampling
//! modes are encoded in the [`SubsamplingScheme`] enum:
//!
//! * `None`  → use every row
//! * `Fraction(f)` where 0 < f < 1 → keep that fraction of rows
//! * `Count(n)` → keep exactly n rows (must be ≤ num_rows)
//! * `Sqrt`  → keep √(num_rows) rows

use ndarray::{Array2, ArrayView2};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

// ── Public types ─────────────────────────────────────────────────────────────

/// How to subsample rows from a matrix.
#[derive(Debug, Clone, PartialEq)]
pub enum SubsamplingScheme {
    /// Use all rows (no subsampling).
    None,
    /// Keep this fraction of rows (0 < f < 1).
    Fraction(f64),
    /// Keep exactly this many rows (must be ≤ num_rows).
    Count(usize),
    /// Keep √(num_rows) rows.
    Sqrt,
}

impl SubsamplingScheme {
    /// Number of rows that will be kept for a matrix with `num_rows` rows.
    pub fn num_sampled_rows(&self, num_rows: usize) -> usize {
        match self {
            SubsamplingScheme::None => num_rows,
            SubsamplingScheme::Fraction(f) => (num_rows as f64 * f) as usize,
            SubsamplingScheme::Count(n) => {
                assert!(*n <= num_rows, "Cannot subsample more rows than present");
                *n
            }
            SubsamplingScheme::Sqrt => (num_rows as f64).sqrt() as usize,
        }
    }

    /// Fraction of rows that will be kept (used for Lipschitz rescaling).
    pub fn fraction(&self, num_rows: usize) -> f64 {
        self.num_sampled_rows(num_rows) as f64 / num_rows as f64
    }
}

// ── Row-index selection ───────────────────────────────────────────────────────

/// Return sorted row indices for the given scheme.
///
/// When `scheme` is [`SubsamplingScheme::None`] this returns `0..num_rows`.
pub fn get_row_indices(
    num_rows: usize,
    scheme: &SubsamplingScheme,
    rng: &mut ChaCha8Rng,
) -> Vec<usize> {
    match scheme {
        SubsamplingScheme::None => (0..num_rows).collect(),
        _ => {
            let n = scheme.num_sampled_rows(num_rows);
            let mut indices: Vec<usize> = (0..num_rows).collect();
            // Partial Fisher-Yates: first `n` elements after the shuffle are
            // our unbiased sample.
            for i in 0..n {
                let j = i + rng.gen_range(0..(num_rows - i));
                indices.swap(i, j);
            }
            let mut chosen = indices[..n].to_vec();
            chosen.sort_unstable();
            chosen
        }
    }
}

/// Select rows from `x` according to `indices`.
pub fn select_rows(x: &ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    let ncols = x.ncols();
    let mut out = Array2::zeros((indices.len(), ncols));
    for (out_row, &src_row) in indices.iter().enumerate() {
        out.row_mut(out_row).assign(&x.row(src_row));
    }
    out
}

// ── Stateful sampler (mirrors the Python `Subsampler` class) ─────────────────

/// Stateful row-sampler that caches the current index set and re-draws on
/// [`update_indices`][Subsampler::update_indices].
pub struct Subsampler {
    pub scheme: SubsamplingScheme,
    num_rows: usize,
    pub current_indices: Vec<usize>,
    rng: ChaCha8Rng,
}

impl Subsampler {
    pub fn new(num_rows: usize, scheme: SubsamplingScheme, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let current_indices = get_row_indices(num_rows, &scheme, &mut rng);
        Self { scheme, num_rows, current_indices, rng }
    }

    /// Draw a fresh set of indices (call after each gradient step).
    pub fn update_indices(&mut self) {
        self.current_indices =
            get_row_indices(self.num_rows, &self.scheme, &mut self.rng);
    }

    /// Return the subsampled view of `x` using the *current* indices.
    pub fn subsample<'a>(&self, x: &'a Array2<f64>) -> std::borrow::Cow<'a, Array2<f64>> {
        match self.scheme {
            SubsamplingScheme::None => std::borrow::Cow::Borrowed(x),
            _ => std::borrow::Cow::Owned(select_rows(&x.view(), &self.current_indices)),
        }
    }

    /// Fraction of rows currently selected.
    pub fn fraction(&self) -> f64 {
        self.scheme.fraction(self.num_rows)
    }
}

// ── Use rand::Rng trait ───────────────────────────────────────────────────────
use rand::Rng;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn fraction_scheme_selects_correct_count() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let scheme = SubsamplingScheme::Fraction(0.5);
        let indices = get_row_indices(100, &scheme, &mut rng);
        assert_eq!(indices.len(), 50);
        assert!(indices.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn sqrt_scheme_selects_sqrt_count() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let indices = get_row_indices(100, &SubsamplingScheme::Sqrt, &mut rng);
        assert_eq!(indices.len(), 10);
    }

    #[test]
    fn none_scheme_returns_all_rows() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let indices = get_row_indices(50, &SubsamplingScheme::None, &mut rng);
        assert_eq!(indices, (0..50).collect::<Vec<_>>());
    }

    #[test]
    fn count_scheme_exact() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let scheme = SubsamplingScheme::Count(7);
        let indices = get_row_indices(20, &scheme, &mut rng);
        assert_eq!(indices.len(), 7);
        assert!(indices.windows(2).all(|w| w[0] < w[1]));
        assert!(*indices.last().unwrap() < 20);
    }

    #[test]
    fn indices_are_valid_range() {
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let indices = get_row_indices(10, &SubsamplingScheme::Fraction(0.3), &mut rng);
        for &i in &indices {
            assert!(i < 10);
        }
    }

    #[test]
    fn indices_no_duplicates() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let indices = get_row_indices(50, &SubsamplingScheme::Fraction(0.8), &mut rng);
        let mut unique = indices.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), indices.len());
    }

    #[test]
    fn select_rows_correct() {
        let x = Array2::from_shape_fn((5, 3), |(i, j)| (i * 10 + j) as f64);
        let selected = select_rows(&x.view(), &[1, 3]);
        assert_eq!(selected.nrows(), 2);
        assert_eq!(selected.ncols(), 3);
        assert_eq!(selected[[0, 0]], 10.0);
        assert_eq!(selected[[1, 0]], 30.0);
    }

    #[test]
    fn num_sampled_rows_consistency() {
        assert_eq!(SubsamplingScheme::None.num_sampled_rows(100), 100);
        assert_eq!(SubsamplingScheme::Fraction(0.25).num_sampled_rows(100), 25);
        assert_eq!(SubsamplingScheme::Count(42).num_sampled_rows(100), 42);
        assert_eq!(SubsamplingScheme::Sqrt.num_sampled_rows(100), 10);
    }

    #[test]
    fn fraction_method_consistency() {
        let f = SubsamplingScheme::Fraction(0.5).fraction(100);
        assert!((f - 0.5).abs() < 1e-10);

        let f_none = SubsamplingScheme::None.fraction(100);
        assert!((f_none - 1.0).abs() < 1e-10);
    }

    #[test]
    fn subsampler_updates_indices() {
        let mut s = Subsampler::new(100, SubsamplingScheme::Fraction(0.1), 42);
        assert_eq!(s.current_indices.len(), 10);
        let first = s.current_indices.clone();
        s.update_indices();
        let second = s.current_indices.clone();
        assert_eq!(second.len(), 10);
        assert_ne!(first, second, "Different draws should differ (with high probability)");
    }

    #[test]
    fn subsampler_none_borrows() {
        let s = Subsampler::new(5, SubsamplingScheme::None, 0);
        let x = Array2::from_shape_fn((5, 2), |(i, j)| (i + j) as f64);
        let cow = s.subsample(&x);
        assert!(matches!(cow, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn subsampler_fraction_owns() {
        let s = Subsampler::new(10, SubsamplingScheme::Fraction(0.5), 0);
        let x = Array2::from_shape_fn((10, 2), |(i, j)| (i + j) as f64);
        let cow = s.subsample(&x);
        assert!(matches!(cow, std::borrow::Cow::Owned(_)));
        assert_eq!(cow.nrows(), 5);
    }
}
