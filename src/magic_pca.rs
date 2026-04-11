//! Randomized PCA aligned with scikit-learn `PCA(svd_solver="randomized")` (1.7.x).
//!
//! The [`efficient_pca`](https://docs.rs/efficient_pca/latest/efficient_pca/) crate (v0.1.8)
//! builds on nightly-only `portable_simd`; on stable we use **faer** for QR / matmul and
//! **LAPACK ?gesdd** (via `ndarray-linalg`) for the thin SVD of `B`, matching sklearn’s default
//! `svd_lapack_driver="gesdd"`.

use std::sync::OnceLock;

use anyhow::{Result, bail};
use faer::Mat;
use faer::linalg::solvers::Qr;
use faer::Par;
use lapack_sys::dgesdd_;
use ndarray::{Array1, Array2, ArrayView2, Axis, s};
use rayon::prelude::*;
use std::os::raw::{c_char, c_int};

pub fn set_faer_rayon_threads(num_threads: usize) {
    faer::set_global_parallelism(Par::rayon(num_threads));
}

static FAER_PAR_INIT: OnceLock<()> = OnceLock::new();

fn ensure_faer_parallelism() {
    FAER_PAR_INIT.get_or_init(|| {
        let n = std::env::var("FAER_RAYON_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| rayon::current_num_threads().max(1));
        set_faer_rayon_threads(n);
    });
}

#[derive(Clone, Debug)]
pub struct SklearnRandomizedPcaConfig {
    pub n_components: usize,
    pub n_oversamples: usize,
    pub n_iter: Option<usize>,
    pub omega: Option<Array2<f64>>,
}

impl SklearnRandomizedPcaConfig {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            n_oversamples: 10,
            n_iter: None,
            omega: None,
        }
    }
}

fn column_means(x: ArrayView2<f64>) -> Array1<f64> {
    let n = x.nrows() as f64;
    x.sum_axis(Axis(0)) / n
}

fn center_in_place(x: &mut Array2<f64>, mean: &Array1<f64>) {
    x.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            row.zip_mut_with(mean, |a, &m| *a -= m);
        });
}

fn view_to_mat(a: ArrayView2<f64>) -> Mat<f64> {
    let (r, c) = a.dim();
    Mat::from_fn(r, c, |i, j| a[(i, j)])
}

fn mat_to_array2(m: faer::MatRef<'_, f64>) -> Array2<f64> {
    let (r, c) = m.shape();
    Array2::from_shape_fn((r, c), |(i, j)| m[(i, j)])
}

fn qr_thin(z: Mat<f64>) -> Mat<f64> {
    Qr::new(z.as_ref()).compute_thin_Q()
}

pub fn gesdd_thin_rowmajor(b: Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let (m, n) = b.dim();
    let k = m.min(n);
    let m_i = m as c_int;
    let n_i = n as c_int;
    let lda = m as c_int;
    let mut a_col = vec![0.0_f64; m * n];
    for row in 0..m {
        for col in 0..n {
            a_col[col * m + row] = b[(row, col)];
        }
    }
    let jobz = b'S' as c_char;
    let mut s = vec![0.0_f64; k];
    let mut u = vec![0.0_f64; m * k];
    let mut vt = vec![0.0_f64; k * n];
    let ldu = m as c_int;
    let ldvt = k as c_int;
    let mut iwork = vec![0 as c_int; 8 * k.max(1)];
    let mut info = 0 as c_int;
    let mut work = vec![0.0_f64; 1];
    let mut lwork = -1 as c_int;
    unsafe {
        dgesdd_(
            &jobz,
            &m_i,
            &n_i,
            a_col.as_mut_ptr(),
            &lda,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            &ldu,
            vt.as_mut_ptr(),
            &ldvt,
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    if info != 0 {
        bail!("dgesdd workspace query info={info}");
    }
       let lwork_opt = work[0] as usize;
    work.resize(lwork_opt.max(1), 0.0);
    lwork = work.len() as c_int;
    unsafe {
        dgesdd_(
            &jobz,
            &m_i,
            &n_i,
            a_col.as_mut_ptr(),
            &lda,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            &ldu,
            vt.as_mut_ptr(),
            &ldvt,
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    if info != 0 {
        bail!("dgesdd info={info}");
    }
    let mut u_rm = Array2::zeros((m, k));
    for row in 0..m {
        for j in 0..k {
            u_rm[(row, j)] = u[row + j * m];
        }
    }
    let mut vt_rm = Array2::zeros((k, n));
    for i in 0..k {
        for col in 0..n {
            vt_rm[(i, col)] = vt[i + col * k];
        }
    }
    Ok((u_rm, Array1::from_vec(s), vt_rm))
}

fn resolve_n_iter(
    n_iter: Option<usize>,
    n_components: usize,
    n_samples: usize,
    n_features: usize,
) -> usize {
    match n_iter {
        Some(k) => k,
        None => {
            if (n_components as f64) < 0.1 * (n_samples.min(n_features) as f64) {
                7
            } else {
                4
            }
        }
    }
}

pub fn randomized_svd_sklearn(
    x_centered: ArrayView2<f64>,
    n_components: usize,
    n_oversamples: usize,
    n_iter: Option<usize>,
    omega: Option<ArrayView2<f64>>,
) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    ensure_faer_parallelism();
    let n_samples = x_centered.nrows();
    let n_features = x_centered.ncols();
    if n_components == 0 || n_components > n_samples.min(n_features) {
        bail!("invalid n_components");
    }
    let n_iter = resolve_n_iter(n_iter, n_components, n_samples, n_features);
    let transpose = n_samples < n_features;
    let m = if transpose {
        view_to_mat(x_centered.t())
    } else {
        view_to_mat(x_centered)
    };
    let n_random = n_components + n_oversamples;
    let mut q: Mat<f64> = if let Some(om) = omega {
        if om.nrows() != m.ncols() || om.ncols() != n_random {
            bail!(
                "omega shape {:?} != ({}, {})",
                om.dim(),
                m.ncols(),
                n_random
            );
        }
        view_to_mat(om)
    } else {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        Mat::from_fn(m.ncols(), n_random, |_, _| StandardNormal.sample(&mut rng))
    };

    for _ in 0..n_iter {
        let z = &m * &q;
        q = qr_thin(z);
        let z2 = m.transpose() * &q;
        q = qr_thin(z2);
    }
    let zf = &m * &q;
    let qf = qr_thin(zf);
    let b = qf.transpose() * &m;
    let b_arr = mat_to_array2(b.as_ref());
    let b_ncols = b_arr.ncols();
    let (uhat, s_all, vt_h) = gesdd_thin_rowmajor(b_arr)?;
    let uhat_f = view_to_mat(uhat.view());
    let u = &qf * &uhat_f;
    let v_mat = Mat::from_fn(b_ncols, uhat.ncols(), |r, c| vt_h[(c, r)]);
    let v = v_mat.as_ref();

    let s = s_all.slice(s![..n_components]).to_owned();

    let (u_arr, vt_arr) = if transpose {
        let vt_part = v.submatrix(0, 0, v.nrows(), n_components);
        let vt_k = mat_to_array2(vt_part.transpose());
        let u_part = u.submatrix(0, 0, u.nrows(), n_components);
        let u_k = mat_to_array2(u_part);
        (vt_k.t().to_owned(), u_k.t().to_owned())
    } else {
        let u_part = u.submatrix(0, 0, u.nrows(), n_components);
        let vt_part = v.submatrix(0, 0, v.nrows(), n_components);
        (mat_to_array2(u_part), mat_to_array2(vt_part.transpose()))
    };

    Ok((u_arr, s, vt_arr))
}

fn svd_flip_v_basis(u: &mut Array2<f64>, vt: &mut Array2<f64>) {
    let k = vt.nrows();
    let mut signs = Vec::with_capacity(k);
    for r in 0..k {
        let row = vt.row(r);
        let j = row
            .indexed_iter()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .unwrap()
            .0;
        signs.push(row[j].signum());
    }
    for j in 0..k {
        let sgn = signs[j];
        u.column_mut(j).iter_mut().for_each(|v| *v *= sgn);
    }
    for r in 0..k {
        let sgn = signs[r];
        vt.row_mut(r).iter_mut().for_each(|x| *x *= sgn);
    }
}

pub fn fit_randomized_pca_sklearn(
    x: ArrayView2<f64>,
    cfg: &SklearnRandomizedPcaConfig,
) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>)> {
    let mean = column_means(x);
    let mut xc = x.to_owned();
    center_in_place(&mut xc, &mean);
    let om = cfg.omega.as_ref().map(|a| a.view());
    let (mut u, s, mut vt) = randomized_svd_sklearn(
        xc.view(),
        cfg.n_components,
        cfg.n_oversamples,
        cfg.n_iter,
        om,
    )?;
    svd_flip_v_basis(&mut u, &mut vt);
    let mut scores = u.clone();
    for j in 0..cfg.n_components {
        let sj = s[j];
        scores.column_mut(j).iter_mut().for_each(|v| *v *= sj);
    }
    Ok((scores, vt, mean))
}

pub fn transform_pca(x: ArrayView2<f64>, mean: &Array1<f64>, components: ArrayView2<f64>) -> Array2<f64> {
    let mut xc = x.to_owned();
    center_in_place(&mut xc, mean);
    xc.dot(&components.t())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray_npy::read_npy;
    use std::path::PathBuf;

    #[test]
    fn gesdd_matches_scipy_fixture() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pca_sklearn_qr");
        let b: Array2<f64> = read_npy(dir.join("B_pca.npy")).expect("B_pca.npy");
        let u_exp: Array2<f64> = read_npy(dir.join("gesdd_U.npy")).expect("gesdd_U.npy");
        let s_exp: Array1<f64> = read_npy(dir.join("gesdd_s.npy")).expect("gesdd_s.npy");
        let vh_exp: Array2<f64> = read_npy(dir.join("gesdd_Vh.npy")).expect("gesdd_Vh.npy");
        let (u, s, vh) = gesdd_thin_rowmajor(b).expect("gesdd");
        assert_eq!(u.dim(), u_exp.dim());
        assert_eq!(s.dim(), s_exp.dim());
        assert_eq!(vh.dim(), vh_exp.dim());
        for (a, b) in u.iter().zip(u_exp.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10, max_relative = 1e-10);
        }
        for (a, b) in s.iter().zip(s_exp.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10, max_relative = 1e-10);
        }
        for (a, b) in vh.iter().zip(vh_exp.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10, max_relative = 1e-10);
        }
    }

    #[test]
    fn scores_match_transform() {
        let x = Array2::from_shape_fn((30, 25), |(i, j)| (i as f64) * 0.1 + (j as f64) * 0.03);
        let cfg = SklearnRandomizedPcaConfig::new(5);
        let (scores, comp, mean) = fit_randomized_pca_sklearn(x.view(), &cfg).unwrap();
        let rec = transform_pca(x.view(), &mean, comp.view());
        assert_eq!(rec.dim(), scores.dim());
        for (a, b) in rec.iter().zip(scores.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }
}
