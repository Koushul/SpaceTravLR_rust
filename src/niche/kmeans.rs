//! Tiny pure-Rust k-means (Lloyd) over `f32` embeddings, with k-means++
//! initialization. Only used to turn niche embeddings into integer labels.

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand::{Rng, rngs::StdRng};

#[derive(Debug)]
pub struct KMeansResult {
    pub labels: Vec<usize>,
    pub centers: Vec<f32>,
    pub n_clusters: usize,
    pub dim: usize,
    pub inertia: f64,
    pub n_iter: usize,
}

/// Run k-means++ + Lloyd's on a row-major `(n × dim)` embedding.
///
/// `n_clusters` must be ≥ 1 and ≤ n. Stops at `max_iter` or when fewer than
/// `tol * n` labels change between iterations.
pub fn kmeans_lloyd(
    points: &[f32],
    n: usize,
    dim: usize,
    n_clusters: usize,
    max_iter: usize,
    seed: u64,
) -> KMeansResult {
    assert!(n_clusters >= 1, "n_clusters >= 1");
    assert!(n_clusters <= n, "n_clusters <= n");
    assert_eq!(points.len(), n * dim, "points size mismatch");

    let mut rng = StdRng::seed_from_u64(seed);

    let mut centers = kmeans_pp_init(points, n, dim, n_clusters, &mut rng);
    let mut labels = vec![0usize; n];
    let mut n_iter = 0usize;
    let mut last_inertia = f64::INFINITY;
    for it in 0..max_iter {
        let mut changed = 0usize;
        let mut inertia = 0.0f64;
        for i in 0..n {
            let p = &points[i * dim..(i + 1) * dim];
            let (best, d2) = nearest(p, &centers, n_clusters, dim);
            if best != labels[i] {
                changed += 1;
            }
            labels[i] = best;
            inertia += d2 as f64;
        }
        let mut sums = vec![0.0f64; n_clusters * dim];
        let mut counts = vec![0usize; n_clusters];
        for i in 0..n {
            let p = &points[i * dim..(i + 1) * dim];
            let c = labels[i];
            counts[c] += 1;
            let off = c * dim;
            for k in 0..dim {
                sums[off + k] += p[k] as f64;
            }
        }
        for c in 0..n_clusters {
            if counts[c] == 0 {
                let i = rng.gen_range(0..n);
                let off = c * dim;
                for k in 0..dim {
                    centers[off + k] = points[i * dim + k];
                }
            } else {
                let off = c * dim;
                let cnt = counts[c] as f64;
                for k in 0..dim {
                    centers[off + k] = (sums[off + k] / cnt) as f32;
                }
            }
        }
        n_iter = it + 1;
        let rel = ((last_inertia - inertia).abs() / (last_inertia.abs() + 1e-12)).abs();
        last_inertia = inertia;
        if changed == 0 || rel < 1e-5 {
            break;
        }
    }

    KMeansResult {
        labels,
        centers,
        n_clusters,
        dim,
        inertia: last_inertia,
        n_iter,
    }
}

fn nearest(p: &[f32], centers: &[f32], k: usize, dim: usize) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_d2 = f32::INFINITY;
    for c in 0..k {
        let off = c * dim;
        let mut d2 = 0.0f32;
        for j in 0..dim {
            let diff = p[j] - centers[off + j];
            d2 += diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            best = c;
        }
    }
    (best, best_d2)
}

fn kmeans_pp_init(points: &[f32], n: usize, dim: usize, k: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut centers = Vec::with_capacity(k * dim);
    let first = rng.gen_range(0..n);
    centers.extend_from_slice(&points[first * dim..(first + 1) * dim]);

    let mut closest = vec![f32::INFINITY; n];
    for c in 0..k - 1 {
        let new_center = &centers[c * dim..(c + 1) * dim];
        for i in 0..n {
            let p = &points[i * dim..(i + 1) * dim];
            let mut d2 = 0.0f32;
            for j in 0..dim {
                let d = p[j] - new_center[j];
                d2 += d * d;
            }
            if d2 < closest[i] {
                closest[i] = d2;
            }
        }
        let total: f64 = closest.iter().map(|&x| x as f64).sum();
        if total <= 0.0 {
            // Degenerate: pick random index.
            let mut idxs: Vec<usize> = (0..n).collect();
            idxs.shuffle(rng);
            let pick = idxs[0];
            centers.extend_from_slice(&points[pick * dim..(pick + 1) * dim]);
            continue;
        }
        let mut r: f64 = rng.r#gen::<f64>() * total;
        let mut pick = n - 1;
        for (i, &d2) in closest.iter().enumerate() {
            r -= d2 as f64;
            if r <= 0.0 {
                pick = i;
                break;
            }
        }
        centers.extend_from_slice(&points[pick * dim..(pick + 1) * dim]);
    }
    centers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_two_well_separated_clusters() {
        let mut points = Vec::new();
        for _ in 0..50 {
            points.extend_from_slice(&[0.0, 0.0]);
        }
        for _ in 0..50 {
            points.extend_from_slice(&[10.0, 10.0]);
        }
        let res = kmeans_lloyd(&points, 100, 2, 2, 50, 0);
        let a = res.labels[0];
        let b = res.labels[60];
        assert_ne!(a, b);
        for i in 0..50 {
            assert_eq!(res.labels[i], a);
        }
        for i in 50..100 {
            assert_eq!(res.labels[i], b);
        }
    }
}
