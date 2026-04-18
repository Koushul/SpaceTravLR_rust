//! Lightweight clustering / spatial metrics used by tests and the binary.

use std::collections::HashMap;

/// Adjusted Rand Index between two integer label vectors.
///
/// Sklearn-equivalent implementation.
pub fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n < 2 {
        return 1.0;
    }

    let mut a_map: HashMap<usize, usize> = HashMap::new();
    let mut b_map: HashMap<usize, usize> = HashMap::new();
    for (&x, &y) in a.iter().zip(b.iter()) {
        let na = a_map.len();
        a_map.entry(x).or_insert(na);
        let nb = b_map.len();
        b_map.entry(y).or_insert(nb);
    }
    let n_a = a_map.len();
    let n_b = b_map.len();
    let mut contingency = vec![0u64; n_a * n_b];
    for (&x, &y) in a.iter().zip(b.iter()) {
        let i = a_map[&x];
        let j = b_map[&y];
        contingency[i * n_b + j] += 1;
    }

    let mut sum_comb_c = 0u64;
    for &c in &contingency {
        sum_comb_c += c.saturating_mul(c.saturating_sub(1)) / 2;
    }
    let mut sum_comb_a = 0u64;
    for i in 0..n_a {
        let mut s = 0u64;
        for j in 0..n_b {
            s += contingency[i * n_b + j];
        }
        sum_comb_a += s.saturating_mul(s.saturating_sub(1)) / 2;
    }
    let mut sum_comb_b = 0u64;
    for j in 0..n_b {
        let mut s = 0u64;
        for i in 0..n_a {
            s += contingency[i * n_b + j];
        }
        sum_comb_b += s.saturating_mul(s.saturating_sub(1)) / 2;
    }
    let n_choose_2 = (n as u64).saturating_mul((n as u64).saturating_sub(1)) / 2;
    if n_choose_2 == 0 {
        return 1.0;
    }

    let expected = (sum_comb_a as f64) * (sum_comb_b as f64) / (n_choose_2 as f64);
    let max_index = 0.5 * ((sum_comb_a as f64) + (sum_comb_b as f64));
    let denom = max_index - expected;
    if denom.abs() < 1e-12 {
        return 1.0;
    }
    ((sum_comb_c as f64) - expected) / denom
}

/// Normalized Mutual Information (arithmetic average normalization, the
/// sklearn default).
pub fn normalized_mutual_info(a: &[usize], b: &[usize]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    if n == 0.0 {
        return 1.0;
    }

    let mut a_counts: HashMap<usize, u64> = HashMap::new();
    let mut b_counts: HashMap<usize, u64> = HashMap::new();
    let mut joint: HashMap<(usize, usize), u64> = HashMap::new();
    for (&x, &y) in a.iter().zip(b.iter()) {
        *a_counts.entry(x).or_default() += 1;
        *b_counts.entry(y).or_default() += 1;
        *joint.entry((x, y)).or_default() += 1;
    }

    let h = |counts: &HashMap<usize, u64>| -> f64 {
        let mut h = 0.0f64;
        for &c in counts.values() {
            if c == 0 {
                continue;
            }
            let p = c as f64 / n;
            h -= p * p.ln();
        }
        h
    };
    let h_a = h(&a_counts);
    let h_b = h(&b_counts);
    let mut mi = 0.0f64;
    for (&(x, y), &n_xy) in &joint {
        if n_xy == 0 {
            continue;
        }
        let p_xy = n_xy as f64 / n;
        let p_x = a_counts[&x] as f64 / n;
        let p_y = b_counts[&y] as f64 / n;
        mi += p_xy * (p_xy / (p_x * p_y)).ln();
    }
    let denom = 0.5 * (h_a + h_b);
    if denom < 1e-12 { 1.0 } else { mi / denom }
}

/// Mean fraction of each cell's `k` nearest spatial neighbours that share its
/// predicted label. Higher = niches are spatially contiguous.
pub fn spatial_purity_knn(coords: &[[f64; 2]], labels: &[usize], k: usize) -> f64 {
    assert_eq!(coords.len(), labels.len());
    let n = coords.len();
    if n == 0 {
        return f64::NAN;
    }
    let kk = k.min(n - 1).max(1);
    let pts: Vec<[f64; 2]> = coords.to_vec();
    let tree = kiddo::ImmutableKdTree::<f64, 2>::new_from_slice(&pts);
    let k_query = std::num::NonZero::new(kk + 1).unwrap();
    let mut sum = 0.0f64;
    for i in 0..n {
        let nbrs = tree.nearest_n::<kiddo::SquaredEuclidean>(&coords[i], k_query);
        let mut hits = 0usize;
        let mut total = 0usize;
        for n_ in &nbrs {
            let j = n_.item as usize;
            if j == i {
                continue;
            }
            total += 1;
            if labels[j] == labels[i] {
                hits += 1;
            }
            if total == kk {
                break;
            }
        }
        if total > 0 {
            sum += hits as f64 / total as f64;
        }
    }
    sum / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ari_perfect_match() {
        let a = vec![0, 0, 1, 1, 2, 2];
        let b = vec![5, 5, 9, 9, 7, 7];
        assert!((adjusted_rand_index(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ari_pseudo_random_is_low() {
        // Use a small Xorshift-style PRNG with high-bit extraction so the
        // resulting labels are uncorrelated with `i % 4`.
        let a: Vec<usize> = (0..400).map(|i| i % 4).collect();
        let mut b: Vec<usize> = Vec::with_capacity(400);
        let mut state = 0xdeadbeefcafebabeu64;
        for _ in 0..400 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            b.push(((state >> 32) as usize) % 4);
        }
        let v = adjusted_rand_index(&a, &b);
        assert!(v.abs() < 0.1, "ari for random labels = {v}");
    }

    #[test]
    fn nmi_perfect_match() {
        let a = vec![0, 0, 1, 1, 2, 2];
        let b = vec![5, 5, 9, 9, 7, 7];
        assert!((normalized_mutual_info(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn spatial_purity_perfect_grid() {
        let coords = vec![[0.0, 0.0], [0.0, 1.0], [10.0, 0.0], [10.0, 1.0]];
        let labels = vec![0, 0, 1, 1];
        let p = spatial_purity_knn(&coords, &labels, 1);
        assert!((p - 1.0).abs() < 1e-9, "p = {p}");
    }
}
