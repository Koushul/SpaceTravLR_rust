use ndarray::{Array1, Array2};
use rand::{rngs::StdRng, Rng, SeedableRng};

pub fn synthetic_pixel_data(seed: u64) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_genes = 48usize;
    let n_types = 5usize;
    let n_pixels = 28usize;

    let mut profiles = Array2::<f64>::zeros((n_genes, n_types));
    for k in 0..n_types {
        for g in 0..n_genes {
            profiles[[g, k]] = rng.gen::<f64>().exp() * 1e-3;
        }
    }
    let markers_per_type = n_genes / n_types;
    for k in 0..n_types {
        let start = k * markers_per_type;
        let end = (start + markers_per_type).min(n_genes);
        for g in start..end {
            profiles[[g, k]] *= 10.0;
        }
    }
    for k in 0..n_types {
        let sum: f64 = (0..n_genes).map(|g| profiles[[g, k]]).sum();
        for g in 0..n_genes {
            profiles[[g, k]] /= sum;
        }
    }

    let mut counts = Array2::<f64>::zeros((n_pixels, n_genes));
    let mut numi = Array1::<f64>::zeros(n_pixels);
    for i in 0..n_pixels {
        numi[i] = rng.gen_range(200..3000) as f64;
        let mut mix = vec![0f64; n_types];
        let mut s = 0f64;
        for m in &mut mix {
            *m = rng.gen::<f64>();
            s += *m;
        }
        for m in &mut mix {
            *m /= s;
        }
        for g in 0..n_genes {
            let mu: f64 = (0..n_types).map(|k| profiles[[g, k]] * mix[k]).sum::<f64>() * numi[i];
            counts[[i, g]] = mu.floor();
        }
    }

    (counts, numi, profiles)
}

pub fn k_names(k: usize) -> Vec<String> {
    (0..k).map(|i| format!("t{i}")).collect()
}
