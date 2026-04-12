use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{Array1, Array2};
use rctd_core::{
    build_x_vals, compute_q_matrix, compute_spline_coefficients, device_cpu, run_full_mode,
};

fn bench_small_full_mode(c: &mut Criterion) {
    let n_pixels = 2000usize;
    let n_genes = 100usize;
    let n_types = 6usize;
    let profiles = Array2::from_elem((n_genes, n_types), 1.0 / n_types as f64);
    let counts = Array2::from_elem((n_pixels, n_genes), 10.0);
    let numi = Array1::from_elem(n_pixels, 500.0);
    let x = build_x_vals();
    let q = compute_q_matrix(1.0, &x, 100);
    let sq = compute_spline_coefficients(&q, &x);
    let dev = device_cpu();

    c.bench_function("full_mode_2k_px", |b| {
        b.iter(|| {
            let r = run_full_mode(
                black_box(&counts),
                black_box(&numi),
                black_box(&profiles),
                black_box(&q),
                black_box(&sq),
                black_box(&x),
                512,
                &dev,
                None,
            );
            black_box(r);
        });
    });
}

criterion_group!(benches, bench_small_full_mode);
criterion_main!(benches);
