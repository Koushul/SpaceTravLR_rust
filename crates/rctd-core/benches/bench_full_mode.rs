use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use rctd_core::irwls::{solve_irwls_batch_shared_prepared, IrwlsSharedPrepared};
use rctd_core::irwls_native::{solve_irwls_native, NativeSharedPrepared};
use rctd_core::{
    build_x_vals, compute_q_matrix, compute_spline_coefficients, device_cpu, run_full_mode,
};

fn synthetic(n_pixels: usize, n_genes: usize, n_types: usize) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let profiles = Array2::from_elem((n_genes, n_types), 1.0 / n_types as f64);
    let counts = Array2::from_elem((n_pixels, n_genes), 10.0);
    let numi = Array1::from_elem(n_pixels, 500.0);
    (counts, numi, profiles)
}

fn bench_small_full_mode(c: &mut Criterion) {
    let (counts, numi, profiles) = synthetic(2000, 100, 6);
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

fn bench_irwls_native_vs_burn(c: &mut Criterion) {
    let (counts, numi, profiles) = synthetic(256, 80, 6);
    let x = build_x_vals();
    let q = compute_q_matrix(100.0, &x, 100);
    let sq = compute_spline_coefficients(&q, &x);
    let dev = device_cpu();
    let native_prep = NativeSharedPrepared::new(&profiles, &q, &sq, &x);
    let burn_prep = IrwlsSharedPrepared::new(&profiles, &q, &sq, &x, &dev);

    let mut group = c.benchmark_group("irwls_batch_256");
    group.bench_function(BenchmarkId::new("native", 256), |b| {
        b.iter(|| {
            let (w, cnv) = solve_irwls_native(
                black_box(&native_prep),
                black_box(counts.view()),
                black_box(numi.view()),
                50,
                0.001,
                0.3,
                false,
                false,
            );
            black_box((w, cnv));
        });
    });
    group.bench_function(BenchmarkId::new("burn", 256), |b| {
        b.iter(|| {
            let (w, cnv) = solve_irwls_batch_shared_prepared(
                black_box(&burn_prep),
                black_box(counts.view()),
                black_box(numi.view()),
                50,
                0.001,
                0.3,
                false,
                false,
                &dev,
            );
            black_box((w, cnv));
        });
    });
    group.finish();
}

criterion_group!(benches, bench_small_full_mode, bench_irwls_native_vs_burn);
criterion_main!(benches);
