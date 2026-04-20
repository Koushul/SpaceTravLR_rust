fn main() {
    let d = spacetravlr::run_benchmark_mock_cluster_cnn_training();
    println!("mock_cluster_cnn_wall_s: {:.4}", d.as_secs_f64());
}
