use crate::estimator::ClusterTrainingSummary;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub fn write_gene_training_log(
    log_path: &Path,
    gene: &str,
    seed_only: bool,
    epochs: usize,
    learning_rate: f64,
    lasso_n_iter_max: usize,
    lasso_tol: f64,
    summaries: &[ClusterTrainingSummary],
) -> std::io::Result<()> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let f = File::create(log_path)?;
    let mut w = BufWriter::with_capacity(256 * 1024, f);

    writeln!(w, "format\tspacetravlr_training_log\tv1")?;
    writeln!(w, "gene\t{}", gene)?;
    writeln!(w, "seed_only\t{}", seed_only)?;
    writeln!(w, "cnn_epochs_config\t{}", epochs)?;
    writeln!(w, "learning_rate\t{}", learning_rate)?;
    writeln!(w, "lasso_n_iter_max\t{}", lasso_n_iter_max)?;
    writeln!(w, "lasso_tol\t{}", lasso_tol)?;
    writeln!(w)?;

    writeln!(
        w,
        "# summary: cluster_id, n_cells, n_modulators, lasso_r2, lasso_train_mse, lasso_fista_iters, lasso_converged, cnn_epochs_ran, cnn_mse_first, cnn_mse_last"
    )?;
    writeln!(
        w,
        "cluster_id\tn_cells\tn_modulators\tlasso_r2\tlasso_train_mse\tlasso_fista_iters\tlasso_converged\tcnn_epochs_ran\tcnn_mse_first\tcnn_mse_last"
    )?;

    for s in summaries {
        let (ran, first_s, last_s) = if s.cnn_train_mse_epochs.is_empty() {
            (0usize, "NA".to_string(), "NA".to_string())
        } else {
            let v = &s.cnn_train_mse_epochs;
            (
                v.len(),
                format!("{:.6}", v[0]),
                format!("{:.6}", v.last().expect("nonempty")),
            )
        };
        writeln!(
            w,
            "{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{}",
            s.cluster_id,
            s.n_cells,
            s.n_modulators,
            s.lasso_r2,
            s.lasso_train_mse,
            s.lasso_fista_iters,
            s.lasso_converged,
            ran,
            first_s,
            last_s,
        )?;
    }

    writeln!(w)?;
    writeln!(w, "# cnn_mse_by_epoch: cluster_id, epoch, train_mse")?;
    writeln!(w, "cluster_id\tepoch\ttrain_mse")?;
    for s in summaries {
        for (epoch, &mse) in s.cnn_train_mse_epochs.iter().enumerate() {
            writeln!(w, "{}\t{}\t{:.6}", s.cluster_id, epoch, mse)?;
        }
    }

    w.flush()
}
