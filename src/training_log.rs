use crate::cnn_gating::CnnGateDecision;
use crate::estimator::ClusterTrainingSummary;
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Debug, Clone, Serialize)]
pub struct ClusterTrainingLogRow {
    pub cluster_id: usize,
    pub n_cells: usize,
    pub n_modulators: usize,
    pub lasso_r2: f64,
    pub lasso_train_mse: f64,
    pub lasso_fista_iters: usize,
    pub lasso_converged: bool,
    pub cnn_epochs_ran: usize,
    pub cnn_mse_first: Option<f64>,
    pub cnn_mse_last: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeneTrainingRollup {
    pub gene: String,
    pub seed_only: bool,
    pub per_cell_cnn_export: bool,
    pub cnn_epochs_config: usize,
    pub learning_rate: f64,
    pub lasso_n_iter_max: usize,
    pub lasso_tol: f64,
    pub mean_lasso_r2: f64,
    pub min_lasso_r2: f64,
    pub max_lasso_r2: f64,
    pub frac_lasso_converged: f64,
    pub sum_cnn_epochs_ran: usize,
    pub n_clusters: usize,
    pub gate: Option<HashMap<String, String>>,
    pub clusters: Vec<ClusterTrainingLogRow>,
}

fn parse_f64_cell(s: &str) -> Option<f64> {
    if s == "NA" || s.is_empty() {
        return None;
    }
    s.parse().ok()
}

fn parse_cluster_summary_row(cols: &[&str]) -> Option<ClusterTrainingLogRow> {
    if cols.len() < 10 {
        return None;
    }
    Some(ClusterTrainingLogRow {
        cluster_id: cols[0].parse().ok()?,
        n_cells: cols[1].parse().ok()?,
        n_modulators: cols[2].parse().ok()?,
        lasso_r2: cols[3].parse().ok()?,
        lasso_train_mse: cols[4].parse().ok()?,
        lasso_fista_iters: cols[5].parse().ok()?,
        lasso_converged: cols[6] == "true",
        cnn_epochs_ran: cols[7].parse().ok()?,
        cnn_mse_first: parse_f64_cell(cols[8]),
        cnn_mse_last: parse_f64_cell(cols[9]),
    })
}

pub fn parse_gene_training_log(path: &Path) -> anyhow::Result<Option<GeneTrainingRollup>> {
    let f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };
    let reader = BufReader::new(f);
    let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;
    parse_gene_training_log_lines(&lines)
}

fn parse_gene_training_log_lines(lines: &[String]) -> anyhow::Result<Option<GeneTrainingRollup>> {
    let mut gene = String::new();
    let mut seed_only = false;
    let mut per_cell_cnn_export = false;
    let mut cnn_epochs_config = 0usize;
    let mut learning_rate = 0.0f64;
    let mut lasso_n_iter_max = 0usize;
    let mut lasso_tol = 0.0f64;
    let mut gate: Option<HashMap<String, String>> = None;
    let mut clusters: Vec<ClusterTrainingLogRow> = Vec::new();

    let mut i = 0usize;
    if lines.is_empty() {
        return Ok(None);
    }
    let first = lines[0].trim();
    if first != "format\tspacetravlr_training_log\tv1" {
        return Ok(None);
    }
    i += 1;

    while i < lines.len() {
        let line = lines[i].trim_end();
        if line.is_empty() {
            i += 1;
            continue;
        }
        if line.starts_with("# hybrid_cnn_gate") {
            i += 1;
            let mut g = HashMap::new();
            while i < lines.len() {
                let cur_line = lines[i].trim_end();
                if cur_line.is_empty() {
                    break;
                }
                if cur_line.starts_with('#') {
                    break;
                }
                if let Some((k, v)) = cur_line.split_once('\t') {
                    if k.starts_with("gate_") {
                        g.insert(k.to_string(), v.to_string());
                    }
                }
                i += 1;
            }
            if !g.is_empty() {
                gate = Some(g);
            }
            continue;
        }
        if line.starts_with("# summary:") {
            i += 1;
            if i >= lines.len() {
                break;
            }
            i += 1;
            while i < lines.len() {
                let cur_line = lines[i].trim_end();
                if cur_line.is_empty() {
                    break;
                }
                if cur_line.starts_with('#') {
                    break;
                }
                let cols: Vec<&str> = cur_line.split('\t').collect();
                if let Some(row) = parse_cluster_summary_row(&cols) {
                    clusters.push(row);
                }
                i += 1;
            }
            break;
        }
        if line.starts_with('#') {
            i += 1;
            continue;
        }
        if let Some((k, v)) = line.split_once('\t') {
            match k {
                "gene" => gene = v.to_string(),
                "seed_only" => seed_only = v == "true",
                "per_cell_cnn_export" => per_cell_cnn_export = v == "true",
                "cnn_epochs_config" => cnn_epochs_config = v.parse().unwrap_or(0),
                "learning_rate" => learning_rate = v.parse().unwrap_or(0.0),
                "lasso_n_iter_max" => lasso_n_iter_max = v.parse().unwrap_or(0),
                "lasso_tol" => lasso_tol = v.parse().unwrap_or(0.0),
                _ => {}
            }
        }
        i += 1;
    }

    if gene.is_empty() || clusters.is_empty() {
        return Ok(None);
    }

    let n = clusters.len() as f64;
    let sum_r2: f64 = clusters.iter().map(|c| c.lasso_r2).sum();
    let mean_lasso_r2 = if n > 0.0 { sum_r2 / n } else { 0.0 };
    let min_lasso_r2 = clusters
        .iter()
        .map(|c| c.lasso_r2)
        .fold(f64::INFINITY, f64::min);
    let max_lasso_r2 = clusters
        .iter()
        .map(|c| c.lasso_r2)
        .fold(f64::NEG_INFINITY, f64::max);
    let conv = clusters.iter().filter(|c| c.lasso_converged).count() as f64;
    let frac_lasso_converged = if n > 0.0 { conv / n } else { 0.0 };
    let sum_cnn_epochs_ran: usize = clusters.iter().map(|c| c.cnn_epochs_ran).sum();

    Ok(Some(GeneTrainingRollup {
        gene,
        seed_only,
        per_cell_cnn_export,
        cnn_epochs_config,
        learning_rate,
        lasso_n_iter_max,
        lasso_tol,
        mean_lasso_r2,
        min_lasso_r2,
        max_lasso_r2,
        frac_lasso_converged,
        sum_cnn_epochs_ran,
        n_clusters: clusters.len(),
        gate,
        clusters,
    }))
}

pub fn scan_gene_training_logs(log_dir: &Path) -> anyhow::Result<Vec<GeneTrainingRollup>> {
    let mut out = Vec::new();
    if !log_dir.is_dir() {
        return Ok(out);
    }
    for e in std::fs::read_dir(log_dir)? {
        let e = e?;
        if !e.file_type()?.is_file() {
            continue;
        }
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("log") {
            continue;
        }
        if let Some(r) = parse_gene_training_log(&p)? {
            out.push(r);
        }
    }
    out.sort_by(|a, b| a.gene.to_lowercase().cmp(&b.gene.to_lowercase()));
    Ok(out)
}

pub fn write_gene_training_log(
    log_path: &Path,
    gene: &str,
    seed_only: bool,
    per_cell_cnn_export: bool,
    epochs: usize,
    learning_rate: f64,
    lasso_n_iter_max: usize,
    lasso_tol: f64,
    summaries: &[ClusterTrainingSummary],
    gate: Option<&CnnGateDecision>,
) -> std::io::Result<()> {
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let f = File::create(log_path)?;
    let mut w = BufWriter::with_capacity(256 * 1024, f);

    writeln!(w, "format\tspacetravlr_training_log\tv1")?;
    writeln!(w, "gene\t{}", gene)?;
    writeln!(w, "seed_only\t{}", seed_only)?;
    writeln!(w, "per_cell_cnn_export\t{}", per_cell_cnn_export)?;
    writeln!(w, "cnn_epochs_config\t{}", epochs)?;
    writeln!(w, "learning_rate\t{}", learning_rate)?;
    writeln!(w, "lasso_n_iter_max\t{}", lasso_n_iter_max)?;
    writeln!(w, "lasso_tol\t{}", lasso_tol)?;
    writeln!(w)?;

    if let Some(g) = gate {
        writeln!(w, "# hybrid_cnn_gate (empty use_cnn means non-hybrid or pass-2 full CNN)")?;
        writeln!(w, "gate_use_cnn\t{}", g.use_cnn)?;
        writeln!(w, "gate_reason\t{}", g.reason.replace('\t', " "))?;
        writeln!(
            w,
            "gate_min_cells_per_cluster\t{}",
            g.min_cells_per_cluster
        )?;
        writeln!(w, "gate_n_modulators\t{}", g.n_modulators)?;
        writeln!(w, "gate_n_lr_pairs\t{}", g.n_lr_pairs)?;
        writeln!(w, "gate_n_tfl_pairs\t{}", g.n_tfl_pairs)?;
        writeln!(
            w,
            "gate_modulator_spatial_fraction\t{:.6}",
            g.modulator_spatial_fraction
        )?;
        writeln!(w, "gate_mean_lasso_r2\t{:.6}", g.mean_lasso_r2)?;
        writeln!(w, "gate_all_lasso_converged\t{}", g.all_lasso_converged)?;
        writeln!(w, "gate_moran_i\t{:.8}", g.moran_i)?;
        writeln!(w, "gate_moran_p_value\t{:.8}", g.moran_p_value)?;
        writeln!(w, "gate_moran_permutations\t{}", g.moran_permutations)?;
        writeln!(w, "gate_forced_allowlist\t{}", g.forced_by_allowlist)?;
        writeln!(w, "gate_blocked_skip_list\t{}", g.blocked_by_denylist)?;
        if let Some(m) = g.mean_target_expression {
            writeln!(w, "gate_mean_target_expression\t{:.8}", m)?;
        } else {
            writeln!(w, "gate_mean_target_expression\tNA")?;
        }
        writeln!(w, "gate_rank_score\t{:.6}", g.rank_score)?;
        writeln!(w)?;
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_kit_style_log() {
        let raw = "format\tspacetravlr_training_log\tv1
gene\tKit
seed_only\ttrue
per_cell_cnn_export\tfalse
cnn_epochs_config\t10
learning_rate\t0.0002
lasso_n_iter_max\t100
lasso_tol\t0.0001

# summary: cluster_id, n_cells, n_modulators, lasso_r2, lasso_train_mse, lasso_fista_iters, lasso_converged, cnn_epochs_ran, cnn_mse_first, cnn_mse_last
cluster_id\tn_cells\tn_modulators\tlasso_r2\tlasso_train_mse\tlasso_fista_iters\tlasso_converged\tcnn_epochs_ran\tcnn_mse_first\tcnn_mse_last
0\t1107\t376\t0.284636\t0.000007\t100\tfalse\t0\tNA\tNA
1\t841\t376\t0.011789\t0.000004\t100\tfalse\t0\tNA\tNA
";
        let lines: Vec<String> = raw.lines().map(|s| s.to_string()).collect();
        let r = parse_gene_training_log_lines(&lines)
            .unwrap()
            .expect("rollup");
        assert_eq!(r.gene, "Kit");
        assert!(r.seed_only);
        assert_eq!(r.clusters.len(), 2);
        assert_eq!(r.clusters[0].cluster_id, 0);
        assert_eq!(r.clusters[0].cnn_mse_first, None);
        assert!((r.mean_lasso_r2 - 0.1482125).abs() < 1e-5);
    }
}
